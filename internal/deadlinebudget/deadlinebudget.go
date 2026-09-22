// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package deadlinebudget implements time-budget management for agent
// invocations, allowing an invocation to wind down gracefully before a hard
// deadline and produce a partial answer instead of a transport error.
//
// An invocation runs inside whatever deadline the caller's context carries.
// When that deadline passes mid-invocation, the run ends in
// context.DeadlineExceeded and the caller receives a transport error rather
// than a partial answer. A [Budget] monitors the caller's deadline (if any)
// and reserves a runway for the model to write a closing response.
//
// When the caller's context carries no deadline, every method reports "no
// constraint" and the framework behaves exactly as before.
//
// Integration points (the budget is created once by the runner and read by the
// flow):
//
//   - base_flow.go's Run loop checks [Budget.Exhausted] at the top of each
//     iteration. When true it issues a wind-down turn instead of another
//     tool-calling step.
//   - handleFunctionCalls checks [Budget.Exhausted] before dispatching each
//     call. A call that would start past the boundary is refused with an
//     explanatory FunctionResponse rather than a transport error.
//   - handleFunctionCalls applies [Budget.ToolDeadline] to each allowed call
//     so a running tool cannot consume the reserve.
//   - The flow records completed and aborted tools via [Budget.RecordCompleted]
//     and [Budget.RecordAborted], and [Budget.WindDownInstruction] feeds them
//     into the model's closing turn.
package deadlinebudget

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// DefaultReserve is the fraction of the total budget reserved for the model's
// closing turn.
const DefaultReserve = 0.15

// MinReserve is the smallest absolute reserve the budget will use. A very
// short deadline gets its full reserve as MinReserve, which may consume the
// entire budget, in that case the invocation winds down immediately. This
// is intentional: a budget shorter than one model call has no room for tools.
const MinReserve = 5 * time.Second

// MaxReserve caps the absolute reserve so a very long deadline does not set
// aside more time than a model call could possibly use.
const MaxReserve = 60 * time.Second

// Budget tracks the time remaining in an invocation and decides when to stop
// taking on new tool calls. A zero-value Budget (no deadline) reports
// Exhausted() == false and ToolDeadline returns the parent unchanged.
type Budget struct {
	hasDeadline bool
	deadline    time.Time
	reserve     time.Duration

	mu        sync.Mutex
	completed []string
	aborted   []string
}

// New creates a Budget from the caller's context. When ctx carries no
// deadline the returned Budget is inert.
func New(ctx context.Context) *Budget {
	dl, ok := ctx.Deadline()
	if !ok {
		return &Budget{}
	}
	total := time.Until(dl)
	reserve := time.Duration(float64(total) * DefaultReserve)
	if reserve < MinReserve {
		reserve = MinReserve
	}
	if reserve > MaxReserve {
		reserve = MaxReserve
	}
	return &Budget{
		hasDeadline: true,
		deadline:    dl,
		reserve:     reserve,
	}
}

// HasDeadline reports whether the budget was created from a context that
// carried a deadline.
func (b *Budget) HasDeadline() bool { return b.hasDeadline }

// Reserve returns the duration reserved for the closing model turn. Zero
// when the budget has no deadline.
func (b *Budget) Reserve() time.Duration {
	if !b.hasDeadline {
		return 0
	}
	return b.reserve
}

// Exhausted reports whether the budget has been consumed to the point where
// no new tool call should be started.
func (b *Budget) Exhausted() bool {
	if !b.hasDeadline {
		return false
	}
	return time.Until(b.deadline) <= b.reserve
}

// Remaining returns how much time is left before the reserve boundary.
// Negative means the reserve has already been entered. A zero-value Budget
// returns a large positive duration.
func (b *Budget) Remaining() time.Duration {
	if !b.hasDeadline {
		return 1<<62 - 1
	}
	return time.Until(b.deadline) - b.reserve
}

// ToolDeadline returns a context whose deadline is shortened so the tool
// cannot run past the reserve boundary. When the budget has no deadline the
// parent is returned unchanged (cancel is a no-op). When the parent already
// has a tighter deadline, that tighter deadline is kept.
func (b *Budget) ToolDeadline(parent context.Context) (context.Context, context.CancelFunc) {
	if !b.hasDeadline {
		return parent, func() {}
	}
	toolDL := b.deadline.Add(-b.reserve)
	if parentDL, ok := parent.Deadline(); ok && parentDL.Before(toolDL) {
		return parent, func() {}
	}
	return context.WithDeadline(parent, toolDL)
}

// RecordCompleted records a tool or step that finished successfully.
func (b *Budget) RecordCompleted(description string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.completed = append(b.completed, description)
}

// RecordAborted records a tool call that was cut short or not started.
func (b *Budget) RecordAborted(description string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.aborted = append(b.aborted, description)
}

// Completed returns a snapshot of the completed-step descriptions.
func (b *Budget) Completed() []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]string, len(b.completed))
	copy(out, b.completed)
	return out
}

// Aborted returns a snapshot of the aborted-step descriptions.
func (b *Budget) Aborted() []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]string, len(b.aborted))
	copy(out, b.aborted)
	return out
}

// WindDownInstruction returns the system-instruction text to prepend when
// asking the model to produce a closing turn. Returns "" when the budget
// has no deadline.
func (b *Budget) WindDownInstruction() string {
	if !b.hasDeadline {
		return ""
	}
	completed := b.Completed()
	aborted := b.Aborted()

	var sb strings.Builder
	sb.WriteString("You are running out of time. You MUST produce a final response NOW.\n")
	sb.WriteString("Do NOT call any more tools. Summarize what has been accomplished and what remains.\n")

	if len(completed) > 0 {
		sb.WriteString("\nCompleted steps:\n")
		for _, s := range completed {
			fmt.Fprintf(&sb, "- %s\n", s)
		}
	}
	if len(aborted) > 0 {
		sb.WriteString("\nAborted (cut short due to time):\n")
		for _, s := range aborted {
			fmt.Fprintf(&sb, "- %s\n", s)
		}
	}
	if len(completed) == 0 && len(aborted) == 0 {
		sb.WriteString("\nNo tool calls were completed before the deadline.\n")
	}
	return sb.String()
}

type contextKey struct{}

// ToContext stores a Budget in the context.
func ToContext(ctx context.Context, b *Budget) context.Context {
	return context.WithValue(ctx, contextKey{}, b)
}

// FromContext retrieves the Budget from the context, or nil when none is
// stored.
func FromContext(ctx context.Context) *Budget {
	b, _ := ctx.Value(contextKey{}).(*Budget)
	return b
}
