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

package deadlinebudget

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestNewNoDeadline(t *testing.T) {
	t.Parallel()
	b := New(context.Background())
	if b.HasDeadline() {
		t.Fatal("Budget from a context without deadline reports HasDeadline == true")
	}
	if b.Exhausted() {
		t.Fatal("no-deadline budget reports Exhausted")
	}
	if b.Reserve() != 0 {
		t.Fatalf("no-deadline budget Reserve = %v, want 0", b.Reserve())
	}
	if r := b.Remaining(); r < time.Hour {
		t.Fatalf("no-deadline budget Remaining = %v, want >> 1h", r)
	}
	if inst := b.WindDownInstruction(); inst != "" {
		t.Fatalf("no-deadline budget WindDownInstruction = %q, want empty", inst)
	}
}

func TestNewWithDeadline(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	b := New(ctx)
	if !b.HasDeadline() {
		t.Fatal("Budget from a context with deadline reports HasDeadline == false")
	}
	// 2 min * 0.15 = 18s, above MinReserve, below MaxReserve.
	wantReserve := time.Duration(float64(2*time.Minute) * DefaultReserve)
	got := b.Reserve()
	if diff := got - wantReserve; diff < -100*time.Millisecond || diff > 100*time.Millisecond {
		t.Fatalf("Reserve = %v, want ~%v", got, wantReserve)
	}
}

func TestReserveClampedToMin(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	b := New(ctx)
	if b.Reserve() != MinReserve {
		t.Fatalf("short deadline Reserve = %v, want MinReserve %v", b.Reserve(), MinReserve)
	}
}

func TestReserveClampedToMax(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), 24*time.Hour)
	defer cancel()
	b := New(ctx)
	if b.Reserve() != MaxReserve {
		t.Fatalf("long deadline Reserve = %v, want MaxReserve %v", b.Reserve(), MaxReserve)
	}
}

func TestExhaustedTransition(t *testing.T) {
	t.Parallel()
	// 6s total, reserve = MinReserve = 5s, remaining = ~1s. Not exhausted.
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
	defer cancel()
	b := New(ctx)
	if b.Exhausted() {
		t.Fatal("budget with 1s remaining should not be exhausted yet")
	}
	// 4s total, reserve = MinReserve = 5s, remaining = -1s. Exhausted.
	ctx2, cancel2 := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel2()
	b2 := New(ctx2)
	if !b2.Exhausted() {
		t.Fatal("budget where reserve > total should be exhausted")
	}
}

func TestToolDeadlineNoDeadline(t *testing.T) {
	t.Parallel()
	b := New(context.Background())
	parent := context.Background()
	got, cancel := b.ToolDeadline(parent)
	defer cancel()
	if got != parent {
		t.Fatal("no-deadline ToolDeadline should return parent unchanged")
	}
}

func TestToolDeadlineShortensTight(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	b := New(ctx)

	toolCtx, toolCancel := b.ToolDeadline(context.Background())
	defer toolCancel()

	dl, ok := toolCtx.Deadline()
	if !ok {
		t.Fatal("ToolDeadline did not set a deadline")
	}
	remaining := time.Until(dl)
	// 2min - 18s reserve = ~102s
	if remaining > 2*time.Minute || remaining < 90*time.Second {
		t.Fatalf("tool deadline remaining = %v, want ~102s", remaining)
	}
}

func TestToolDeadlinePreservesTighterParent(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	b := New(ctx)

	parentCtx, parentCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer parentCancel()

	toolCtx, toolCancel := b.ToolDeadline(parentCtx)
	defer toolCancel()

	dl, ok := toolCtx.Deadline()
	if !ok {
		t.Fatal("ToolDeadline did not set a deadline")
	}
	remaining := time.Until(dl)
	if remaining > 31*time.Second {
		t.Fatalf("tool deadline remaining = %v, want ~30s (parent's tighter deadline)", remaining)
	}
}

func TestRecordAndWindDown(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	b := New(ctx)

	b.RecordCompleted("fetched user profile")
	b.RecordCompleted("queried order history")
	b.RecordAborted("payment lookup (timed out)")

	inst := b.WindDownInstruction()
	for _, want := range []string{
		"final response NOW",
		"Do NOT call any more tools",
		"fetched user profile",
		"queried order history",
		"payment lookup (timed out)",
	} {
		if !strings.Contains(inst, want) {
			t.Fatalf("wind-down instruction missing %q", want)
		}
	}
}

func TestWindDownNoSteps(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	b := New(ctx)
	inst := b.WindDownInstruction()
	if !strings.Contains(inst, "No tool calls were completed") {
		t.Fatal("wind-down with no steps should say so")
	}
}

func TestContextRoundTrip(t *testing.T) {
	t.Parallel()
	b := New(context.Background())
	ctx := ToContext(context.Background(), b)
	got := FromContext(ctx)
	if got != b {
		t.Fatal("FromContext did not return the Budget stored with ToContext")
	}
}

func TestFromContextNil(t *testing.T) {
	t.Parallel()
	got := FromContext(context.Background())
	if got != nil {
		t.Fatal("FromContext on a context without a Budget should return nil")
	}
}

func TestRemainingDecreases(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	b := New(ctx)
	r1 := b.Remaining()
	time.Sleep(10 * time.Millisecond)
	r2 := b.Remaining()
	if r2 >= r1 {
		t.Fatalf("Remaining did not decrease: %v then %v", r1, r2)
	}
}

func TestRecordConcurrentSafety(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	b := New(ctx)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 100; i++ {
			b.RecordCompleted("step")
		}
	}()
	for i := 0; i < 100; i++ {
		b.RecordAborted("step")
	}
	<-done
	_ = b.WindDownInstruction()
}

func TestCompletedAbortedSnapshots(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	b := New(ctx)

	b.RecordCompleted("a")
	b.RecordCompleted("b")
	b.RecordAborted("c")

	c := b.Completed()
	a := b.Aborted()
	if len(c) != 2 || c[0] != "a" || c[1] != "b" {
		t.Fatalf("Completed = %v, want [a b]", c)
	}
	if len(a) != 1 || a[0] != "c" {
		t.Fatalf("Aborted = %v, want [c]", a)
	}

	// mutating the snapshot must not affect the budget
	c[0] = "z"
	if b.Completed()[0] != "a" {
		t.Fatal("mutating Completed snapshot affected the budget")
	}
}
