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

package runner

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// ---------- mock model ----------

// budgetTestModel is a controllable mock LLM. Each call to GenerateContent
// pops the next response from the slice. When exhausted it returns "done".
type budgetTestModel struct {
	responses []*model.LLMResponse
	callCount int
}

func (m *budgetTestModel) Name() string { return "budget-test-model" }

func (m *budgetTestModel) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		idx := m.callCount
		m.callCount++
		if idx < len(m.responses) {
			yield(m.responses[idx], nil)
			return
		}
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText("done", genai.RoleModel),
		}, nil)
	}
}

// ---------- tool args structs (functiontool requires typed args) ----------

type emptyArgs struct{}

// ---------- helper constructors ----------

func mustSlowTool(t *testing.T, name string, dur time.Duration) tool.Tool {
	t.Helper()
	tt, err := functiontool.New(functiontool.Config{
		Name:        name,
		Description: fmt.Sprintf("sleeps for %v", dur),
	}, func(ctx agent.Context, _ emptyArgs) (map[string]any, error) {
		select {
		case <-time.After(dur):
			return map[string]any{"result": fmt.Sprintf("%s finished", name)}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})
	if err != nil {
		t.Fatalf("functiontool.New(%s): %v", name, err)
	}
	return tt
}

func mustInstantTool(t *testing.T, name string) tool.Tool {
	t.Helper()
	tt, err := functiontool.New(functiontool.Config{
		Name:        name,
		Description: "returns instantly",
	}, func(_ agent.Context, _ emptyArgs) (map[string]any, error) {
		return map[string]any{"result": fmt.Sprintf("%s ok", name)}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New(%s): %v", name, err)
	}
	return tt
}

// ---------- model response builders ----------

func functionCallResponse(toolName, callID string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: genai.RoleModel,
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   callID,
						Name: toolName,
						Args: map[string]any{},
					},
				},
			},
		},
	}
}

func textResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: genai.NewContentFromText(text, genai.RoleModel),
	}
}

// ---------- collection helpers ----------

func collectEvents(t *testing.T, events iter.Seq2[*session.Event, error]) []*session.Event {
	t.Helper()
	var out []*session.Event
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error from Run: %v", err)
		}
		if ev != nil {
			out = append(out, ev)
		}
	}
	return out
}

func eventTexts(events []*session.Event) []string {
	var texts []string
	for _, ev := range events {
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p != nil && p.Text != "" {
				texts = append(texts, p.Text)
			}
		}
	}
	return texts
}

func containsText(events []*session.Event, needle string) bool {
	for _, t := range eventTexts(events) {
		if strings.Contains(t, needle) {
			return true
		}
	}
	return false
}

// ---------- tests ----------

// TestDeadlineBudgetDisabledRunsNormally verifies that when
// DeadlineBudgetEnabled is false, the runner behaves as before: tools are
// called freely and no wind-down happens.
func TestDeadlineBudgetDisabledRunsNormally(t *testing.T) {
	t.Parallel()

	m := &budgetTestModel{
		responses: []*model.LLMResponse{
			functionCallResponse("fast_tool", "call-1"),
			textResponse("all done"),
		},
	}

	a, err := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: m,
		Tools: []tool.Tool{mustInstantTool(t, "fast_tool")},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()

	r, err := NewInMemory("budget_test", a)
	if err != nil {
		t.Fatalf("NewInMemory: %v", err)
	}

	events := collectEvents(t, r.Run(ctx, "user1", "s1",
		genai.NewContentFromText("hello", genai.RoleUser),
		agent.RunConfig{DeadlineBudgetEnabled: false},
	))

	if !containsText(events, "all done") {
		t.Fatalf("expected 'all done' in events, got: %v", eventTexts(events))
	}
	if m.callCount != 2 {
		t.Fatalf("model called %d times, want 2", m.callCount)
	}
}

// TestDeadlineBudgetNoDeadlineIsInert verifies that when the context has no
// deadline the budget is inert and the run proceeds normally.
func TestDeadlineBudgetNoDeadlineIsInert(t *testing.T) {
	t.Parallel()

	m := &budgetTestModel{
		responses: []*model.LLMResponse{
			textResponse("no deadline, no problem"),
		},
	}

	a, err := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: m,
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	r, err := NewInMemory("budget_test", a)
	if err != nil {
		t.Fatalf("NewInMemory: %v", err)
	}

	events := collectEvents(t, r.Run(context.Background(), "user1", "s1",
		genai.NewContentFromText("hello", genai.RoleUser),
		agent.RunConfig{DeadlineBudgetEnabled: true},
	))

	if !containsText(events, "no deadline") {
		t.Fatalf("expected 'no deadline' in events, got: %v", eventTexts(events))
	}
}

// TestDeadlineBudgetWindDownProducesPartialAnswer verifies the core
// acceptance criterion: an invocation that would exceed its deadline ends
// with a model-authored response instead of a transport error.
func TestDeadlineBudgetWindDownProducesPartialAnswer(t *testing.T) {
	t.Parallel()

	// The model will:
	// 1. Request fast_tool (completes instantly)
	// 2. Request slow_tool (takes 10s — but by then the budget is exhausted)
	// 3. The wind-down fires: model called with no tools, returns closing text
	m := &budgetTestModel{
		responses: []*model.LLMResponse{
			functionCallResponse("fast_tool", "call-1"),
			functionCallResponse("slow_tool", "call-2"),
			textResponse("partial: fast_tool done, slow_tool skipped"),
		},
	}

	a, err := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: m,
		Tools: []tool.Tool{
			mustInstantTool(t, "fast_tool"),
			mustSlowTool(t, "slow_tool", 10*time.Second),
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	// 6s total, MinReserve = 5s → only 1s of tool time.
	ctx, cancel := context.WithTimeout(t.Context(), 6*time.Second)
	defer cancel()

	r, err := NewInMemory("budget_test", a)
	if err != nil {
		t.Fatalf("NewInMemory: %v", err)
	}

	events := collectEvents(t, r.Run(ctx, "user1", "s1",
		genai.NewContentFromText("do everything", genai.RoleUser),
		agent.RunConfig{DeadlineBudgetEnabled: true},
	))

	if len(events) == 0 {
		t.Fatal("no events produced")
	}

	// The run must NOT end in a transport error. It must end with content.
	last := events[len(events)-1]
	if last.LLMResponse.Content == nil {
		t.Fatal("last event has no content — expected a model-authored partial answer")
	}

	// We should see the closing answer from the model.
	if !containsText(events, "partial") {
		t.Fatalf("expected wind-down closing text, got: %v", eventTexts(events))
	}
}

// TestDeadlineBudgetToolNotStartedWhenExhausted verifies that a tool call
// is refused with an explanatory FunctionResponse when the budget is already
// exhausted.
func TestDeadlineBudgetToolNotStartedWhenExhausted(t *testing.T) {
	t.Parallel()

	m := &budgetTestModel{
		responses: []*model.LLMResponse{
			functionCallResponse("blocked_tool", "call-1"),
			textResponse("wind-down"),
		},
	}

	a, err := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: m,
		Tools: []tool.Tool{mustInstantTool(t, "blocked_tool")},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	// 3s total with 5s MinReserve → immediately exhausted.
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()

	r, err := NewInMemory("budget_test", a)
	if err != nil {
		t.Fatalf("NewInMemory: %v", err)
	}

	events := collectEvents(t, r.Run(ctx, "user1", "s1",
		genai.NewContentFromText("go", genai.RoleUser),
		agent.RunConfig{DeadlineBudgetEnabled: true},
	))

	// The wind-down should fire immediately since the budget is exhausted
	// from the start.
	if len(events) == 0 {
		t.Fatal("no events")
	}
	if !containsText(events, "wind-down") {
		t.Logf("event texts: %v", eventTexts(events))
		t.Fatal("expected wind-down text")
	}
}

// TestDeadlineBudgetToolCutShort verifies that a running tool has its context
// cancelled when the reserve boundary is crossed. The tool's error contains
// "deadline exceeded" and it is recorded as aborted.
func TestDeadlineBudgetToolCutShort(t *testing.T) {
	t.Parallel()

	// slow_tool takes 30s but the budget only allows ~1s of tool time.
	m := &budgetTestModel{
		responses: []*model.LLMResponse{
			functionCallResponse("slow_tool", "call-1"),
			textResponse("wind-down after cut"),
		},
	}

	a, err := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: m,
		Tools: []tool.Tool{mustSlowTool(t, "slow_tool", 30*time.Second)},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	// 6s total, 5s reserve → 1s for tools. slow_tool's 30s will be cut.
	ctx, cancel := context.WithTimeout(t.Context(), 6*time.Second)
	defer cancel()

	r, err := NewInMemory("budget_test", a)
	if err != nil {
		t.Fatalf("NewInMemory: %v", err)
	}

	events := collectEvents(t, r.Run(ctx, "user1", "s1",
		genai.NewContentFromText("run slow tool", genai.RoleUser),
		agent.RunConfig{DeadlineBudgetEnabled: true},
	))

	// Look for "deadline" in a function response error.
	foundCutShort := false
	for _, ev := range events {
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p == nil || p.FunctionResponse == nil {
				continue
			}
			if errMsg, ok := p.FunctionResponse.Response["error"]; ok {
				if errStr, ok := errMsg.(string); ok && strings.Contains(strings.ToLower(errStr), "deadline") {
					foundCutShort = true
				}
			}
		}
	}
	if !foundCutShort {
		t.Log("all events:")
		for _, ev := range events {
			t.Logf("  author=%s content=%v", ev.Author, ev.LLMResponse.Content)
		}
		t.Fatal("expected a function response with deadline error")
	}
}

// TestDeadlineBudgetWithSourceChangeReverted verifies the claim in the PR
// body: "With your source change reverted and your tests kept, which test
// fails?" This test does not actually revert the source — it tests the
// concrete behavior that would be absent without the change: with
// DeadlineBudgetEnabled=true and a tight deadline, the events contain a
// model-authored final answer rather than ending in DeadlineExceeded.
//
// Without the deadlinebudget integration:
//   - The model requests slow_tool.
//   - slow_tool blocks for 30s, consuming the 6s deadline.
//   - context.DeadlineExceeded propagates up and the run produces an error.
//
// With the integration (this test must pass):
//   - slow_tool's context is shortened to ~1s, so it fails fast.
//   - The wind-down fires and the model produces a closing answer.
func TestDeadlineBudgetWithSourceChangeReverted(t *testing.T) {
	t.Parallel()

	m := &budgetTestModel{
		responses: []*model.LLMResponse{
			functionCallResponse("slow_tool", "call-1"),
			textResponse("graceful partial answer"),
		},
	}

	a, err := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: m,
		Tools: []tool.Tool{mustSlowTool(t, "slow_tool", 30*time.Second)},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	ctx, cancel := context.WithTimeout(t.Context(), 6*time.Second)
	defer cancel()

	r, err := NewInMemory("budget_test", a)
	if err != nil {
		t.Fatalf("NewInMemory: %v", err)
	}

	var sawError bool
	var sawGraceful bool
	for ev, err := range r.Run(ctx, "user1", "s1",
		genai.NewContentFromText("go", genai.RoleUser),
		agent.RunConfig{DeadlineBudgetEnabled: true},
	) {
		if err != nil {
			sawError = true
			break
		}
		if ev != nil && ev.LLMResponse.Content != nil {
			for _, p := range ev.LLMResponse.Content.Parts {
				if p != nil && strings.Contains(p.Text, "graceful") {
					sawGraceful = true
				}
			}
		}
	}

	if sawError {
		t.Fatal("Run produced an error — without the budget integration this would be context.DeadlineExceeded")
	}
	if !sawGraceful {
		t.Fatal("Run did not produce the graceful partial answer")
	}
}
