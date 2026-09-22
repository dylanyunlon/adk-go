// Copyright 2025 Google LLC
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

package agent

// StreamingMode defines the streaming mode for agent execution.
type StreamingMode string

const (
	// StreamingModeNone indicates no streaming.
	StreamingModeNone StreamingMode = "none"
	// StreamingModeSSE enables server-sent events streaming, one-way, where
	// LLM response parts are streamed immediately as they are generated.
	StreamingModeSSE StreamingMode = "sse"
)

// RunConfig controls runtime behavior of an agent.
type RunConfig struct {
	// StreamingMode defines the streaming mode for an agent.
	StreamingMode StreamingMode
	// If true, ADK runner will save each part of the user input that is a blob
	// (e.g., images, files) as an artifact.
	SaveInputBlobsAsArtifacts bool

	// DeadlineBudgetEnabled, when true, makes the runner reserve a fraction
	// of the caller's context deadline for a closing model turn. When the
	// budget is exhausted the run stops starting new tool calls, tells any
	// still-running tool that its time is up, and asks the model to produce a
	// partial answer that states what was completed and what was not.
	//
	// When the caller's context carries no deadline, or when this field is
	// false (the default), behavior is unchanged: no budget is tracked and no
	// wind-down happens.
	DeadlineBudgetEnabled bool
}
