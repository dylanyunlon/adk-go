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

// Behavioral contract for callers of RunConfig.DeadlineBudgetEnabled:
//
//   - When DeadlineBudgetEnabled is false (the default), or when the caller's
//     context carries no deadline, behavior is identical to the pre-existing
//     framework: no budget is tracked, no wind-down happens, and tools run
//     until the context expires or the model produces a final response.
//
//   - When DeadlineBudgetEnabled is true AND the context carries a deadline,
//     the runner reserves 15% of the total budget (clamped to [5s, 60s]) for
//     the model's closing turn. The remaining time is available for tool
//     calls. When the remaining time crosses the reserve boundary:
//
//       1. New tool calls are refused. The function response says the tool
//          was not started and the invocation needs to produce a partial
//          answer.
//
//       2. A tool that is already running has its context cancelled at the
//          reserve boundary. The tool's error is recorded as "cut short."
//
//       3. The model is called one more time with no tools and an instruction
//          listing what was completed and what was cut short. The model's
//          response is the invocation's final answer.
//
//   - The wind-down instruction is prepended to the existing system
//     instruction, not appended, because the model should see the urgency
//     directive before the agent's persona. The existing instruction is
//     preserved in full.
//
//   - The reserve is computed once at invocation start, not at each step.
//     This means the fraction is measured against the original deadline,
//     not against whatever time happens to remain when the check runs. A
//     consequence is that a very fast tool does not "earn back" time for
//     another tool: the boundary is fixed.
//
//   - A goroutine cannot be cancelled from outside in Go. A tool that does
//     not observe its context cannot be stopped, only abandoned, and
//     abandoning it leaks whatever it holds until it returns on its own.
//     ToolDeadline shortens the context; it does not (and cannot) forcibly
//     stop a tool.
package deadlinebudget
