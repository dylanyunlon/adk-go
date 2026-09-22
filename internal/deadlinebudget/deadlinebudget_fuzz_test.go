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
	"testing"
	"time"
)

// FuzzBudgetReserve verifies that the reserve computation never panics and
// always lands in [MinReserve, MaxReserve] for any positive duration.
func FuzzBudgetReserve(f *testing.F) {
	f.Add(int64(0))
	f.Add(int64(1))
	f.Add(int64(time.Second))
	f.Add(int64(time.Minute))
	f.Add(int64(time.Hour))
	f.Add(int64(24 * time.Hour))
	f.Add(int64(365 * 24 * time.Hour))

	f.Fuzz(func(t *testing.T, ns int64) {
		if ns <= 0 {
			return // negative or zero durations are not meaningful deadlines
		}
		dur := time.Duration(ns)
		ctx, cancel := context.WithTimeout(context.Background(), dur)
		defer cancel()
		b := New(ctx)
		if !b.HasDeadline() {
			t.Fatal("budget from a context with deadline reports HasDeadline == false")
		}
		r := b.Reserve()
		if r < MinReserve {
			t.Fatalf("reserve %v < MinReserve %v for duration %v", r, MinReserve, dur)
		}
		if r > MaxReserve {
			t.Fatalf("reserve %v > MaxReserve %v for duration %v", r, MaxReserve, dur)
		}
	})
}

// FuzzWindDownInstruction verifies that WindDownInstruction never panics
// regardless of the strings recorded.
func FuzzWindDownInstruction(f *testing.F) {
	f.Add("completed step", "aborted step")
	f.Add("", "")
	f.Add("a\x00b", "c\x00d")
	f.Add("very long "+string(make([]byte, 4096)), "")

	f.Fuzz(func(t *testing.T, completed, aborted string) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		b := New(ctx)
		b.RecordCompleted(completed)
		b.RecordAborted(aborted)
		inst := b.WindDownInstruction()
		if inst == "" {
			t.Fatal("wind-down instruction should not be empty when budget has deadline")
		}
	})
}
