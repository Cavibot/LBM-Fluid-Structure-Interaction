# CUDA Graph Capture Compatibility

## Background

NVIDIA CUDA graph capture (`wp.ScopedCapture` in Warp) records GPU kernel launches
into a replayable graph.  At replay time **only the recorded GPU ops run** — Python
code does not re-execute.  Buffer pointers are baked in at capture time.

This is attractive for high-frequency simulation loops because it eliminates
Python overhead and driver call latency.  Newton's own `sensor_contact` example
uses it for exactly this reason.

## Current Status: Not Compatible

`CollisionPipeline` is **not currently compatible**
with CUDA graph capture.  Four concrete reasons are documented below.

### 1. Python object creation inside `run()`

```python
# CollisionPipeline.run() — called every sim.step()
result = CollisionResult()
result.intra[name] = DomainContacts(domain_name=name, raw=raw)
```

These Python objects are created at capture time and never refreshed.
At replay, `contacts.raw` holds a stale pointer to whatever GPU buffer
existed during capture.  If `model.collide()` ever returns a new allocation
the solver silently reads old data.

### 2. `states.get(name)` dict lookup

```python
state = states.get(name)   # inside _run_intra()
```

Python dict access is resolved once at capture time.  At replay the Python
code does not run again, so the lookup result is permanently baked into
the captured kernel arguments.

### 3. `if contacts is not None` branch in `domain.step()`

```python
raw = contacts.raw if contacts is not None else self._model.collide(state_in)
```

The branch is evaluated in Python at capture time.  Whichever GPU code path
was chosen then is permanently recorded.  The other path is never captured
and can never execute during replay.

### 4. Double-buffer swap in domain step loops

```python
state_in, state_out = state_out, state_in
```

This Python variable swap does not re-execute at replay time.
After the first graph replay, `state_in` and `state_out` are frozen
at their capture-time values, so every subsequent replay reads and writes
the same buffers.

---

## Why `sensor_contact` Works

Newton's `example_sensor_contact.py` is compatible because it avoids every
problem above:

```python
def simulate(self):
    self.state_0.clear_forces()
    self.viewer.apply_forces(self.state_0)
    self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)
```

- **Single buffer** — no Python swap needed.
- **Pre-allocated `self.contacts`** — fixed GPU pointer, written in-place each step.
- **No orchestration layer** — one flat Python → GPU kernel chain.

---

## What Would Be Required

Making `CollisionPipeline` graph-compatible requires architectural changes:

| Problem | Required change |
|---------|-----------------|
| New Python objects each call | Pre-allocate `Contacts` / `CollisionResult` buffers at init; pass as output targets |
| Python dict dispatch loop | Replace `for name, domain in ...` with a fixed, pre-wired call sequence built at `pipeline.build()` time |
| `contacts is not None` branch | Always take the same code path — decided once at pipeline setup, never branched at runtime |
| Double-buffer swap | Perform the swap **outside** the captured region, or switch to single-buffer inside the capture scope |

The cleanest approach is a **`build()` / `step_captured()` split**:

```python
# At init time (outside capture):
pipeline.build()          # pre-allocates buffers, wires fixed call chain
graph = pipeline.capture(states_in, states_out, dt)  # records GPU ops

# Per frame (pure GPU replay):
wp.capture_launch(graph)  # zero Python overhead
```

This mirrors the pattern Newton's own solvers use for graph capture and is
the target design for a future `CaptureableCollisionPipeline`.

---

## Priority

Low for the current phase.  The immediate goal of `CollisionPipeline` is
**isolation and centralization**, not performance.  Graph capture becomes
relevant when per-step Python overhead is measurable — typically at very
high frame rates (> 1 kHz) or with large numbers of domains.

Track as Phase 3 work alongside the WanPhys geometry dispatch layer.
