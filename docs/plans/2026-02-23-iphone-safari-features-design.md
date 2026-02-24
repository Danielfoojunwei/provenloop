# iPhone Safari — Missing Features Design

## Scope

Add two missing features to the existing `demonstrator/frontend/` web app:
1. Encryption Pipeline Visualization
2. Expert Routing Display

No server changes needed — all data is already streamed via WebSocket.

## Feature 1: Encryption Pipeline Visualization

**Location:** New `#pipeline-bar` div between `#he-breakdown` and `#chat-area`.

**Layout:** 5 horizontal stages connected by arrow separators:
```
[Plaintext] → [Encrypt] → [HE Compute] → [Decrypt] → [Response]
  (idle)       12.3ms       45.7ms         6.8ms       (done)
```

**Behavior:**
- All stages start dim/inactive (dark surface color)
- When query is sent: `Plaintext` lights up (accent color)
- On `input_info` WS message: `Encrypt` lights up with encrypt_time_ms
- On each `token` WS message: `HE Compute` pulses with compute_ms, `Decrypt` shows decrypt_ms
- On `done` WS message: `Response` lights up green, all timings finalize
- After 3s idle: stages fade back to dim

**CSS:** `.pipe-stage` with `.active` class toggle. `background 0.3s, color 0.3s` transitions. Arrow via `::after` pseudo-element.

## Feature 2: Expert Routing Display

**Location:** New `#routing-bar` div directly below `#pipeline-bar`.

**Layout:** 3 pill badges in a flex row:
```
[Banking ○]  [Investment ●]  [Shared ◆]
```

**Behavior:**
- All experts shown as dim pills by default
- On `input_info`: active_expert pill highlighted (accent background)
- `shared_attention` always shown as active
- On `done`: expert distribution counts displayed

**CSS:** `.route-pill` with border-radius, `.active` gets accent background.

## Files Modified

- `demonstrator/frontend/index.html` — add pipeline-bar and routing-bar divs
- `demonstrator/frontend/styles.css` — add pipeline and routing styles
- `demonstrator/frontend/app.js` — add updatePipeline() and updateRouting() functions
