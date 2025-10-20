# TTT Surrogate Gradient Analysis

## Date
2025-10-21

## Problem Statement
The current TTT surrogate gradient implementation (`ueaj/model/ttt/impl.py`) produces incorrect gradients for the key input (dk). While dq and dv gradients work well, dk gradients have cosine similarity oscillating around 0, indicating they are essentially random noise.

## Code Structure

### Forward Pass (`make_scan_fn`, lines 15-29)
The TTT forward pass uses a self-supervised reconstruction objective:

```python
def make_scan_fn(fwd_fn, distance=True, n_iters=1, wd=.1, lr=.01):
    def update_fn(state, x):
        k, v, q = x
        o = fwd_fn(state, q)  # Output from query

        for _ in range(n_iters):
            v_pred, dstate_fn = jax.vjp(lambda state: fwd_fn(state, k), state)
            dv = (v - v_pred) if distance else v  # Reconstruction target
            dstate, = dstate_fn(dv)  # Gradient of reconstruction loss

            # SGD update with weight decay
            state = jax.tree.map(lambda a, b: (1-wd*lr)*a + lr*jax.nn.tanh(b), state, dstate)

        return state, o
    return update_fn
```

**Key insight**: State is updated by minimizing reconstruction error: `||fwd_fn(state, k) - v||²`

### Backward Pass (`_ttt_bwd`, lines 59-96)

The backward pass has two phases:

#### Phase 1: Compute dq (lines 63-74) - **WORKS CORRECTLY**

```python
def q_scan(carry, x):
    k, v, q, do = x
    state, dstate = carry

    # Differentiate through forward scan
    (new_state, o), q_update_jvp = jax.vjp(
        lambda state, q: fwd_scan(state, (k, v, q)),
        state, q
    )
    new_dstate, dq = q_update_jvp((dstate, do))

    return (new_state, new_dstate), (o, dq)
```

**Why it works**:
- Uses true BPTT through `fwd_scan` w.r.t both `state` and `q`
- Properly accumulates `dstate` (gradient w.r.t state)
- Output gradient `do` is backpropagated correctly

**Result**: dq has RMS error ≈ 0 (exact match with true BPTT)

#### Phase 2: Compute dk, dv (lines 76-93) - **dk FAILS**

```python
def kv_scan(carry, x):
    k_state, v_state = carry
    k, v, q, o, do, dq = x

    # dv computation (works okay)
    new_v_state, dv = v_scan(v_state, (q, do, k))

    # dk computation (FAILS)
    new_k_state, dk = k_scan(k_state, (q, q+dq, k))

    return (new_k_state, new_v_state), (dk, dv)
```

**Critical lines**:
- Line 80: `v_scan(v_state, (q, do, k))`
  - Passes `(q, do, k)` as `(k, v, q)` to scan function
  - So reconstruction target is `do` (output gradient)
  - Reconstructs: `||fwd_fn(v_state, k) - do||²`
  - Gradient w.r.t v_state → `dv`

- Line 82: `k_scan(k_state, (q, q+dq, k))`
  - Passes `(q, q+dq, k)` as `(k, v, q)` to scan function
  - So reconstruction target is `q+dq` (perturbed query)
  - Reconstructs: `||fwd_fn(k_state, k) - (q+dq)||²`
  - Gradient w.r.t k_state → `dk`

#### State Hacking (lines 87-91)

```python
k_state = jax.tree.map(lambda x: x, end_state)
k_state.down_proj = jax.tree.map(jnp.zeros_like, k_state.down_proj)

v_state = jax.tree.map(lambda x: x, end_state)
v_state.down_proj = jax.tree.map(jnp.zeros_like, v_state.down_proj)
```

**Purpose unclear**:
- Copies final state from q_scan
- Zeros out `down_proj` component
- Likely a hack to prevent gradient explosion
- No theoretical justification

## Root Cause Analysis

### Why dk fails

The fundamental issue is **disconnected gradient flow**:

1. **True gradient** dk should flow through: `k → state_update → all_future_outputs → loss`
   - k affects how state is updated at each timestep
   - Updated states affect all subsequent outputs
   - This is a complex, temporal dependency

2. **Surrogate gradient** dk uses: `||fwd_fn(k_state, k) - (q+dq)||²`
   - Tries to make k reconstruct the perturbed query `q+dq`
   - But k's actual role is to update the state, not reconstruct queries
   - The target `q+dq` has no theoretical connection to the actual gradient

3. **Asymmetry with dv**:
   - dv uses `do` (output gradient) as target - at least this is related to the loss
   - dk uses `q+dq` (perturbed query) as target - completely arbitrary
   - No principled reason for this asymmetry

### Why dv works (partially)

dv cosine similarity ranges [-0.52, 0.80], with mean > 0:
- Using `do` as reconstruction target is more sensible than `q+dq`
- Output gradient `do` is at least related to the loss function
- But it's still not the true gradient - just a heuristic that happens to work better

### Why dq works perfectly

dq uses true BPTT through the forward scan:
- No surrogate approximation
- Properly accumulates state gradients
- Correctly applies chain rule through all operations

## Experimental Evidence

From `ttt_gradient_analysis.ipynb` (seq_len=200, hidden_d=512):

| Gradient | Metric | Result | Status |
|----------|--------|--------|--------|
| dq | RMS error | ≈ 0.0000 | ✅ Perfect |
| dv | Cosine sim | [-0.52, 0.80], mean > 0 | ⚠️ Acceptable |
| dk | Cosine sim | [-0.46, 0.36], oscillates around 0 | ❌ Complete failure |

**Interpretation**:
- Cosine sim ≈ 0 means dk is orthogonal to true gradient (random direction)
- Not just approximation error - fundamentally wrong direction
- Would cause divergence in training

## Commented-Out Code (Line 81)

```python
# new_k_state, dk = k_scan(k_state, ((q, o+do), dq, (k, v+dv)))
```

This suggests the author tried a different approach:
- Use `dq` as reconstruction target (instead of `q+dq`)
- Include perturbed inputs `(q, o+do)` and `(k, v+dv)`
- Apparently this also failed (hence commented out)

## Hypotheses for Fix Attempts

### Fix 1: Symmetric Treatment
**Hypothesis**: Make dk use same pattern as dv
- Change line 82 to: `k_scan(k_state, (q, do, k))`
- Use output gradient `do` as target for both
- **Expected outcome**: dk might improve to dv-level performance

### Fix 2: State Gradient Propagation
**Hypothesis**: Use accumulated `dstate` from q_scan
- The q_scan already computes `dstate` (gradient w.r.t state)
- Chain rule: dk = dL/dstate × dstate/dk
- Compute dstate/dk via reconstruction VJP properly
- **Expected outcome**: More principled, could match true BPTT

### Fix 3: Remove State Hacking
**Hypothesis**: The zero-ing of down_proj is masking deeper issues
- Remove lines 87-91 entirely
- Use `end_state` directly without modification
- **Expected outcome**: May reveal why the hack was needed, guide better fix

## Next Steps

1. Try Fix 1 (simplest, quick test)
2. If Fix 1 fails, try Fix 2 (more complex but principled)
3. If both fail, try Fix 3 to understand root cause
4. Document findings from each attempt
5. If all fail, proceed to alternative methods (implicit diff, forward-mode, hybrid)

## References

- Original implementation: `ueaj/model/ttt/impl.py`
- Test notebook: `ueaj/model/ttt/ttt_gradient_analysis.ipynb`
- Blog post: `BPTT got hands - ueaj - Obsidian Publish.pdf`
