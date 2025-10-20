# Fix 1 Results: Symmetric Treatment

## Date
2025-10-21

## Fix Applied

**File**: `ueaj/model/ttt/impl.py`, line 84

**Change**:
```python
# ORIGINAL (line 82):
new_k_state, dk = k_scan(k_state, (q, q+dq, k))

# FIX 1 (line 84):
new_k_state, dk = k_scan(k_state, (q, do, k))
```

**Rationale**: Make dk computation symmetric with dv - both use output gradient `do` as reconstruction target.

## Test Results

**Setup**:
- Seq length: 200 tokens
- Hidden dim: 512
- Batch size: 1
- Model: GMLP with surrogate=True vs surrogate=False
- Notebook: `ueaj/model/ttt/ttt_gradient_analysis_new.ipynb`

### Metrics Comparison

| Gradient | Metric | Original (`q+dq`) | Fix 1 (`do`) | Change |
|----------|--------|-------------------|--------------|---------|
| **dk** | Cosine sim range | [-0.4565, 0.3614] | [-0.4340, 0.3694] | Δmin: +0.0225, Δmax: +0.0080 |
| **dk** | Mean cosine sim | ~0.0 | ~0.0 | No improvement |
| **dv** | Cosine sim range | [-0.5208, 0.7992] | [-0.5498, 0.8010] | Minor changes |
| **dq** | RMS error | 0.0000 | 0.0000 | Still perfect ✅ |

### Detailed Numbers

```
Original Results (before Fix 1):
  dk: [-0.4565, 0.3614]
  dv: [-0.5208, 0.7992]
  dq: [0.0000, 0.0000]

Fix 1 Results:
  dk: [-0.4340, 0.3694]
  dv: [-0.5498, 0.8010]
  dq: [0.0000, 0.0000]
```

## Analysis

### Verdict: ❌ **COMPLETE FAILURE**

Fix 1 did **NOT** improve dk gradient quality:
- dk cosine similarity still oscillates around 0 (mean ≈ 0)
- Changes are at noise level (~2-3% variance)
- No meaningful improvement in gradient direction
- Still fails descent direction test (orthogonal to true gradient)

### Why Fix 1 Failed

The symmetric treatment hypothesis was **incorrect**. The problem is NOT asymmetry between dk and dv.

**Root cause**: Neither dk nor dv properly captures gradient flow through state updates.

#### k's Actual Role in TTT

1. **Forward pass**: At each timestep:
   ```python
   v_pred = fwd_fn(state_t, k_t)  # Use current state + k to predict
   loss = ||v_pred - v_t||²        # Reconstruction loss
   grad_state = ∇_state(loss)      # Gradient w.r.t. state
   state_{t+1} = state_t - lr * grad_state  # Update state
   ```

2. **k's influence**: k_t affects state_{t+1}, which affects ALL future outputs
   - This is a **temporal dependency** through state updates
   - Gradient must flow: dk_t ← state_{t+1} ← state_{t+2} ← ... ← output ← loss

#### What Fix 1 Does (Wrong)

```python
# Fix 1 computes:
reconstruction_loss = ||fwd_fn(k_state, k) - do||²
dk = ∇_k(reconstruction_loss)
```

**Problems**:
1. Uses `do` (output gradient) as reconstruction target
2. But k doesn't directly produce outputs - it **updates state**!
3. The reconstruction target `do` is disconnected from k's actual role
4. Ignores temporal dependencies completely

#### Why dv Works Better (Partially)

dv has cosine_sim in [-0.55, 0.80] with mean > 0:
- v IS the reconstruction target in forward pass: `||fwd_fn(state, k) - v||²`
- Using `do` maintains some connection to the loss function
- Still not true BPTT, but less disconnected than dk

**Key insight**: dv works because v's role (reconstruction target) happens to align with using `do`. dk has no such alignment.

## The Fundamental Problem

The surrogate method tries to **avoid BPTT through state updates**, but:
- dk REQUIRES gradients flowing through state updates (temporal dependency)
- You can't compute correct dk without BPTT or an equivalent method

This is why:
- **dq works perfectly**: Uses true BPTT (line 67-70 in impl.py)
- **dv works partially**: Heuristic happens to work due to role alignment
- **dk fails completely**: No heuristic can replace true temporal gradient flow

## Next Steps

### Fix 2: State Gradient Propagation (More Complex)

**Hypothesis**: Properly chain gradients through state updates using accumulated `dstate` from q_scan.

**Key insight**: After q_scan (line 74), we have:
- `end_state`: Final state after sequence
- `dstate_final`: Gradient w.r.t. state (computed via true BPTT)

**Approach**: Compute dk by chaining:
```
dk_t = dstate_{t+1}/dk_t × dstate_{t+1}
```

This captures how k_t affects state_{t+1}, weighted by gradient from future.

**Challenge**: More complex to implement correctly, requires careful tree manipulation.

### Fix 3: Remove State Hacking (Diagnostic)

**Hypothesis**: The zeroing of `down_proj` (lines 87-91) is masking deeper issues.

**Approach**: Remove the hack, see what breaks and why.

**Purpose**: Understand WHY the hack was needed - may reveal the true problem.

### If All Fixes Fail: Alternative Methods

The surrogate approach may be fundamentally flawed for dk. Proceed to:
1. **Implicit differentiation**: Most theoretically sound
2. **Forward-mode with truncation**: Approximate but principled
3. **Hybrid chunking**: Engineering solution (guaranteed to work)

## Files

- **Test notebook**: `ueaj/model/ttt/ttt_gradient_analysis_new.ipynb`
- **Implementation**: `ueaj/model/ttt/impl.py` (reverted to Fix 1)
- **Analysis**: `surrogate_analysis.md`
- **Git commit**: `6149ac7` (Fix 1 on branch `surrogate-ttt`)

## References

- Original blog post: `BPTT got hands - ueaj - Obsidian Publish.pdf`
- Testing guide: `FIX1_TESTING_GUIDE.md`
- Full plan: `plan.txt`
