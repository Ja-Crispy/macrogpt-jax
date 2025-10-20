# Fix 2 & Fix 3: Testing Guide

## Date
2025-10-21

## What Was Changed

### Fix 2: State Gradient Propagation

**File**: `ueaj/model/ttt/impl.py`, lines 85-100

**Approach**: Chain gradients through accumulated `dstate` from q_scan (true BPTT)

**Implementation**:
```python
# Fix 2 computes:
def state_update_via_k(k_input):
    # How does k affect state update?
    v_pred, vjp_fn = jax.vjp(lambda s: fwd_fn(s, k_input), k_state)
    dv_local = v - v_pred
    dstate_local, = vjp_fn(dv_local)
    return dstate_local

# Compute dstate/dk
_, vjp_dstate_wrt_k = jax.vjp(state_update_via_k, k)

# Chain rule: dk = (dL/dstate) × (dstate/dk)
dk, = vjp_dstate_wrt_k(dstate_final)
```

**Rationale**:
- `dstate_final` from q_scan contains gradient w.r.t. state (true BPTT)
- Compute how k affects state update via VJP
- Chain them: captures temporal dependency properly

### Fix 3: Remove State Hacking

**File**: `ueaj/model/ttt/impl.py`, lines 109-113

**Change**:
```python
# ORIGINAL:
k_state = jax.tree.map(lambda x: x, end_state)
k_state.down_proj = jax.tree.map(jnp.zeros_like, k_state.down_proj)  # Zeroed!

# FIX 3:
k_state = jax.tree.map(lambda x: x, end_state)
# k_state.down_proj = ... # REMOVED
```

**Rationale**:
- Original code zeros out `down_proj` component (unclear why)
- Likely a hack to prevent gradient explosion
- Test if Fix 2's proper chaining makes this unnecessary
- Diagnostic: may reveal why hack existed

## Combined Effect

**Hypothesis**: Fix 2 + Fix 3 together might work because:
1. Fix 2 properly chains gradients (should give correct direction)
2. Fix 3 removes artificial constraint (allows full gradient flow)
3. Together: correct gradient computation without hacks

**Alternative hypothesis**: Fix 3 causes errors/NaNs, revealing that the hack was necessary.

## Test Setup

**Colab notebook**: Re-run `ttt_gradient_analysis_new.ipynb`

**What to check**:
1. Does code run without errors? (Fix 3 might break things)
2. Are there NaNs/Infs in gradients?
3. dk cosine similarity range
4. dk mean cosine similarity

## Expected Outcomes

### Best Case (Success): ✅
- **dk cosine_sim**: > 0.5 (target threshold)
- **dk mean**: Consistently positive
- **No NaNs/Infs**: Gradients are stable
- **Interpretation**: Fix 2's proper chaining works!

### Good Case (Improvement): ⚠️
- **dk cosine_sim**: 0.2 - 0.5 (better than Fix 1's ~0)
- **dk mean**: Positive but below target
- **Stable gradients**: No numerical issues
- **Interpretation**: Fix 2 helps but not enough, try alternative methods

### Diagnostic Case (Fix 3 breaks it): 🔍
- **Errors or NaNs**: Code crashes or produces invalid gradients
- **Much worse than Fix 1**: dk cosine_sim < -0.5
- **Interpretation**: State hacking was necessary, reveals gradient explosion issue

### Failure Case (No improvement): ❌
- **dk cosine_sim**: Still ~0 (same as Fix 1)
- **No change**: Similar to original
- **Interpretation**: Surrogate approach fundamentally flawed, proceed to alternatives

## How to Test

### Step 1: Open Colab
```
https://colab.research.google.com/
```

### Step 2: Run Notebook

Create new cells or modify existing:

```python
# Cell 1: Clone updated repo
!git clone https://github.com/Ja-Crispy/macrogpt-jax.git -b surrogate-ttt

# Cell 2: Change directory
%cd macrogpt-jax

# Cell 3: Check which fix is active
!git log --oneline -5

# Expected output should show:
# f0ac3d9 Fix 3: Remove state hacking (down_proj zeroing)
# cfd8928 Fix 2: Chain gradients via accumulated dstate from q_scan
# 6149ac7 Fix 1: Make dk gradient symmetric with dv (use do as target)

# Proceed with rest of notebook...
```

### Step 3: Run Analysis Cells

Run all cells in the notebook (same as before). Pay attention to:

**Cell 9**: Does it complete without errors?
- ✅ Yes → Continue
- ❌ No → Note the error, report back

**Cell 10**: Check metrics
```python
print(f"Cosine similarity ranges:")
print(f"  k: [{cos_k.min():.4f}, {cos_k.max():.4f}]")  # KEY METRIC
print(f"  v: [{cos_v.min():.4f}, {cos_v.max():.4f}]")
print(f"Mean cosine similarity:")
print(f"  k: {cos_k.mean():.4f}")  # MOST IMPORTANT
```

### Step 4: Compare Results

| Fix | dk cosine range | dk mean | Verdict |
|-----|-----------------|---------|---------|
| Original | [-0.46, 0.36] | ~0.0 | ❌ Failure |
| Fix 1 | [-0.43, 0.37] | ~0.0 | ❌ Failure (5% improvement) |
| Fix 2+3 | **[?, ?]** | **?** | **?** |

**Target**: dk mean > 0.5, consistent positive values

## Debugging Guide

### If you get errors:

**Error in `kv_scan`**:
- Likely Fix 2's VJP computation has shape mismatch
- Report the full traceback

**NaN/Inf gradients**:
- Likely Fix 3 removal caused gradient explosion
- Check `jnp.isnan(dk).any()` and `jnp.isinf(dk).any()`
- If True, Fix 3 broke it → state hacking WAS necessary

**Shape mismatch errors**:
- Possible tree structure issue in Fix 2
- Check that `dstate_final` has same structure as state

## What to Report Back

1. **Did it run?**: Yes / No (if no, error message)
2. **Any NaNs/Infs?**: Yes / No
3. **dk cosine range**: [min, max]
4. **dk mean cosine similarity**: value
5. **Compared to Fix 1**: Better / Worse / Same

## Next Steps Based on Results

### If Fix 2+3 succeeds (dk mean > 0.5):
- ✅ Document success
- Create test framework
- Run scaling tests
- Write up findings

### If Fix 2+3 partially works (dk mean 0.2-0.5):
- Promising but not enough
- Still proceed to alternative methods
- Use Fix 2+3 as baseline for comparison

### If Fix 2+3 fails (dk mean ~0):
- Surrogate approach confirmed broken for dk
- Skip remaining surrogate fixes
- **Proceed directly to Alternative Methods**:
  1. Implicit differentiation (Phase 2)
  2. Hybrid chunking (reliable fallback)

### If Fix 3 causes errors:
- Revert Fix 3 (keep only Fix 2)
- Test Fix 2 alone
- Document why state hacking is necessary

## Commits

- **Fix 2**: `cfd8928`
- **Fix 3**: `f0ac3d9`
- **Branch**: `surrogate-ttt` on `Ja-Crispy/macrogpt-jax`

## Files

- Implementation: `ueaj/model/ttt/impl.py`
- Test notebook: `ueaj/model/ttt/ttt_gradient_analysis_new.ipynb` (on Colab)
- This guide: `FIX2_FIX3_TESTING_GUIDE.md`
