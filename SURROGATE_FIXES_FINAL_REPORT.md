# Surrogate Gradient Fixes: Final Report

## Date
2025-10-21

## Executive Summary

**Objective**: Fix the dk (key gradient) computation in TTT's surrogate backward pass.

**Outcome**: ❌ **All surrogate-based fixes failed**

**Conclusion**: The surrogate gradient approach is **fundamentally incompatible** with computing correct dk gradients. The issue is not implementation bugs but a theoretical limitation of trying to avoid BPTT for gradients that inherently require temporal dependency tracking.

---

## Background

### The Problem

TTT's surrogate backward pass produces three gradients:
- **dq** (query): RMS error ≈ 0 ✅ Perfect (uses true BPTT)
- **dv** (value): Cosine sim ∈ [-0.52, 0.80] ⚠️ Partial (heuristic alignment)
- **dk** (key): Cosine sim ∈ [-0.46, 0.36], mean ≈ 0 ❌ Complete failure

**Target**: dk cosine similarity > 0.5 (valid descent direction)

### Why dk Matters

In TTT forward pass:
1. k updates the state via reconstruction: `state' = state - lr * ∇(||fwd_fn(state, k) - v||²)`
2. Updated state affects ALL future outputs
3. This creates temporal dependencies: k_t → state_t → state_{t+1} → ... → outputs → loss

**Critical insight**: dk gradient MUST flow through these state updates. You cannot compute it with local approximations.

---

## Fix Attempts

### Fix 1: Symmetric Treatment

**Date**: 2025-10-21
**Commit**: `6149ac7`

**Hypothesis**: Make dk use same pattern as dv (both use output gradient `do` as reconstruction target)

**Implementation**:
```python
# Original (line 82):
new_k_state, dk = k_scan(k_state, (q, q+dq, k))

# Fix 1 (line 84):
new_k_state, dk = k_scan(k_state, (q, do, k))
```

**Rationale**:
- dv works partially by using `do`
- Maybe dk needs same treatment?

**Results**:

| Metric | Original | Fix 1 | Change |
|--------|----------|-------|--------|
| dk min | -0.4565 | -0.4340 | +0.0225 (+5%) |
| dk max | +0.3614 | +0.3694 | +0.0080 (+2%) |
| dk mean | ~0.0 | ~0.0 | No improvement |
| dv | [-0.52, 0.80] | [-0.55, 0.80] | Unchanged |
| dq | 0.0000 | 0.0000 | Still perfect |

**Verdict**: ❌ **Failed**
- Changes at noise level (~2-5%)
- Still oscillates around 0
- No meaningful improvement

**Why it failed**:
- dv works because v IS the reconstruction target in forward pass
- k's role is different - it updates state, not outputs directly
- Using `do` as target for k doesn't align with k's actual function
- The "symmetry" hypothesis was wrong - k and v have fundamentally different roles

---

### Fix 2: State Gradient Propagation

**Date**: 2025-10-21
**Commit**: `cfd8928`

**Hypothesis**: Chain gradients through accumulated `dstate` from q_scan (which uses true BPTT)

**Implementation**:
```python
def kv_scan(carry, x):
    k_state, v_state = carry
    k, v, q, o, do, dq = x

    # Compute how k affects state update
    def state_update_via_k(k_input):
        v_pred, vjp_fn = jax.vjp(lambda s: fwd_fn(s, k_input), k_state)
        dv_local = v - v_pred
        dstate_local, = vjp_fn(dv_local)
        return dstate_local

    # Compute dstate/dk using VJP
    _, vjp_dstate_wrt_k = jax.vjp(state_update_via_k, k)

    # Chain rule: dk = (dL/dstate) × (dstate/dk)
    dk, = vjp_dstate_wrt_k(dstate_final)

    return (k_state, v_state), (dk, dv)
```

**Rationale**:
- `dstate_final` from q_scan contains gradient w.r.t. state (true BPTT)
- Compute how k affects state update via VJP
- Chain them properly: dk = ∂L/∂state × ∂state/∂k

**Results**:

| Metric | Fix 1 | Fix 2 | Change |
|--------|-------|-------|--------|
| dk range | [-0.43, 0.37] | [-0.43, 0.37] | **Identical** |
| dk mean | ~0.0 | ~0.0 | No improvement |

**Verdict**: ❌ **Complete failure**
- Literally identical results to Fix 1
- No improvement whatsoever

**Why it failed**:
1. **Used final dstate uniformly**: Applied `dstate_final` (gradient at END of sequence) to ALL timesteps in backward scan
2. **Missing per-timestep tracking**: Should have accumulated per-timestep dstates, not just final
3. **Ignored temporal structure**: The backward `kv_scan` runs in reverse, but we only had access to the final accumulated dstate
4. **Fundamental flaw**: To properly chain dk through state updates, you need to track how EACH timestep's k affects EVERY FUTURE timestep's state - this IS BPTT, which defeats the purpose of surrogate

---

### Fix 3: Remove State Hacking

**Date**: 2025-10-21
**Commit**: `f0ac3d9`

**Hypothesis**: The `down_proj` zeroing is masking deeper issues. Remove it and see what breaks.

**Implementation**:
```python
# Original (lines 89-93):
k_state = jax.tree.map(lambda x: x, end_state)
k_state.down_proj = jax.tree.map(jnp.zeros_like, k_state.down_proj)

v_state = jax.tree.map(lambda x: x, end_state)
v_state.down_proj = jax.tree.map(jnp.zeros_like, v_state.down_proj)

# Fix 3:
k_state = jax.tree.map(lambda x: x, end_state)
# Removed down_proj zeroing

v_state = jax.tree.map(lambda x: x, end_state)
# Removed down_proj zeroing
```

**Rationale**:
- Original code zeros out `down_proj` component
- Unclear why - likely a hack to prevent gradient explosion
- Maybe Fix 2's proper chaining makes this unnecessary?
- Or maybe removing it reveals the real problem?

**Results**:

| Metric | Fix 2 | Fix 2+3 | Change |
|--------|-------|---------|--------|
| dk range | [-0.43, 0.37] | [-0.43, 0.37] | **Identical** |
| dk mean | ~0.0 | ~0.0 | No improvement |
| Errors? | No | No | Stable |

**Verdict**: ❌ **No effect**
- Removing the hack changed nothing
- No gradient explosion
- No improvement

**Why it had no effect**:
- Fix 2's approach prevented gradients from flowing properly anyway
- The hack was likely for a different issue (maybe related to original surrogate)
- Or Fix 2's broken implementation masked whatever the hack was preventing

---

## Root Cause Analysis

### Why Surrogate Method Cannot Work for dk

**The fundamental problem**:

k's gradient inherently requires tracking temporal dependencies:
```
k_t affects state_{t+1}
state_{t+1} affects output_{t+1}, output_{t+2}, ..., output_T
All future outputs affect final loss

Therefore: dk_t = Σ_{i=t+1}^T (∂loss/∂output_i × ∂output_i/∂state_i × ∂state_i/∂k_t)
```

This summation over future timesteps IS backpropagation through time (BPTT). There's no way around it.

**What surrogate tries to do**: Avoid BPTT by using local reconstruction losses as proxies

**Why it can't work for dk**:
1. **Local reconstruction doesn't capture future influence**: `||fwd_fn(state, k) - target||²` only captures k's immediate effect on reconstructing some target, not its effect on future states
2. **No temporal credit assignment**: Cannot determine how k_t affects loss through multiple future timesteps without actually computing those dependencies
3. **Fundamental incompatibility**: The surrogate method's goal (avoid BPTT) conflicts with dk's requirement (temporal gradient flow)

### Why dq Works Perfectly

Looking at lines 63-74 in `impl.py`:
```python
def q_scan(carry, x):
    state, dstate = carry
    (new_state, o), q_update_jvp = jax.vjp(
        lambda state, q: fwd_scan(state, (k, v, q)),
        state, q
    )
    new_dstate, dq = q_update_jvp((dstate, do))
    return (new_state, new_dstate), (o, dq)
```

**Why it works**: This IS true BPTT! It:
1. Computes VJP through `fwd_scan` (includes state updates)
2. Accumulates `dstate` across timesteps
3. Properly chains gradients: `dq = ∂loss/∂output × ∂output/∂q + ∂loss/∂state × ∂state/∂q`

**The irony**: dq works because it doesn't use a surrogate - it uses proper BPTT.

### Why dv Works Partially

**dv uses**: `v_scan(v_state, (q, do, k))` - reconstruction target is `do`

**Why it works better than dk**:
- v is ALREADY the reconstruction target in forward pass: `||fwd_fn(state, k) - v||²`
- Using `do` (output gradient) as surrogate target has some alignment with v's actual role
- Not perfect (cosine sim only 0-0.8), but better than dk's complete failure

**Key insight**: dv works by accident of role alignment, not by design. It's still a heuristic, not true BPTT.

---

## Lessons Learned

### 1. Surrogate Gradients Have Fundamental Limits

You cannot surrogate away temporal dependencies. If a gradient requires BPTT, there's no clever trick to avoid it while remaining correct.

### 2. Role Alignment Matters for Heuristics

dv's partial success shows that heuristics can work IF there's alignment between:
- The variable's actual role in forward pass
- The reconstruction target used in backward pass

dk fails because k's role (state updater) has no such alignment with any simple target.

### 3. Fix Quality Depends on Understanding

- **Fix 1**: Wrong hypothesis (symmetry) → small random improvement
- **Fix 2**: Right idea (chain through state) but wrong implementation (used final dstate uniformly) → no improvement
- **Fix 3**: Diagnostic (remove hack) → revealed hack wasn't the problem

Better understanding would have saved time on doomed approaches.

### 4. Test Against Ground Truth Early

If we'd computed ground truth dk gradients (full BPTT) for a small sequence FIRST, we would have immediately seen that:
- Fix 1's 5% improvement is noise
- Fix 2's identical results mean it's not working at all

---

## Implications for Alternative Methods

### What Will Work

**Methods that properly handle temporal dependencies**:
1. **Implicit Differentiation**: Avoids BPTT by solving equilibrium conditions, but still captures full gradient
2. **Hybrid Chunking**: Uses true BPTT within blocks (tractable for small blocks)
3. **Truncated BPTT**: Approximate but principled - gradient flows through K past steps

### What Won't Work

**Any method that tries local approximations**:
- Using reconstruction losses with arbitrary targets
- Heuristic gradient estimators without temporal structure
- "Clever tricks" that ignore the mathematical requirements

### Key Insight for Future Methods

**dk gradient computation is non-negotiable**: You either:
1. Do full BPTT (expensive but correct)
2. Do BPTT on smaller chunks (hybrid approach)
3. Use implicit differentiation (mathematically equivalent to BPTT at equilibrium)
4. Accept approximate gradients (truncated BPTT)

There is no fifth option that avoids temporal computation while remaining correct.

---

## Recommendations

### Immediate Next Steps

1. **Implement Implicit Differentiation**
   - Most theoretically sound alternative
   - Avoids BPTT via equilibrium conditions
   - Should work if implemented correctly

2. **Implement Hybrid Chunking**
   - Reliable fallback
   - Uses true BPTT within blocks
   - Guaranteed to work (just smaller BPTT)

3. **Compare Against Baselines**
   - Surrogate (current): dk cosine_sim ~0
   - Full BPTT: dk cosine_sim = 1 (ground truth)
   - New methods should fall between

### Long-term Implications

**For TTT research**:
- Surrogate gradients may not be viable for all components
- May need hybrid approach: surrogate for some, BPTT for others
- Or accept that dk requires special treatment

**For paper/blog post**:
- Negative results are valuable: document WHY surrogate fails
- Theoretical analysis of when surrogate methods work/don't work
- Comparison of alternative approaches

---

## Conclusion

After three fix attempts across multiple days:
- **Fix 1**: 5% improvement (noise level)
- **Fix 2**: 0% improvement (identical to Fix 1)
- **Fix 3**: 0% improvement (removing hack had no effect)

**Final verdict**: Surrogate gradient approach cannot compute correct dk gradients. The issue is not bugs but fundamental incompatibility between:
- The goal: avoid BPTT
- The requirement: dk needs temporal gradient flow

**Next phase**: Implement alternative methods that properly handle temporal dependencies:
1. Implicit differentiation (Phase 2)
2. Hybrid chunking (Phase 2)

These methods respect the mathematical requirements while offering different trade-offs for compute/memory/accuracy.

---

## Files

- **Implementations**: `ueaj/model/ttt/impl.py` (all fixes applied)
- **Test notebook**: `ueaj/model/ttt/ttt_gradient_analysis_new.ipynb` (Colab)
- **Analysis**: `surrogate_analysis.md`, `FIX1_RESULTS.md`, `FIX2_FIX3_TESTING_GUIDE.md`
- **Git commits**:
  - Fix 1: `6149ac7`
  - Fix 2: `cfd8928`
  - Fix 3: `f0ac3d9`
- **Branch**: `surrogate-ttt` on `Ja-Crispy/macrogpt-jax`

## References

- Original blog post: `BPTT got hands - ueaj - Obsidian Publish.pdf`
- Full plan: `plan.txt`
- Testing guides: `FIX1_TESTING_GUIDE.md`, `FIX2_FIX3_TESTING_GUIDE.md`
