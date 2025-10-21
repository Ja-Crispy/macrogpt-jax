# TTT Alternative Methods Testing Guide

## Quick Start (Colab)

After `git pull origin surrogate-ttt`, you have three new gradient methods to test:
1. **Implicit** - Implicit differentiation (priority 1)
2. **Hybrid** - BPTT within blocks (reliable fallback)
3. **Forward** - Forward-mode with truncated history

## Method 1: Quick Test in Existing Notebook

Modify `ueaj/model/ttt/ttt_gradient_analysis.ipynb` cell 3:

```python
# Original (failed):
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate=True, rngs=rng.Rngs(seed))

# Test Implicit:
model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='implicit', rngs=rng.Rngs(seed))

# Or test Hybrid:
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='hybrid', rngs=rng.Rngs(seed))

# Or test Forward:
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='forward', rngs=rng.Rngs(seed))

model_nonsurr = TTTModel(model_d, hidden_d, GMLP, surrogate=False, rngs=rng.Rngs(seed))
```

Run the notebook. Check output of cell 6:

**Success if:**
- dk cosine_sim > 0.5 (minimum)
- dk cosine_sim > 0.8 (strong success)

**Current baseline (failed surrogate):**
- dk: [-0.46, 0.36] ❌
- dv: [-0.52, 0.80] ✓
- dq: [0.00, 0.00] ✓

## Method 2: Unified Testing Framework (Recommended)

Add new cell to notebook:

```python
from ueaj.model.ttt import TTTGradientTester

# Create tester (matches original notebook config)
tester = TTTGradientTester(seq_len=200, hidden_d=512, batch_size=1, model_d=128)

# Test implicit method
print("\n" + "="*60)
print("TESTING IMPLICIT DIFFERENTIATION")
print("="*60)
model_implicit = TTTModel(128, 512, GMLP, surrogate='implicit', rngs=rng.Rngs(42))
model_bptt = TTTModel(128, 512, GMLP, surrogate=False, rngs=rng.Rngs(42))
metrics_implicit, summary_implicit = tester.test_method(model_implicit, model_bptt, "Implicit")

# Test hybrid method
print("\n" + "="*60)
print("TESTING HYBRID CHUNKING")
print("="*60)
model_hybrid = TTTModel(128, 512, GMLP, surrogate='hybrid', rngs=rng.Rngs(42))
metrics_hybrid, summary_hybrid = tester.test_method(model_hybrid, model_bptt, "Hybrid")

# Test forward method
print("\n" + "="*60)
print("TESTING FORWARD-MODE")
print("="*60)
model_forward = TTTModel(128, 512, GMLP, surrogate='forward', rngs=rng.Rngs(42))
metrics_forward, summary_forward = tester.test_method(model_forward, model_bptt, "Forward")
```

This automatically:
- Computes gradients for all methods
- Compares to ground truth BPTT
- Prints detailed metrics
- Shows SUCCESS/PARTIAL/FAILURE assessment

## Method 3: Full Comparison with Plots

```python
from ueaj.model.ttt import TTTGradientTester

tester = TTTGradientTester(seq_len=200, hidden_d=512, batch_size=1, model_d=128)

# Create all models
model_surrogate = TTTModel(128, 512, GMLP, surrogate=True, rngs=rng.Rngs(42))
model_implicit = TTTModel(128, 512, GMLP, surrogate='implicit', rngs=rng.Rngs(42))
model_hybrid = TTTModel(128, 512, GMLP, surrogate='hybrid', rngs=rng.Rngs(42))
model_forward = TTTModel(128, 512, GMLP, surrogate='forward', rngs=rng.Rngs(42))
model_bptt = TTTModel(128, 512, GMLP, surrogate=False, rngs=rng.Rngs(42))

# Compare all methods
methods = {
    'Surrogate (Original)': (model_surrogate, model_bptt),
    'Implicit Diff': (model_implicit, model_bptt),
    'Hybrid Chunking': (model_hybrid, model_bptt),
    'Forward Mode': (model_forward, model_bptt),
}

results, summaries = tester.compare_methods(methods, save_dir='./ttt_results')
```

This generates:
- `ttt_results/comparison.png` - Visual comparison
- `ttt_results/summary.txt` - Human-readable results
- `ttt_results/summary.csv` - Data table

## Expected Results

### Implicit Differentiation
- **Theory**: Uses equilibrium conditions + CG solver
- **Expected dk**: 0.3 to 0.7 (if working)
- **Memory**: O(d²) for Hessian-vector products
- **Risk**: CG may not converge, equilibrium assumption may fail

### Hybrid Chunking
- **Theory**: BPTT within 64-token blocks
- **Expected dk**: 0.7 to 0.95 (close to full BPTT)
- **Memory**: O(64 × params) - tractable
- **Risk**: None (BPTT is mathematically correct)
- **Note**: This is the reliable fallback

### Forward-Mode
- **Theory**: JVP with truncated history (32 timesteps)
- **Expected dk**: 0.2 to 0.5
- **Memory**: O(32 × d²)
- **Risk**: Truncation may lose long-range dependencies

## Success Criteria

**Minimum Viable Success (any method):**
- dk cosine_sim mean > 0.5
- Valid descent direction (d·g > 0)
- No NaNs or Infs
- Stable across 200 token sequence

**Strong Success:**
- dk cosine_sim mean > 0.8
- Matches dv/dq quality
- Scales to longer sequences

## Testing Priority

1. **Start with Implicit** - Most promising, mathematically rigorous
2. **If Implicit fails → Hybrid** - Guaranteed to work (BPTT is correct)
3. **Forward-mode** - Optional, interesting for research

## Troubleshooting

### If you see errors about missing modules:
```bash
# In Colab, pull latest changes
!git pull origin surrogate-ttt

# Restart runtime (Runtime → Restart runtime)
```

### If gradients are NaN/Inf:
- Check numerical stability flags in test output
- Try hybrid method (most stable)
- Reduce sequence length for testing

### If dk is still ~0:
- Verify you're using the new methods (check `model.method` attribute)
- Check that git pull succeeded
- Try hybrid (can't fail mathematically)

## What to Report Back

For each method tested, report:
1. **dk cosine_sim range** (e.g., [-0.46, 0.36])
2. **dk cosine_sim mean** (e.g., 0.02)
3. **Success assessment** from test output
4. **Any errors or warnings**

Example good report:
```
Implicit: dk ∈ [0.45, 0.82], mean=0.63 → SUCCESS ✓
Hybrid: dk ∈ [0.71, 0.94], mean=0.85 → SUCCESS ✓
Forward: dk ∈ [0.18, 0.43], mean=0.31 → PARTIAL ⚠
```

## Next Steps After Testing

**If at least one method succeeds (dk > 0.5):**
1. Run scaling tests (longer sequences)
2. Test on real language modeling task
3. Document results
4. Compare methods quantitatively

**If all methods fail:**
1. Check for implementation bugs
2. Verify test setup matches expected config
3. Document failure modes
4. Investigate root causes

## Files Reference

- **Implementations**: `ueaj/model/ttt/impl_*.py`
- **Test framework**: `ueaj/model/ttt/test_framework.py`
- **Model setup**: `ueaj/model/ttt/module.py`
- **Failure analysis**: `SURROGATE_FIXES_FINAL_REPORT.md`
