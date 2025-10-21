# TTT Alternative Learning Methods Testing Guide

## Quick Start (Colab)

After `git pull origin surrogate-ttt`, you have **6 new alternative learning mechanisms** to test.

**IMPORTANT**: All backprop-based gradient approximation methods (implicit, hybrid, forward-mode) failed mathematically. See `ALTERNATIVE_METHODS_FAILURE_ANALYSIS.md` for details.

**New methods replace backpropagation entirely:**
1. **Feedback Alignment** (`'feedback'`) - Fixed random feedback matrices (priority 1)
2. **Direct Feedback Alignment** (`'dfa'`) - Simplified single-matrix variant
3. **Forward-Forward** (`'forward_forward'`) - Hinton's contrastive learning
4. **Goodness** (`'goodness'`) - Simplified forward-forward without negatives
5. **Perturbation** (`'perturbation'`) - Activity perturbation (SLOW)
6. **SPSA** (`'spsa'`) - Simultaneous perturbation (faster variant)

## Method 1: Quick Test in Existing Notebook

Modify `ueaj/model/ttt/ttt_gradient_analysis.ipynb` cell 3:

```python
# Original methods (ALL FAILED - see ALTERNATIVE_METHODS_FAILURE_ANALYSIS.md):
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate=True, rngs=rng.Rngs(seed))  # Surrogate ❌
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='implicit', rngs=rng.Rngs(seed))  # Implicit ❌
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='hybrid', rngs=rng.Rngs(seed))  # Hybrid ❌
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='forward', rngs=rng.Rngs(seed))  # Forward ❌

# Test Feedback Alignment (PRIORITY 1):
model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='feedback', rngs=rng.Rngs(seed))

# Or test Direct Feedback Alignment:
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='dfa', rngs=rng.Rngs(seed))

# Or test Forward-Forward:
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='forward_forward', rngs=rng.Rngs(seed))

# Or test Goodness (simpler forward-forward):
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='goodness', rngs=rng.Rngs(seed))

# Or test Perturbation (WARNING: VERY SLOW):
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='perturbation', rngs=rng.Rngs(seed))

# Or test SPSA (faster perturbation):
# model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate='spsa', rngs=rng.Rngs(seed))

model_nonsurr = TTTModel(model_d, hidden_d, GMLP, surrogate=False, rngs=rng.Rngs(seed))
```

Run the notebook. Check output of cell 6:

**NOTE**: Alternative learning methods don't compute true gradients - they use pseudo-gradients or learning signals. Success criteria:
- Learning signal provides valid descent direction
- No NaNs or Infs
- Cosine similarity to BPTT pseudo-gradients indicates alignment

**For comparison - failed backprop methods:**
- Surrogate: dk ∈ [-0.46, 0.36] ❌
- Implicit: RMS divergence 10^6 ❌
- Hybrid: dk ∈ [-0.10, 0.96] (gradient interference) ❌
- Forward-mode: dk = 0.000 exactly ❌

## Method 2: Unified Testing Framework (Recommended)

Add new cell to notebook:

```python
from ueaj.model.ttt import TTTGradientTester

# Create tester (matches original notebook config)
tester = TTTGradientTester(seq_len=200, hidden_d=512, batch_size=1, model_d=128)

# Test feedback alignment (PRIORITY 1)
print("\n" + "="*60)
print("TESTING FEEDBACK ALIGNMENT")
print("="*60)
model_feedback = TTTModel(128, 512, GMLP, surrogate='feedback', rngs=rng.Rngs(42))
model_bptt = TTTModel(128, 512, GMLP, surrogate=False, rngs=rng.Rngs(42))
metrics_feedback, summary_feedback = tester.test_method(model_feedback, model_bptt, "Feedback")

# Test direct feedback alignment
print("\n" + "="*60)
print("TESTING DIRECT FEEDBACK ALIGNMENT")
print("="*60)
model_dfa = TTTModel(128, 512, GMLP, surrogate='dfa', rngs=rng.Rngs(42))
metrics_dfa, summary_dfa = tester.test_method(model_dfa, model_bptt, "DFA")

# Test forward-forward
print("\n" + "="*60)
print("TESTING FORWARD-FORWARD")
print("="*60)
model_ff = TTTModel(128, 512, GMLP, surrogate='forward_forward', rngs=rng.Rngs(42))
metrics_ff, summary_ff = tester.test_method(model_ff, model_bptt, "Forward-Forward")

# Test goodness (simpler variant)
print("\n" + "="*60)
print("TESTING GOODNESS-BASED LEARNING")
print("="*60)
model_goodness = TTTModel(128, 512, GMLP, surrogate='goodness', rngs=rng.Rngs(42))
metrics_goodness, summary_goodness = tester.test_method(model_goodness, model_bptt, "Goodness")

# Test SPSA (skip perturbation - too slow)
print("\n" + "="*60)
print("TESTING SPSA")
print("="*60)
model_spsa = TTTModel(128, 512, GMLP, surrogate='spsa', rngs=rng.Rngs(42))
metrics_spsa, summary_spsa = tester.test_method(model_spsa, model_bptt, "SPSA")
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
model_feedback = TTTModel(128, 512, GMLP, surrogate='feedback', rngs=rng.Rngs(42))
model_dfa = TTTModel(128, 512, GMLP, surrogate='dfa', rngs=rng.Rngs(42))
model_ff = TTTModel(128, 512, GMLP, surrogate='forward_forward', rngs=rng.Rngs(42))
model_goodness = TTTModel(128, 512, GMLP, surrogate='goodness', rngs=rng.Rngs(42))
model_spsa = TTTModel(128, 512, GMLP, surrogate='spsa', rngs=rng.Rngs(42))
model_bptt = TTTModel(128, 512, GMLP, surrogate=False, rngs=rng.Rngs(42))

# Compare all alternative methods
methods = {
    'Feedback Alignment': (model_feedback, model_bptt),
    'Direct FA (DFA)': (model_dfa, model_bptt),
    'Forward-Forward': (model_ff, model_bptt),
    'Goodness': (model_goodness, model_bptt),
    'SPSA': (model_spsa, model_bptt),
}

results, summaries = tester.compare_methods(methods, save_dir='./ttt_results')
```

This generates:
- `ttt_results/comparison.png` - Visual comparison
- `ttt_results/summary.txt` - Human-readable results
- `ttt_results/summary.csv` - Data table

## Expected Results

**IMPORTANT**: These are alternative learning mechanisms, NOT gradient approximations. They provide pseudo-gradients/learning signals that may or may not align with true BPTT gradients.

### Feedback Alignment
- **Theory**: Fixed random feedback matrices replace backprop
- **Expected**: Typical performance 80-95% of backprop in supervised tasks
- **Memory**: O(1) - no computation graph
- **Speed**: Fast - just matrix multiplications
- **Success if**: Model learns to align forward weights with random feedback

### Direct Feedback Alignment (DFA)
- **Theory**: Single feedback matrix for all inputs (simpler than FA)
- **Expected**: Sometimes works better than standard FA
- **Memory**: O(1) - single random matrix
- **Speed**: Fast - same as FA
- **Note**: Even simpler architecture

### Forward-Forward
- **Theory**: Two forward passes (positive + negative), contrastive goodness
- **Expected**: Unknown for reconstruction tasks (designed for classification)
- **Memory**: O(1) - two forward passes only
- **Speed**: 2x forward pass cost
- **Note**: Experimental - Hinton's recent algorithm

### Goodness-Based Learning
- **Theory**: Simplified forward-forward without negative samples
- **Expected**: Simpler than full forward-forward, may be more stable
- **Memory**: O(1) - single forward pass + goodness computation
- **Speed**: Fast - ~1.5x forward pass
- **Note**: Maximizes state activation magnitude

### Perturbation / SPSA
- **Theory**: Finite differences with noise injection
- **Expected**: Noisy but unbiased gradient estimates
- **Memory**: O(1) - no computation graph
- **Speed**: VERY SLOW - O(n_samples) forward passes (10-100x slower)
- **Note**: Use SPSA variant (faster), skip standard perturbation

## Success Criteria

**IMPORTANT**: Success is NOT about matching BPTT gradients - these methods don't compute gradients! Success is about whether the learning signal enables TTT to adapt effectively.

**Minimum Viable Success (any method):**
- No NaNs or Infs in pseudo-gradients
- Stable across 200 token sequence
- Provides consistent learning signal (not random noise)
- Model runs without crashes

**Strong Success (requires actual training):**
- Model loss decreases during TTT adaptation
- Reconstruction accuracy improves over sequence
- Comparable performance to BPTT in downstream tasks

**NOTE**: Cosine similarity to BPTT is informative but NOT the success metric. Alternative methods may provide different but equally valid learning signals.

## Testing Priority

1. **Start with Feedback Alignment** - Most proven, 80-95% of backprop performance in literature
2. **Try DFA** - Simpler variant, sometimes better
3. **Test Goodness** - Simplest implementation, may be most stable
4. **Try Forward-Forward** - More experimental but interesting
5. **SPSA** - If others fail, this provides true (noisy) gradients
6. **Skip Perturbation** - Too slow, SPSA is better

## Troubleshooting

### If you see errors about missing modules:
```bash
# In Colab, pull latest changes
!git pull origin surrogate-ttt

# Restart runtime (Runtime → Restart runtime)
```

### If pseudo-gradients are NaN/Inf:
- Check numerical stability flags in test output
- Try feedback alignment or DFA (most stable)
- Reduce sequence length for testing
- Check implementation files for bugs

### If learning signals are random noise:
- Verify you're using the new methods (check `model.method` attribute)
- Check that git pull succeeded
- Try multiple methods to see if pattern is consistent

## What to Report Back

For each method tested, report:
1. **Method name** (feedback, dfa, forward_forward, goodness, spsa)
2. **Stability** (NaN/Inf present? Crashes?)
3. **Learning signal characteristics** (range, mean, std)
4. **Any errors or warnings**
5. **Subjective assessment** (does it look like valid learning signal?)

Example good report:
```
Feedback: Stable, signal ∈ [-0.2, 0.3], looks consistent
DFA: Stable, signal ∈ [-0.1, 0.4], similar to feedback
Forward-Forward: Stable, signal ∈ [0.0, 0.5], positive bias (expected)
Goodness: Stable, signal ∈ [-0.3, 0.2], symmetric distribution
SPSA: Noisy but stable, signal ∈ [-0.5, 0.5], high variance (expected)
```

## Next Steps After Testing

**If at least one method provides stable learning signals:**
1. **Actual training test**: Run TTT adaptation loop and measure loss
2. **Compare methods**: Which provides best learning signal?
3. **Scaling test**: Try longer sequences (500, 1000 tokens)
4. **Document results**: Update failure analysis with what works

**If all methods fail or produce noise:**
1. **Debug**: Check implementation for bugs
2. **Theory check**: Do these methods even apply to TTT's reconstruction task?
3. **Alternative ideas**:
   - Meta-learning the adaptation rule
   - Evolutionary strategies for state updates
   - Reinforcement learning for gradient-free optimization
4. **Document**: Update analysis with why alternative methods failed

**Important**: Alternative learning mechanisms may work differently than expected for TTT. Be open to interpreting results creatively.

## Files Reference

- **Alternative method implementations**:
  - `ueaj/model/ttt/impl_feedback.py` - Feedback alignment variants
  - `ueaj/model/ttt/impl_forward_forward.py` - Forward-forward variants
  - `ueaj/model/ttt/impl_perturbation.py` - Perturbation variants
- **Test framework**: `ueaj/model/ttt/test_framework.py`
- **Model setup**: `ueaj/model/ttt/module.py`
- **Failure analysis**: `ALTERNATIVE_METHODS_FAILURE_ANALYSIS.md`

## Background: Why Alternative Learning Mechanisms?

See `ALTERNATIVE_METHODS_FAILURE_ANALYSIS.md` for full details. Summary:

**The fundamental problem**: dk gradients require full temporal dependency graph from k_t through all future timesteps to final output. This is mathematically impossible to approximate without O(n) memory per timestep.

**Failed approaches**:
- Surrogate gradients: dk showed random noise
- Implicit differentiation: Hessian ill-conditioned, diverged
- Hybrid chunking: Gradient interference at block boundaries
- Forward-mode: Circular buffer lost gradient information

**Why alternative mechanisms**: Rather than approximate BPTT (impossible), we try entirely different learning algorithms that don't rely on backpropagation through time. These may provide weaker but usable learning signals for TTT adaptation.
