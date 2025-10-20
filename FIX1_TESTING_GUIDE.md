# Fix 1: Testing Guide

## What Was Changed

**File**: `ueaj/model/ttt/impl.py`, line 84

**Original code** (line 82):
```python
new_k_state, dk = k_scan(k_state, (q, q+dq, k))
```

**Fix 1** (line 84):
```python
new_k_state, dk = k_scan(k_state, (q, do, k))
```

## Rationale

Make dk computation **symmetric** with dv computation:
- **dv**: uses `do` (output gradient) as reconstruction target
- **dk**: now also uses `do` (output gradient) as reconstruction target

Previously, dk used `q+dq` (perturbed query) which has no theoretical connection to the actual gradient flow.

## Expected Outcome

If this fix works:
- **dk cosine_sim**: Should improve from [-0.46, 0.36] to match dv's range [-0.52, 0.80]
- **dk mean cosine_sim**: Should be consistently positive (> 0)
- **Valid descent direction**: dk · true_dk > 0

## How to Test

### Option 1: Quick Test with Jupyter Notebook

1. Open the notebook:
   ```bash
   cd E:\Work\Projects\macrogpt-jax\ueaj\model\ttt
   jupyter notebook ttt_gradient_analysis.ipynb
   ```

2. Run all cells (Cell → Run All)

3. Check the outputs:
   - **Cell 6**: Look at "Cosine similarity ranges" for k
   - **Cell 7**: View the gradient comparison plot
   - **Cell 8**: View RMS error plots

4. Look for improvement:
   - Is dk cosine_sim now mostly positive?
   - Is it in the same range as dv?
   - Does the plot show dk above the zero line?

### Option 2: Run with Python Script

```bash
cd E:\Work\Projects\macrogpt-jax
./scripts/run_python.sh -c "
import sys
sys.path.insert(0, 'ueaj/model/ttt')
exec(open('ueaj/model/ttt/ttt_gradient_analysis.ipynb').read())
"
```

### Option 3: Create Simple Test Script

Create `test_fix1.py`:
```python
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
import sys
sys.path.insert(0, '../../../')

from ueaj.model.ttt import TTTModel
from ueaj.model import GMLP

# Config
model_d, hidden_d = 128, 512
batch_size, seq_len = 1, 200
seed = 42

# Create models
print("Creating models...")
model_surr = TTTModel(model_d, hidden_d, GMLP, surrogate=True, rngs=rng.Rngs(seed))
model_nonsurr = TTTModel(model_d, hidden_d, GMLP, surrogate=False, rngs=rng.Rngs(seed))

# Generate k, q, v
key = jax.random.PRNGKey(123)
k_key, v_key, q_key = jax.random.split(key, 3)
k = jax.random.normal(k_key, (batch_size, seq_len, hidden_d))
v = jax.random.normal(v_key, (batch_size, seq_len, hidden_d))
q = jax.random.normal(q_key, (batch_size, seq_len, hidden_d))

# Random output gradient
grad_output = jax.random.normal(jax.random.PRNGKey(456), (batch_size, seq_len, hidden_d))

def compute_grads(model, k, v, q, grad_output):
    def forward(k, v, q):
        return model.apply_ttt(k, v, q)

    output, vjp_fn = jax.vjp(forward, k, v, q)
    grad_k, grad_v, grad_q = vjp_fn(grad_output)
    return grad_k, grad_v, grad_q

print("Computing gradients...")
gk_surr, gv_surr, gq_surr = compute_grads(model_surr, k, v, q, grad_output)
gk_nonsurr, gv_nonsurr, gq_nonsurr = compute_grads(model_nonsurr, k, v, q, grad_output)

# Compute cosine similarity
def cosine_similarity_per_token(grad_surr, grad_nonsurr):
    dot_product = jnp.sum(grad_surr * grad_nonsurr, axis=2)
    norm_surr = jnp.linalg.norm(grad_surr, axis=2)
    norm_nonsurr = jnp.linalg.norm(grad_nonsurr, axis=2)
    cosine_sim = dot_product / (norm_surr * norm_nonsurr + 1e-8)
    return cosine_sim.mean(axis=0)

cos_k = cosine_similarity_per_token(gk_surr, gk_nonsurr)
cos_v = cosine_similarity_per_token(gv_surr, gv_nonsurr)

print("\n" + "="*60)
print("RESULTS: Fix 1 - Symmetric Treatment")
print("="*60)
print(f"dk cosine similarity: [{cos_k.min():.4f}, {cos_k.max():.4f}]")
print(f"dk mean cosine sim:   {cos_k.mean():.4f}")
print(f"dv cosine similarity: [{cos_v.min():.4f}, {cos_v.max():.4f}]")
print(f"dv mean cosine sim:   {cos_v.mean():.4f}")
print()

# Check if improvement
if cos_k.mean() > 0.5:
    print("✓ SUCCESS: dk mean cosine_sim > 0.5!")
    print("  Fix 1 achieves minimum viable success.")
elif cos_k.mean() > 0.3:
    print("⚠ PARTIAL: dk improved but still below target")
    print(f"  Target: > 0.5, Actual: {cos_k.mean():.4f}")
elif cos_k.mean() > 0:
    print("⚠ SLIGHT IMPROVEMENT: dk now positive on average")
    print(f"  But still far from target (> 0.5)")
else:
    print("✗ FAILURE: dk still oscillates around zero")
    print("  Fix 1 does not address the core issue.")
print("="*60)
```

Run it:
```bash
cd E:\Work\Projects\macrogpt-jax\ueaj\model\ttt
../../../scripts/run_python.sh test_fix1.py
```

## Success Criteria

### Minimum Success (proceed to next phase):
- dk mean cosine_sim > 0.5
- dk mostly positive across sequence
- Valid descent direction

### Failure (try Fix 2):
- dk mean cosine_sim < 0.3
- Still oscillates around zero
- No clear improvement

## What to Report Back

Please provide:
1. **dk cosine similarity range**: [min, max]
2. **dk mean cosine similarity**: value
3. **Comparison with original**: Did it improve?
4. **Visual check**: If using notebook, does the dk plot look better?

## Next Steps

- **If Fix 1 succeeds**: Document findings, move to test framework implementation
- **If Fix 1 fails**: Try Fix 2 (state gradient propagation)
- **If Fix 2 fails**: Try Fix 3 (remove state hacking)
- **If all fail**: Proceed to alternative methods (implicit diff, etc.)
