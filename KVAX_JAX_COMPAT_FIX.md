# KVAX + JAX 0.8.0 Compatibility Fix

## Date
2025-10-21

## Error

**Location**: Notebook cell importing `from ueaj.model.ttt import TTTModel`

**Error Message**:
```
AttributeError: module 'jax.experimental.shard_map' has no attribute 'Specs'
```

**Root Cause**: JAX 0.8.0 removed deprecated `jax.experimental.shard_map.Specs` API

**Import Chain**:
```
ueaj.model.ttt → ueaj.model → ueaj.model.soft_attn → kvax → kvax/utils/typing.py
```

## Fix Applied

**File Modified**: `/usr/local/lib/python3.12/dist-packages/kvax/utils/typing.py:11`

**Original Code (Broken)**:
```python
Specs = jax.experimental.shard_map.Specs  # ❌ Removed in JAX 0.8.0
```

**Fixed Code**:
```python
# Specs was removed from jax.experimental.shard_map in JAX 0.8.0
# Use PartitionSpec as the modern equivalent
from jax.sharding import PartitionSpec
Specs = PartitionSpec  # ✅ Modern JAX API
```

## Key Files

- **Error source**: `/usr/local/lib/python3.12/dist-packages/kvax/utils/typing.py`
- **Import chain**: `ueaj/model/soft_attn.py` → kvax flash attention
- **Test notebook**: `ueaj/model/ttt/ttt_gradient_analysis.ipynb`

## Notes

- This is a **system package patch** - will need reapplication if kvax is reinstalled
- kvax package may need to be updated upstream to support JAX 0.8.0+
- Alternative: Pin JAX to < 0.8.0 in requirements.txt if kvax is not updated

## Workaround for Future

If kvax is reinstalled, reapply this fix or use:

```bash
# Option 1: Pin JAX version (temporary)
pip install 'jax<0.8.0' 'jaxlib<0.8.0'

# Option 2: Patch kvax automatically after install
pip install kvax
python -c "
import kvax
import os
typing_path = os.path.join(os.path.dirname(kvax.__file__), 'utils/typing.py')
with open(typing_path, 'r') as f:
    content = f.read()
content = content.replace(
    'Specs = jax.experimental.shard_map.Specs',
    'from jax.sharding import PartitionSpec\nSpecs = PartitionSpec'
)
with open(typing_path, 'w') as f:
    f.write(content)
print('✓ kvax patched for JAX 0.8.0')
"
```

## References

- JAX 0.8.0 release notes: https://github.com/google/jax/releases/tag/jax-v0.8.0
- Modern sharding API: https://jax.readthedocs.io/en/latest/jax.sharding.html
