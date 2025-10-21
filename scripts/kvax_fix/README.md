# KVAX + JAX 0.8.0 Compatibility Fix

This directory contains an automated fix for the kvax package to work with JAX 0.8.0+.

## Problem

JAX 0.8.0 removed the deprecated `jax.experimental.shard_map.Specs` API, which kvax relies on. This causes an `AttributeError` when importing models that use kvax flash attention.

## Solution

This directory contains:
- `typing.py.fixed` - A patched version of kvax's typing.py that uses the modern `jax.sharding.PartitionSpec` API

## Usage

To apply the fix, simply run:

```bash
./scripts/fix_kvax_jax.sh
```

The script will:
1. Locate your kvax installation automatically
2. Create a backup of the original file (first time only)
3. Apply the compatibility fix
4. Verify the fix works correctly

## When to Use

Run this script whenever:
- You reinstall the kvax package
- You get the error: `AttributeError: module 'jax.experimental.shard_map' has no attribute 'Specs'`
- After upgrading JAX to version 0.8.0 or higher

## Files Modified

- `/usr/local/lib/python3.12/dist-packages/kvax/utils/typing.py`

A backup of the original is saved as `typing.py.backup-original` in the same directory.

## Technical Details

The fix replaces:
```python
Specs = jax.experimental.shard_map.Specs  # ❌ Removed in JAX 0.8.0
```

With:
```python
from jax.sharding import PartitionSpec
Specs = PartitionSpec  # ✅ Modern JAX API
```

This maintains API compatibility while using JAX's current sharding primitives.
