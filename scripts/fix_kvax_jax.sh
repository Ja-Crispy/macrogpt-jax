#!/bin/bash
# Script to apply JAX 0.8.0 compatibility fix to kvax package
# This fixes the AttributeError: module 'jax.experimental.shard_map' has no attribute 'Specs'

set -e  # Exit on error

echo "🔧 Applying kvax + JAX 0.8.0 compatibility fix..."

# Find kvax installation location
KVAX_PATH=$(python3 -c "import kvax; import os; print(os.path.dirname(kvax.__file__))" 2>/dev/null)

if [ -z "$KVAX_PATH" ]; then
    echo "❌ Error: kvax package not found. Please install kvax first."
    exit 1
fi

TYPING_FILE="$KVAX_PATH/utils/typing.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXED_FILE="$SCRIPT_DIR/kvax_fix/typing.py.fixed"

# Check if the fixed file exists
if [ ! -f "$FIXED_FILE" ]; then
    echo "❌ Error: Fixed typing.py not found at $FIXED_FILE"
    echo "   The backup file may have been deleted."
    exit 1
fi

# Check if target file exists
if [ ! -f "$TYPING_FILE" ]; then
    echo "❌ Error: Target file not found at $TYPING_FILE"
    exit 1
fi

# Create backup of original if it doesn't exist
BACKUP_FILE="$TYPING_FILE.backup-original"
if [ ! -f "$BACKUP_FILE" ]; then
    echo "📦 Creating backup of original file..."
    cp "$TYPING_FILE" "$BACKUP_FILE"
    echo "   Backed up to: $BACKUP_FILE"
fi

# Apply the fix
echo "✏️  Applying fix to: $TYPING_FILE"
cp "$FIXED_FILE" "$TYPING_FILE"

# Verify the fix
if python3 -c "from kvax.utils.typing import Specs; from jax.sharding import PartitionSpec; assert Specs is PartitionSpec" 2>/dev/null; then
    echo "✅ Fix applied successfully!"
    echo ""
    echo "📍 Modified file: $TYPING_FILE"
    echo "💾 Original backup: $BACKUP_FILE"
    echo ""
    echo "You can now import ueaj.model.ttt without errors."
else
    echo "⚠️  Fix applied but verification failed. Please check manually."
    exit 1
fi
