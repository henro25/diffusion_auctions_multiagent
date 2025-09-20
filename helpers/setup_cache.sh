#!/bin/bash

# HuggingFace Cache Setup for Cluster Environment
# This script sets up local SSD caching to avoid slow NFS downloads

set -e

# Get the current user
USER=${USER:-$(whoami)}

# Try different potential local SSD paths in order of preference
CACHE_PATHS=("/scratch/$USER" "/tmp/$USER" "/dev/shm/$USER" "/var/tmp/$USER")

echo "=== HuggingFace Cache Setup ==="

# Find the best available local path
CACHE_BASE=""
for path in "${CACHE_PATHS[@]}"; do
    parent_dir=$(dirname "$path")
    if [[ -w "$parent_dir" ]]; then
        CACHE_BASE="$path"
        echo "Selected cache location: $CACHE_BASE"
        break
    fi
done

if [[ -z "$CACHE_BASE" ]]; then
    echo "Error: No writable local directory found. Tried: ${CACHE_PATHS[*]}"
    echo "Please specify a local SSD path manually:"
    echo "export HF_CACHE_BASE=/your/local/ssd/path"
    exit 1
fi

# Override if user provides custom path
if [[ -n "$HF_CACHE_BASE" ]]; then
    CACHE_BASE="$HF_CACHE_BASE"
    echo "Using user-specified cache location: $CACHE_BASE"
fi

# Create cache directories
HF_CACHE_DIR="$CACHE_BASE/hf-cache"
mkdir -p "$HF_CACHE_DIR/hub"
mkdir -p "$HF_CACHE_DIR/transformers"

echo "Created cache directories:"
echo "  Base: $HF_CACHE_DIR"
echo "  Hub: $HF_CACHE_DIR/hub"
echo "  Transformers: $HF_CACHE_DIR/transformers"

# Export environment variables
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"

echo ""
echo "=== Environment Variables Set ==="
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

# Create a persistent environment file
ENV_FILE="$HF_CACHE_DIR/hf_env.sh"
cat > "$ENV_FILE" << EOF
# HuggingFace cache environment variables
# Source this file: source $ENV_FILE
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
EOF

echo ""
echo "=== Cache Setup Complete ==="
echo "Environment saved to: $ENV_FILE"
echo ""
echo "To use this cache in future sessions:"
echo "  source $ENV_FILE"
echo ""
echo "Current cache usage:"
if [[ -d "$HF_CACHE_DIR" ]]; then
    du -sh "$HF_CACHE_DIR" 2>/dev/null || echo "  Cache directory is empty"
fi