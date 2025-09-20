#!/bin/bash

# Convenient wrapper script to run generation with proper cache setup
# Usage: ./run_with_cache.sh [script_name] [script_args...]

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to 3-agent generation script if no script specified
SCRIPT_NAME="${1:-generate_images_3_agent.py}"
shift 2>/dev/null || true  # Remove first argument if it exists

echo "=== Running with HuggingFace Cache Setup ==="

# Load environment variables (including HF_TOKEN)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    echo "Loading environment variables from .env..."
    set -a  # Automatically export variables
    source "$PROJECT_ROOT/.env"
    set +a

    # Validate HF_TOKEN
    if [[ -z "$HF_TOKEN" || "$HF_TOKEN" == "your_huggingface_token_here" ]]; then
        echo "ERROR: Please set your HuggingFace token in .env file"
        echo "Get your token from: https://huggingface.co/settings/tokens"
        echo "Then set: HF_TOKEN=your_actual_token"
        exit 1
    fi

    # Set HuggingFace Hub token for authentication
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HF_HUB_TOKEN="$HF_TOKEN"
    echo "HuggingFace authentication configured"
else
    echo "WARNING: .env file not found at $PROJECT_ROOT/.env"
    echo "You may encounter authentication issues with gated models"
fi

echo ""

# Setup cache environment
HELPERS_DIR="$(dirname "$SCRIPT_DIR")/helpers"
if [[ -f "$HELPERS_DIR/setup_cache.sh" ]]; then
    echo "Setting up cache environment..."
    source "$HELPERS_DIR/setup_cache.sh"
else
    echo "Warning: helpers/setup_cache.sh not found, using default cache"
fi

echo ""
echo "Cache status:"
if [[ -n "$HF_HOME" ]]; then
    echo "  HF_HOME: $HF_HOME"
    echo "  HF_HUB_CACHE: $HF_HUB_CACHE"
    if [[ -d "$HF_HOME" ]]; then
        CACHE_SIZE=$(du -sh "$HF_HOME" 2>/dev/null | cut -f1 || echo "unknown")
        echo "  Current cache size: $CACHE_SIZE"
    else
        echo "  Cache directory will be created"
    fi
else
    echo "  Using default HuggingFace cache location"
fi

echo ""
echo "=== Running: python $SCRIPT_NAME $* ==="
echo ""

# Change to script directory and run
cd "$SCRIPT_DIR"

# Run the specified script with any additional arguments
python "$SCRIPT_NAME" "$@"

echo ""
echo "=== Generation Complete ==="

# Show final cache size
if [[ -n "$HF_HOME" && -d "$HF_HOME" ]]; then
    FINAL_SIZE=$(du -sh "$HF_HOME" 2>/dev/null | cut -f1 || echo "unknown")
    echo "Final cache size: $FINAL_SIZE"
fi