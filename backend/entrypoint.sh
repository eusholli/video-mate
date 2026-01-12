#!/bin/bash
set -e

# Model Path (Default to env or hardcoded fallback)
MODEL_PATH="${IMAGE_BIND_MODEL_PATH:-/app/models/imagebind_huge.pth}"
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_URL="https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"

echo "🚀 Starting Video Mate Backend..."

# check if model exists
if [ -f "$MODEL_PATH" ]; then
    echo "✅ ImageBind model found at $MODEL_PATH"
else
    echo "⚠️  ImageBind model NOT found at $MODEL_PATH"
    echo "⬇️  Downloading from $MODEL_URL..."
    
    mkdir -p "$MODEL_DIR"
    
    # Attempt download with curl (since we'll install it in Dockerfile)
    if command -v curl >/dev/null 2>&1; then
        curl -L "$MODEL_URL" -o "$MODEL_PATH"
    elif command -v wget >/dev/null 2>&1; then
        # Fallback to wget if available
        wget "$MODEL_URL" -O "$MODEL_PATH"
    else
        echo "❌ Error: Neither curl nor wget found. Cannot download model."
        exit 1
    fi
    
    if [ -f "$MODEL_PATH" ]; then
        echo "✅ Download complete."
    else
        echo "❌ Download failed."
        exit 1
    fi
fi

# Execute the main application
echo "🎬 Launching API..."
exec python videorag_api.py
