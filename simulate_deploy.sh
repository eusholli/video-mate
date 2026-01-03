#!/bin/bash
set -e

# Local Deployment Verification Script
# This mimics the steps you'd strictly take on the server, but locally in a separate folder.

echo "🚀 Starting Local Deployment Test..."

# 1. Create a Clean 'Remote' Directory
# We simulate the server by creating a separate folder
TEST_DIR="../video-mate-deploy-test"
echo "📂 Creating clean test directory at: $TEST_DIR"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# 2. 'Rsync' Code (Simulated with cp)
# Copying only what we sending to the server
echo "📦 Copying source code..."
rsync -av --exclude 'node_modules' --exclude '.next' --exclude 'backend/ds' --exclude '.git' --exclude 'models' \
    ./ "$TEST_DIR"

# 3. Setup Environment
echo "⚙️  Configuring Environment..."
cd "$TEST_DIR"
# Copy existing local env to backend .env for testing (assuming you have keys there)
# On the real server, you'd edit this manually as per the guide.
if [ -f "backend/.env" ]; then
    echo "   Using existing backend/.env"
else
    # Fallback to copying from original location if not caught by rsync
    cp "../video-mate/backend/.env" "backend/.env" 2>/dev/null || echo "⚠️  Could not find .env! You might need to create it manually in $TEST_DIR/backend/.env"
fi

# Ensure paths in .env match Docker paths (simulating user edit)
# We use sed to force these paths for the Docker test, just like the guide says
sed -i '' 's|^BASE_STORAGE_PATH=.*|BASE_STORAGE_PATH=/app/ds|' backend/.env
sed -i '' 's|^IMAGE_BIND_MODEL_PATH=.*|IMAGE_BIND_MODEL_PATH=/app/models/imagebind_huge.pth|' backend/.env

# 4. Download Model (Simulated)
echo "⬇️  Checking for ImageBind model..."
mkdir -p models
MODEL_URL="https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
TARGET_MODEL="models/imagebind_huge.pth"

# Optimization: Check if user already has it locally to avoid 4.5GB download every test
LOCAL_MODEL_CACHE="../imagebind_huge.pth" # Look in parent? or just download.
# For this test script, we'll try to download if missing.
if [ -f "$TARGET_MODEL" ]; then
    echo "   Model already exists."
else
    echo "   Downloading model (This may take a while)..."
    curl -L $MODEL_URL -o $TARGET_MODEL
fi

# 5. Build & Run Docker
echo "🐳 Building and Starting Docker Containers..."
# We use project name 'videomate-test' to avoid conflict with 'videomate' if running
docker compose -p videomate-test up -d --build

echo "⏳ Waiting for backend to initialize..."
echo "   Tail the logs with: docker compose -p videomate-test logs -f backend"
echo "   Once 'Server initialized', access at http://localhost:3000"
echo "   To stop: docker compose -p videomate-test down"
