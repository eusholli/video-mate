#!/bin/bash
set -e

# Local Deployment Verification Script (Coolify Simulation)
# This tests if the Docker container correctly downloads the model and starts up.

echo "🚀 Starting Local Coolify Simulation..."

# 1. Create a Clean 'Remote' Directory
TEST_DIR="../video-mate-deploy-test"
echo "📂 Creating clean test directory at: $TEST_DIR"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# 2. Copy Code
echo "📦 Copying source code..."
rsync -av --exclude 'node_modules' --exclude '.next' --exclude 'backend/ds' --exclude '.git' --exclude 'models' --exclude 'venv' --exclude '.venv' --exclude '__pycache__' \
    ./ "$TEST_DIR"

# 3. Setup Environment
cd "$TEST_DIR"
echo "⚙️  Configuring Environment..."
# Copy keys
if [ -f "../video-mate/backend/.env" ]; then
    cp "../video-mate/backend/.env" "backend/.env"
else
    echo "⚠️  Could not find backend/.env! Please ensure it exists."
    exit 1
fi

# Force paths to match what Coolify would use (mapped volumes)
# We set IMAGE_BIND_MODEL_PATH to /app/models/imagebind_huge.pth
# And map a local folder to /app/models
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's|^BASE_STORAGE_PATH=.*|BASE_STORAGE_PATH=/app/ds|' backend/.env
    sed -i '' 's|^IMAGE_BIND_MODEL_PATH=.*|IMAGE_BIND_MODEL_PATH=/app/models/imagebind_huge.pth|' backend/.env
else
    sed -i 's|^BASE_STORAGE_PATH=.*|BASE_STORAGE_PATH=/app/ds|' backend/.env
    sed -i 's|^IMAGE_BIND_MODEL_PATH=.*|IMAGE_BIND_MODEL_PATH=/app/models/imagebind_huge.pth|' backend/.env
fi

# 4. Create Empty Volume Directories
mkdir -p models
mkdir -p backend/ds

# 5. Build & Run Docker
echo "🐳 Building Docker Images..."
docker compose -p videomate-test build

echo "🧪 Starting Container (Testing Auto-Download)..."
echo "   NOTE: The first run should take time to download the 4.5GB model."
echo "   We are NOT pre-downloading it. The container must do it."

docker compose -p videomate-test up -d backend

echo "⏳ Tailing logs to verify download..."
echo "   Press Ctrl+C once you see 'Server initialized' or if it fails."
docker compose -p videomate-test logs -f backend

# Instructions for full stack
echo ""
echo "✅ If backend started successfully:"
echo "   1. Start frontend: docker compose -p videomate-test up -d frontend"
echo "   2. Visit: http://localhost:3000"
echo "   3. Cleanup: docker compose -p videomate-test down"
