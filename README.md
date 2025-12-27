# Video Mate

A beautifully designed, intelligent video analytics and RAG (Retrieval-Augmented Generation) agent.

## Repository Structure

- **`frontend/`**: Next.js web application (React, Tailwind CSS, TypeScript).
- **`backend/`**: Python API service (Flask, Video Processing, ImageBind).

## Getting Started

### 1. Backend Setup

Prerequisites: Python 3.11, FFmpeg.

```bash
cd backend
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
# 1. Install standard dependencies
pip install -r requirements.txt

# 2. Install ImageBind separately with the flag
pip install --no-deps "git+https://github.com/facebookresearch/ImageBind.git@3fcf5c9039de97f6ff5528ee4a9dce903c5979b3"

# Run the API server
python videorag_api.py
```
The backend will start on port `64451` (by default).

### 2. Frontend Setup

Prerequisites: Node.js 18+.

```bash
cd frontend
# Install dependencies
npm install

# Run development server
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the application.

### 3. Docker Setup (Backend)

You can also run the backend using Docker:

```bash
cd backend
docker build -t videomate-backend .
# Mount the models directory (requires models to be present locally)
docker run -p 64451:64451 -v "$(pwd)/ds/models:/app/ds/models" videomate-backend
```

> **Note:** The `ds/models` directory is excluded from the image to keep it lightweight. You must mount your local models directory as shown above.




## Features

- **Video Ingestion**: Upload and index videos for visual and semantic search.
- **AI Chat**: Ask questions about your video content.
- **Smart Auto-Load**: Automatically manages heavy AI model loading in the background.
- **Visual Graph**: Knowledge graph construction for video understanding.
