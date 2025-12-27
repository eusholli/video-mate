# VideoRAG (Video Mate)

A beautifully designed, intelligent video analytics and RAG (Retrieval-Augmented Generation) agent.

## Repository Structure

- **`frontend/`**: Next.js web application (React, Tailwind CSS, TypeScript).
- **`backend/`**: Python API service (Flask, Video Processing, ImageBind).

## Getting Started

### 1. Backend Setup

Prerequisites: Python 3.8+, FFmpeg.

```bash
cd backend
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

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

## Configuration

The application uses a configuration file located at `frontend/ds/config.json`.
On first launch, the frontend will automatically initialize the backend with settings from this file.

## Features

- **Video Ingestion**: Upload and index videos for visual and semantic search.
- **AI Chat**: Ask questions about your video content.
- **Smart Auto-Load**: Automatically manages heavy AI model loading in the background.
- **Visual Graph**: Knowledge graph construction for video understanding.
