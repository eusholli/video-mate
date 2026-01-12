# Video Mate Deployment Guide (Coolify Edition)

This guide describes how to deploy Video Mate using **Coolify**, a self-hosted Platform-as-a-Service (PaaS) that runs on your own VPS. This is the recommended approach for balancing cost, performance, and ease of management.

## Recommended Server

*   **Provider**: Hetzner Cloud
*   **Model**: **CX43** (Intel/AMD)
*   **Specs**: 8 vCPUs, 16 GB RAM
*   **Storage**: 160 GB NVMe
*   **Cost**: ~€9.49/month

## Phase 1: Server & Coolify Setup
---

## Phase 0: Verify Locally (Critical)

Since Coolify automates everything, you want to guarantee your Docker logic works **before** deploying.

1.  **Run the Simulation Script**:
    This script mimics a fresh server. It creates a test folder, builds your Docker image, and runs it with empty volume folders.
    
    ```bash
    chmod +x simulate_deploy.sh
    ./simulate_deploy.sh
    ```

2.  **What to Watch For**:
    *   The script will tail the logs of the backend container.
    *   **Success**: You should see `⬇️ Downloading from https://...` followed by `✅ Download complete` and then `Server initialized`.
    *   **Failure**: If it crashes or says "Model not found", do NOT deploy to Coolify yet.

3.  **Cleanup**:
    ```bash
    docker compose -p videomate-test down
    ```

---
1.  **Create Server**: 
    *   Launch a **CX43** instance on Hetzner with **Ubuntu 22.04 LTS**.
    *   Add your SSH Key.

2.  **Install Coolify**:
    *   SSH into your server: `ssh root@<YOUR_SERVER_IP>`
    *   Run the official install script:
        ```bash
        curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash
        ```
    *   Wait ~5-10 minutes.
    *   Open `http://<YOUR_SERVER_IP>:8000` in your browser.
    *   Register your admin account.

---

## Phase 2: Deploy Backend

1.  **Create Project**: in Coolify, create a new Project -> `VideoMate`.
2.  **Add Resource**: Click `+ New` -> `Git Repository`.
3.  **Connect Repo**: Select your `eusholli/video-mate` repository.
4.  **Configuration**:
    *   **Build Pack**: `Docker`
    *   **Docker File**: `backend/Dockerfile`
    *   **Port Exposes**: `64451`
5.  **Environment Variables**:
    *   Click `Environment Variables`.
    *   Add your keys:
        *   `OPENAI_API_KEY`
        *   `DASHSCOPE_API_KEY`
        *   `BASE_STORAGE_PATH` = `/app/ds`
        *   `IMAGE_BIND_MODEL_PATH` = `/app/ds/models/imagebind_huge.pth`
        *   `CAPTION_MODEL` ... (and other model configs)
6.  **Persistent Storage (Critical)**:
    *   Go to `Storage`.
    *   Add a single volume for all data (library, sessions, and models):
        *   **Volume Name**: `video-mate-data`
        *   **Destination Path**: `/app/ds`
7.  **Deploy**: Click `Deploy`.
    *   *Note*: The first deployment will take longer because the container will automatically download the 4.5GB ImageBind model to the `/app/ds/models` directory (inside your persistent volume). Check the logs to see the download progress.

---

## Phase 3: Deploy Frontend

1.  **Add Resource**: In the same Coolify environment, click `+ New` -> `Git Repository`.
2.  **Connect Repo**: Select `eusholli/video-mate` again.
3.  **Configuration**:
    *   **Build Pack**: `Nixpacks` (Recommended for Next.js) or `Docker` (using `frontend/Dockerfile`).
    *   **Base Directory**: `/frontend`
    *   **Port Exposes**: `3000`
4.  **Environment Variables**:
    *   `BACKEND_URL`: Use the internal Docker DNS from Coolify (e.g., `http://videomate-backend:64451`).
        *   *Tip*: In Coolify, look at the Backend's "Network" tab to find its internal DNS name.
5.  **Deploy**: Click `Deploy`.

---

## Phase 4: Access Application

1.  **Public Domains**:
    *   In Coolify, go to your Frontend resource -> `Settings`.
    *   Set **Domains** to `http://<YOUR_SERVER_IP>:3000` (or connect a custom domain like `https://video.example.com`).
2.  **Open**: Visit your domain.
3.  **Ingest**: Upload your videos to re-populate the library.

## Troubleshooting

*   **Model Download Failed**: Check backend logs. Ensure the server has internet access.
*   **"OOM Killed"**: Ensure you used the **CX43 (16GB RAM)** server. 8GB is risky.
*   **Connection Refused**: Ensure Frontend `BACKEND_URL` is correct. Use the container name or internal IP, not `localhost`.
