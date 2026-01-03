# Video Mate Deployment Guide (Hetzner CPX41 Edition)

This guide provides step-by-step instructions to deploy Video Mate on a **Hetzner Cloud CX43** server. This server was chosen for its excellent price/performance ratio (8 vCPUs, 16GB RAM) which is critical for handling the AI models without crashing.

## Prerequisites

*   A **Hetzner Cloud** account.
*   The **Video Mate** source code on your local computer.
*   Use of **Terminal** (Mac) or **PowerShell** (Windows).

---

## Phase 0: Verify Locally (Optional but Recommended)

Before renting a server, you can simulate the entire deployment on your local Mac to ensure there are no errors.

1.  **Run Simulation Script**:
    I have created a script that creates a fake "server" folder (`../video-mate-deploy-test`), copies your code (simulating the transfer), and runs the Docker containers.
    
    ```bash
    chmod +x simulate_deploy.sh
    ./simulate_deploy.sh
    ```

2.  **Verify**:
    *   Wait for the containers to start.
    *   Open `http://localhost:3000`.
    *   Try processing a small video.
    *   *Note: On Apple Silicon (M1/M2/M3), Docker might use emulation for some x86 parts, but it verifies the logic works.*

3.  **Cleanup**:
    When finished testing:
    ```bash
    docker compose -p videomate-test down
    ```

---

## Phase 1: Create Your Server

1.  Log in to the [Hetzner Cloud Console](https://console.hetzner.cloud).
2.  Click **New Project** and name it `VideoMate`.
3.  Click **Add Server**:
    *   **Location**: Choose `Falkenstein` or `Nuremberg` (often cheapest) or `Ashburn, VA` (if you are in the US).
    *   **Image**: Choose **Ubuntu 22.04**.
    *   **Type**: Select **Standard (Intel/AMD)** -> **CX43**.
        *   *Specs: 8 vCPUs, 16 GB RAM.*
        *   *Cost: ~€9.49/month (hourly billing).*
    *   **Networking**: Keep default (IPv4 + IPv6).
    *   **SSH Keys**:
        *   If you don't have one, click "Add SSH Key", upload your public key (usually `~/.ssh/id_rsa.pub` on Mac).
        *   *Tip: If you don't know how to create separate keys, you can just use `ssh-keygen -t ed25519` on your Mac terminal first.*
    *   **Name**: Call it `videomate-demo`.
4.  Click **Create & Buy now**.

---

## Phase 2: Server Setup

1.  **Get IP Address**: Copy the IPv4 address from the Hetzner dashboard (e.g., `123.45.67.89`).
2.  **Connect via SSH**:
    Open your terminal and run:
    ```bash
    ssh root@<YOUR_SERVER_IP>
    ```
    *Type `yes` if asked about authenticity.*

3.  **Install Docker**:
    Copy and paste this entire block into the server terminal to install Docker automatically:
    ```bash
    # Update system
    apt-get update && apt-get upgrade -y
    
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    
    # Install Docker Compose
    apt-get install -y docker-compose-plugin
    
    # Verify installation
    docker compose version
    ```

---

## Phase 3: Transfer Code

We will copy your project from your local computer to the server.

1.  **On your Local Computer**:
    Navigate to your project folder:
    ```bash
    cd /Users/eusholli/dev/video-mate
    ```

2.  **Copy Files**:
    Use `rsync` to copy the code. This excludes heavy/useless folders like `node_modules` or local databases (`ds/`).
    
    *Replace `<YOUR_SERVER_IP>` with your actual server IP.*
    
    ```bash
    rsync -avz --exclude 'node_modules' --exclude '.next' --exclude 'backend/ds' --exclude '.git' \
    ./ root@<YOUR_SERVER_IP>:~/video-mate
    ```

---

## Phase 4: Configure & Download Models

**Back on the Server** (SSH session):

1.  **Enter Project Directory**:
    ```bash
    cd ~/video-mate
    ```

2.  **Create Environment File**:
    Create the `.env` file for the backend.
    ```bash
    cp backend/.env.example backend/.env
    nano backend/.env
    ```
    
    **Edit the file** (use Arrow keys) to set your keys and paths. It MUST look like this:
    
    ```ini
    # --- API KEYS (Required) ---
    OPENAI_API_KEY=sk-...your-key-here...
    DASHSCOPE_API_KEY=sk-...your-key-here...
    
    # --- PATHS (Do NOT Change) ---
    BASE_STORAGE_PATH=/app/ds
    IMAGE_BIND_MODEL_PATH=/app/models/imagebind_huge.pth
    
    # --- MODELS ---
    CAPTION_MODEL=qwen-vl-max
    ASR_MODEL=paraformer-realtime-v1
    ANALYSIS_MODEL=gpt-4o
    PROCESSING_MODEL=gpt-4o-mini
    ```
    *Press `Ctrl+X`, then `Y`, then `Enter` to save.*

3.  **Download ImageBind Model**:
    We download the 4.5GB model directly to the server.
    ```bash
    mkdir -p models
    
    # Download (this is fast on Hetzner, approx 30-60 seconds)
    wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth -O models/imagebind_huge.pth
    ```

---

## Phase 5: Start & Ingest

1.  **Start Services**:
    Build and launch the containers.
    ```bash
    docker compose up -d --build
    ```
    *This will take 2-5 minutes to build the initial images.*

2.  **Check Status**:
    Monitor the backend logs. You are waiting for "Server initialized".
    ```bash
    docker compose logs -f backend
    ```
    *Press `Ctrl+C` to exit existing logs.*

3.  **Access the App**:
    Open your browser and go to: `http://<YOUR_SERVER_IP>:3000`

4.  **Re-Ingest Videos**:
    *   The library will be empty (because we didn't copy your local database).
    *   Click **"Add Video"**.
    *   Upload your video files again.
    *   **Wait**: Formatting and processing on CPU takes time. A 1-minute video might take 3-5 minutes to process.
    *   *Tip: Upload all your demo videos and let it run overnight if needed.*

---

## Management Tips

*   **View Logs**: `docker compose logs -f`
*   **Restart**: `docker compose restart`
*   **Stop**: `docker compose down`
*   **Update Code**:
    1.  Make changes locally.
    2.  Run the `rsync` command (Phase 3) again.
    3.  On server: `docker compose up -d --build --no-deps backend frontend`

## Troubleshooting

*   **"Video found but file missing"**: This happens if you accidentally copied your local `backend/ds` folder. Run `rm -rf backend/ds` on the server and restart.
*   **Server Out of Memory**: If the backend crashes during ingestion, ensure you selected the **CPX41 (16GB)** server. The CPX31 (8GB) is too small.
