#!/bin/bash
# Variables
PORT="{PORT}"
BUCKET_NAME="{BUCKET}"
POD_PATTERN="{WORKLOAD}"

# Paths
WORKSPACE_DIR="$HOME/vscode_workspace"
mkdir -p "$WORKSPACE_DIR"

# 1. Install Dependencies
echo "Step 1: Installing dependencies..."
sudo apt-get update -qq >/dev/null
sudo apt-get install -y -qq inotify-tools curl >/dev/null

if ! command -v code-server &> /dev/null; then
    echo "Installing code-server..."
    curl -fsSL https://code-server.dev/install.sh | sh >/dev/null 2>&1
fi

if ! command -v gsutil &> /dev/null; then
    echo "Installing Google Cloud SDK..."
    sudo apt-get install -y -qq apt-transport-https ca-certificates gnupg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - >/dev/null 2>&1
    sudo apt-get update -qq >/dev/null && sudo apt-get install -y -qq google-cloud-cli >/dev/null
fi

# 2. Bucket Logic for Syncing Workspace
if [ -n "$BUCKET_NAME" ]; then
    GCS_PATH="gs://$BUCKET_NAME/vscode_workspaces/$POD_PATTERN"
    echo "Step 2: Syncing with: $GCS_PATH"
    
    # --- Initial Recovery or Init ---
    echo "   -> Checking GCS for existing files..."
    if gsutil ls "$GCS_PATH" >/dev/null 2>&1; then
        echo "   ✅ Found files! Downloading..."
        gsutil -m rsync -r "$GCS_PATH" "$WORKSPACE_DIR"
    else
        echo "   ⚠️  No existing workspace found. initializing..."
        # Create a placeholder and upload immediately so directory appears in GCS
        touch "$WORKSPACE_DIR/.workspace_init"
        gsutil cp "$WORKSPACE_DIR/.workspace_init" "$GCS_PATH/.workspace_init"
        echo "   ✅ Initialized GCS directory with placeholder."
    fi
    # --- Background Watcher Process ---
    (
        echo "Watcher started at $(date)" > $HOME/sync.log
        while true; do
            inotifywait -r -e modify,create,delete,move "$WORKSPACE_DIR" >> $HOME/sync.log 2>&1
            echo "[AutoSync] Change detected. Waiting 5s..." >> $HOME/sync.log
            sleep 5
            
            echo "[AutoSync] Syncing (Mirroring)..." >> $HOME/sync.log
            # Explicitly use -r (recursive) and -d (delete)
            gsutil -m rsync -r -d "$WORKSPACE_DIR" "$GCS_PATH" >> $HOME/sync.log 2>&1
            
            echo "[AutoSync] Done at $(date)" >> $HOME/sync.log
        done
    ) &
else
    echo "Step 2: No bucket provided. Skipping sync."
fi

# 3. Launch VS Code
# Kill any existing remote process on this port
kill -9 $(lsof -t -i:$PORT) 2>/dev/null
echo "Step 3: Launching VS Code..."
echo "   -> Local URL: http://127.0.0.1:$PORT"

code-server --bind-addr 127.0.0.1:$PORT --auth none "$WORKSPACE_DIR"
