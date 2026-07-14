#!/bin/bash
if [ -w /proc/1/fd/1 ]; then
    exec > >(tee /proc/1/fd/1) 2> >(tee /proc/1/fd/2 >&2)
fi
# Variables
PORT="{PORT}"
BUCKET="{BUCKET}"
WORKLOAD="{WORKLOAD}"
# Paths
VSCODE_USER_DIR="$HOME/.local/share/code-server/User"
VSCODE_EXT_DIR="$HOME/.local/share/code-server/extensions"
SHELL_HIST_FILE="$HOME/.bash_history"
# [NEW] The directory where your code lives (Current Folder)
WORKSPACE_DIR=$(pwd)
# 0. SHELL CONFIGURATION
if ! grep -q "history -a" "$HOME/.bashrc"; then
    echo "export PROMPT_COMMAND='history -a'" >> "$HOME/.bashrc"
fi
if ! grep -q "/proc/1/fd/1" "$HOME/.bashrc"; then
    cat <<'EOF' >> "$HOME/.bashrc"
if [ -w /proc/1/fd/1 ]; then
    exec > >(tee /proc/1/fd/1) 2> >(tee /proc/1/fd/2 >&2)
fi
EOF
fi
# 1. RESTORE
if [ ! -z "$BUCKET" ] && command -v gcloud &> /dev/null; then
    echo "[Restore] Checking gs://$BUCKET/$WORKLOAD/..."
    
    mkdir -p "$VSCODE_USER_DIR"
    mkdir -p "$VSCODE_EXT_DIR"
    # A. Restore Project Files (Code, Notebooks) -- [NEW SECTION]
    if gcloud storage ls "gs://$BUCKET/$WORKLOAD/workspace/**" >/dev/null 2>&1; then
        echo "[Restore] Downloading Project Files..." 
        # Exclude hidden folders like .git or .local to save time/errors
        gcloud storage rsync "gs://$BUCKET/$WORKLOAD/workspace" "$WORKSPACE_DIR" --recursive --delete-unmatched-destination-objects --exclude="\..*"
    fi
    # B. Restore Shell History
    if gcloud storage ls "gs://$BUCKET/$WORKLOAD/vscode/shell_history" >/dev/null 2>&1; then
        gcloud storage cp "gs://$BUCKET/$WORKLOAD/vscode/shell_history" "$SHELL_HIST_FILE" 
    fi
    # C. Restore User Data
    if gcloud storage ls "gs://$BUCKET/$WORKLOAD/vscode/User/**" >/dev/null 2>&1; then
        gcloud storage rsync "gs://$BUCKET/$WORKLOAD/vscode/User" "$VSCODE_USER_DIR" --recursive --delete-unmatched-destination-objects
    fi
    # D. Restore Extensions
    if gcloud storage ls "gs://$BUCKET/$WORKLOAD/vscode/extensions/**" >/dev/null 2>&1; then
        gcloud storage rsync "gs://$BUCKET/$WORKLOAD/vscode/extensions" "$VSCODE_EXT_DIR" --recursive --delete-unmatched-destination-objects
    fi
fi
# 2. SYNC SERVICE
if [ ! -z "$BUCKET" ] && command -v gcloud &> /dev/null; then
    echo "[Sync] Starting background sync service..."
    cat <<EOF > /tmp/sync_loop.sh
#!/bin/bash
if [ -w /proc/1/fd/1 ]; then
    exec > >(tee /proc/1/fd/1) 2> >(tee /proc/1/fd/2 >&2)
fi
while true; do
    # 1. Sync Project Files (Code) -- [NEW SECTION]
    # We exclude hidden files (starts with .) and __pycache__ to keep it clean
    if [ -d "$WORKSPACE_DIR" ]; then
         gcloud storage rsync "$WORKSPACE_DIR" "gs://$BUCKET/$WORKLOAD/workspace" --recursive --delete-unmatched-destination-objects
    fi
    # 2. Sync Shell History
    if [ -f "$SHELL_HIST_FILE" ]; then
        gcloud storage cp "$SHELL_HIST_FILE" "gs://$BUCKET/$WORKLOAD/vscode/shell_history" >/dev/null 2>&1
    fi
    
    # 3. Sync User Data
    if [ -d "$VSCODE_USER_DIR" ]; then
         gcloud storage rsync "$VSCODE_USER_DIR" "gs://$BUCKET/$WORKLOAD/vscode/User" --recursive --delete-unmatched-destination-objects >/dev/null 2>&1
    fi
    # 4. Sync Extensions
    if [ -d "$VSCODE_EXT_DIR" ]; then
         gcloud storage rsync "$VSCODE_EXT_DIR" "gs://$BUCKET/$WORKLOAD/vscode/extensions" --recursive --delete-unmatched-destination-objects >/dev/null 2>&1
    fi
    sleep 30
done
EOF
    chmod +x /tmp/sync_loop.sh
    /tmp/sync_loop.sh &
fi
# 3. START VS CODE
if ! command -v code-server &> /dev/null; then
    echo "Installing code-server..."
    sudo apt update > /dev/null && sudo apt install -y curl > /dev/null
    curl -fsSL https://code-server.dev/install.sh | sh > /dev/null
fi
echo "Starting VS Code Server on port $PORT..."
code-server --bind-addr 127.0.0.1:{PORT} --auth none --disable-telemetry --disable-update-check .
