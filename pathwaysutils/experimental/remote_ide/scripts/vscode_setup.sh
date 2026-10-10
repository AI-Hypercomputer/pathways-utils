#!/bin/bash
if [ -w /proc/1/fd/1 ]; then
    exec > >(tee /proc/1/fd/1) 2> >(tee /proc/1/fd/2 >&2)
fi

PORT="{PORT}"
BUCKET="{BUCKET}"
WORKLOAD="{WORKLOAD}"
ALIASES="{BASH_ALIASES_BASE64}"

VSCODE_USER_DIR="$HOME/.local/share/code-server/User"
VSCODE_EXT_DIR="$HOME/.local/share/code-server/extensions"
SHELL_HIST_FILE="$HOME/.bash_history"

REMOTE_DOWNLOAD_DIR="$HOME/workspace"
mkdir -p "$REMOTE_DOWNLOAD_DIR"
cd "$REMOTE_DOWNLOAD_DIR"

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

if [ ! -z "$ALIASES" ] && [ "$ALIASES" != "{BASH_ALIASES_BASE64}" ]; then
    echo "$ALIASES" | base64 -d > "$HOME/.bash_aliases"
    if ! grep -q "\$HOME/.bash_aliases" "$HOME/.bashrc"; then
        echo 'if [ -f "$HOME/.bash_aliases" ]; then . "$HOME/.bash_aliases"; fi' >> "$HOME/.bashrc"
    fi
fi

# 0.5. INSTALL DEPENDENCIES (kubectl, gke-gcloud-auth-plugin)
echo "Ensuring system dependencies are installed..."
SUDO_CMD=""
if [ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null; then
    SUDO_CMD="sudo"
fi

if ! command -v curl &> /dev/null; then
    $SUDO_CMD apt-get update || true
    $SUDO_CMD apt-get install -y curl || true
fi

if ! command -v kubectl &> /dev/null; then
    echo "Installing kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    $SUDO_CMD mv kubectl /usr/local/bin/
fi

if ! command -v gke-gcloud-auth-plugin &> /dev/null; then
    echo "Installing gke-gcloud-auth-plugin..."
    $SUDO_CMD apt-get update || true
    $SUDO_CMD apt-get install -y google-cloud-cli-gke-gcloud-auth-plugin || $SUDO_CMD apt-get install -y google-cloud-sdk-gke-gcloud-auth-plugin || true
fi

# 1. RESTORE
if [ ! -z "$BUCKET" ] && [ "$BUCKET" != "{BUCKET}" ] && command -v gcloud &> /dev/null; then
    echo "[Restore] Checking gs://$BUCKET/$WORKLOAD/..."
    
    mkdir -p "$VSCODE_USER_DIR"
    mkdir -p "$VSCODE_EXT_DIR"
    
    if gcloud storage ls "gs://$BUCKET/$WORKLOAD/workspace/**" >/dev/null 2>&1; then
        echo "[Restore] Downloading Project Files..." 
        gcloud storage rsync "gs://$BUCKET/$WORKLOAD/workspace" "$REMOTE_DOWNLOAD_DIR" --recursive --delete-unmatched-destination-objects --exclude='^\..*'
    fi
    
    if gcloud storage ls "gs://$BUCKET/$WORKLOAD/vscode/shell_history" >/dev/null 2>&1; then
        gcloud storage cp "gs://$BUCKET/$WORKLOAD/vscode/shell_history" "$SHELL_HIST_FILE" 
    fi
    
    if gcloud storage ls "gs://$BUCKET/$WORKLOAD/vscode/User/**" >/dev/null 2>&1; then
        gcloud storage rsync "gs://$BUCKET/$WORKLOAD/vscode/User" "$VSCODE_USER_DIR" --recursive --delete-unmatched-destination-objects
    fi
    
    if gcloud storage ls "gs://$BUCKET/$WORKLOAD/vscode/extensions/**" >/dev/null 2>&1; then
        gcloud storage rsync "gs://$BUCKET/$WORKLOAD/vscode/extensions" "$VSCODE_EXT_DIR" --recursive --delete-unmatched-destination-objects
    fi
    
    if gcloud storage ls "gs://$BUCKET/$WORKLOAD/vscode/bash_aliases" >/dev/null 2>&1; then
        gcloud storage cp "gs://$BUCKET/$WORKLOAD/vscode/bash_aliases" "$HOME/.bash_aliases"
    fi
fi

# 2. SYNC SERVICE
if [ ! -z "$BUCKET" ] && [ "$BUCKET" != "{BUCKET}" ] && command -v gcloud &> /dev/null; then
    echo "[Sync] Starting background sync service..."
    cat <<EOF > /tmp/sync_loop.sh
#!/bin/bash
if [ -w /proc/1/fd/1 ]; then
    exec > >(tee /proc/1/fd/1) 2> >(tee /proc/1/fd/2 >&2)
fi
while true; do
    if [ -d "$REMOTE_DOWNLOAD_DIR" ]; then
         gcloud storage rsync "$REMOTE_DOWNLOAD_DIR" "gs://$BUCKET/$WORKLOAD/workspace" --recursive --delete-unmatched-destination-objects --exclude='^\..*' >/dev/null 2>&1
    fi
    if [ -f "$SHELL_HIST_FILE" ]; then
        gcloud storage cp "$SHELL_HIST_FILE" "gs://$BUCKET/$WORKLOAD/vscode/shell_history" >/dev/null 2>&1
    fi
    if [ -d "$VSCODE_USER_DIR" ]; then
         gcloud storage rsync "$VSCODE_USER_DIR" "gs://$BUCKET/$WORKLOAD/vscode/User" --recursive --delete-unmatched-destination-objects >/dev/null 2>&1
    fi
    if [ -d "$VSCODE_EXT_DIR" ]; then
         gcloud storage rsync "$VSCODE_EXT_DIR" "gs://$BUCKET/$WORKLOAD/vscode/extensions" --recursive --delete-unmatched-destination-objects >/dev/null 2>&1
    fi
    if [ -f "$HOME/.bash_aliases" ]; then
         gcloud storage cp "$HOME/.bash_aliases" "gs://$BUCKET/$WORKLOAD/vscode/bash_aliases" >/dev/null 2>&1
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
    curl -fsSL https://code-server.dev/install.sh | sh -s -- --method=standalone --prefix="$HOME/.local"
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! grep -q "\$HOME/.local/bin" "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

echo "Starting VS Code Server on port $PORT..."
code-server --bind-addr 0.0.0.0:$PORT --auth none --disable-telemetry --disable-update-check "$REMOTE_DOWNLOAD_DIR"