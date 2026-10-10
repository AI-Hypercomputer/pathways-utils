#!/bin/bash
# jupyter_setup.sh

# Update and install dependencies
sudo apt update > /dev/null
pip3 install jupyterlab > /dev/null

# Set up bash aliases if present
if [ ! -z "{BASH_ALIASES_BASE64}" ]; then
    echo "{BASH_ALIASES_BASE64}" | base64 -d > "$HOME/.bash_aliases"
    if [ -f "$HOME/.bashrc" ] && ! grep -q "\$HOME/.bash_aliases" "$HOME/.bashrc" && ! grep -q "~/.bash_aliases" "$HOME/.bashrc"; then
        cat <<'EOF' >> "$HOME/.bashrc"
if [ -f "$HOME/.bash_aliases" ]; then
    . "$HOME/.bash_aliases"
fi
EOF
    fi
fi

# Launch Jupyter Lab
# We use {PORT} as a placeholder to be replaced by Python
echo "Starting Jupyter Lab on port {PORT}..."
jupyter lab --allow-root --ip=0.0.0.0 --port={PORT}
