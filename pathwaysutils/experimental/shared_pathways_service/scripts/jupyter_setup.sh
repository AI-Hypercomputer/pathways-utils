#!/bin/bash
# jupyter_setup.sh

# Update and install dependencies
sudo apt update > /dev/null
pip3 install jupyterlab > /dev/null

# Launch Jupyter Lab
# We use {PORT} as a placeholder to be replaced by Python
echo "Starting Jupyter Lab on port {PORT}..."
jupyter lab --allow-root --ip=127.0.0.1 --port={PORT}
