#!/bin/bash
export BUCKET_NAME="$(grep BUCKET_NAME ~/.bashrc | cut -d '=' -f2 | tr -d '"')"
export SERVICE_ACCOUNT_JSON_KEY="$(grep SERVICE_ACCOUNT_JSON_KEY ~/.bashrc | cut -d '=' -f2 | tr -d '"')"
# Change to the script directory
cd /home/cephla-ps/Desktop/octopi-malaria-gui

# Run the Python script
python3 run.py
