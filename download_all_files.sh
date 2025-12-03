#!/bin/bash

USERNAME="stiten"
REMOTE_HOST="ssh.rc.byu.edu"
REMOTE_DIR="groups/grp_quant/database/production"
LOCAL_DIR="/Users/seantitensor/Documents/silverfund/data/database/prod"

# Download all files using scp
echo "Downloading files from $REMOTE_HOST:$REMOTE_DIR to $LOCAL_DIR"
scp -r "$USERNAME@$REMOTE_HOST:$REMOTE_DIR/*" "$LOCAL_DIR"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully"
else
    echo "Error occurred during download"
    exit 1
fi