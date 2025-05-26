#!/bin/bash

# Start Docker container in detached mode
echo "Starting Docker container..."
docker run -d -p 10000:10000 -e GEMINI_API_KEY=$GEMINI_API_KEY --name rag-chat rag-chat

# Check if tmux session 'ngrok' already exists
tmux has-session -t ngrok 2>/dev/null

if [ $? != 0 ]; then
    echo "Starting ngrok in tmux session..."
    tmux new-session -d -s ngrok "systemd-inhibit --why='Keep system awake for ngrok' --what=sleep ngrok http --url=real-large-cricket.ngrok-free.app 10000 > ~/ngrok.log 2>&1"
else
    echo "tmux session 'ngrok' already exists, not starting another"
fi

echo "Done."
