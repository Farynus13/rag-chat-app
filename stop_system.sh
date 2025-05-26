#!/bin/bash

echo "Stopping Docker container..."
docker stop rag-chat
docker rm rag-chat

echo "Killing ngrok tmux session..."
tmux kill-session -t ngrok 2>/dev/null || echo "No ngrok tmux session found."

echo "Done."
