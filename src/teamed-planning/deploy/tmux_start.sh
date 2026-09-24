#!/usr/bin/env bash

SESSION="ad"
DEPLOY="/home/ad/Documents/p5/codegit/Teamed-Planning/src/teamed-planning/deploy"
DOCKER_DIR="/home/ad/Documents/p5/codegit/Teamed-Planning/src/teamed-planning/docker"

# Attach if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists, attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

# Create new session (pane 0)
tmux new-session -d -s "$SESSION"

# Pane 1 (bottom-left): deploy folder
tmux split-window -v -t "$SESSION:0.0"

# Pane 2 (bottom-middle): teamedplanning-ad docker
tmux split-window -h -t "$SESSION:0.1"

# Pane 3 (bottom-right): xtdrone-ad docker
tmux split-window -h -t "$SESSION:0.2"

# Apply main-horizontal layout: pane 0 is large on top, panes 1-3 on bottom row
tmux select-layout -t "$SESSION" main-horizontal

# Send commands to each pane
# Pane 0: deploy folder (main working pane)
tmux send-keys -t "$SESSION:0.0" "cd $DEPLOY" Enter

# Pane 1: deploy folder (second shell)
tmux send-keys -t "$SESSION:0.1" "cd $DEPLOY" Enter

# Pane 2: start teamedplanning-ad docker
tmux send-keys -t "$SESSION:0.2" "cd $DOCKER_DIR && ./run_ad.bash teamedplanning-ad:v1" Enter

# Pane 3: start xtdrone-ad docker
tmux send-keys -t "$SESSION:0.3" "cd $DOCKER_DIR && ./xtdron-ad-docker.bash" Enter

# Return focus to pane 0
tmux select-pane -t "$SESSION:0.0"

tmux attach -t "$SESSION"
