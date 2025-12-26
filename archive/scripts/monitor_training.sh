#!/bin/bash
# Monitor V6 training progress
LOG_FILE="/home/aldo/skullking/training_v6.log"
MODEL_FILE="/home/aldo/skullking/models/masked_ppo/masked_ppo_final.zip"

while true; do
    # Check if training is complete
    if [ -f "$MODEL_FILE" ] && grep -q "Training complete" "$LOG_FILE" 2>/dev/null; then
        echo "$(date): Training complete!"
        echo "Final model saved to: $MODEL_FILE"

        # Get final stats
        tail -20 "$LOG_FILE"
        exit 0
    fi

    # Show progress
    if [ -f "$LOG_FILE" ]; then
        LATEST=$(grep "total_timesteps" "$LOG_FILE" | tail -1 | awk -F'|' '{print $3}' | tr -d ' ')
        REWARD=$(grep "ep_rew_mean" "$LOG_FILE" | tail -1 | awk -F'|' '{print $3}' | tr -d ' ')
        echo "$(date): Steps=$LATEST, Reward=$REWARD"
    fi

    sleep 300  # Check every 5 minutes
done
