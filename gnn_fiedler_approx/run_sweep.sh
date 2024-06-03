#!/bin/bash

CONFIG_FILE="config/full_sweep.yaml"
PROJECT="gnn_fiedler_approx"

# Create the sweep and extract the sweep ID
SWEEP_ID=$(wandb sweep --project $PROJECT $CONFIG_FILE 2>&1 >/dev/null | grep 'Run' | awk '{print $8}')

# Check if the sweep ID was extracted successfully
if [ -z "$SWEEP_ID" ]; then
  echo "Failed to create sweep or extract sweep ID."
  exit 1
fi

echo "Sweep created with ID: $SWEEP_ID"

# Run the wandb agent with the extracted sweep ID
wandb agent $SWEEP_ID
