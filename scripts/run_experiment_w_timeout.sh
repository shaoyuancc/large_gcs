#!/bin/bash

TIMEOUT_DURATION=10s

# Note: on Mac you'll need to install coreutils to get timeout `brew install coreutils`
# Run the Python script with a timeout of 60 seconds
timeout $TIMEOUT_DURATION python3 ../experiments/run_contact_graph_experiment.py --config-name cg_gcs_astar_conv_res.yaml

# Check the exit status
if [ $? -eq 124 ]; then
  echo "The script timed out after $TIMEOUT_DURATION"
else
  echo "The script completed."
fi
