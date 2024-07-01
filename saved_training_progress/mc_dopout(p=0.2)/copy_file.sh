#!/bin/bash

# Define source and target directories
SOURCE_DIR="/data/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task901_AUH/nnUNetTrainerV2_dropout1__nnUNetPlansv2.1/"
TARGET_DIR="."

# Loop through fold_0 to fold_9
for i in {0..4}; do
  # Define the source and target paths
  SOURCE_PATH="${SOURCE_DIR}/fold_${i}/debug.json"
  TARGET_PATH="${TARGET_DIR}/fold_${i}/"
  # Define the source and target paths for progress.png
  SOURCE_PROGRESS_PATH="${SOURCE_DIR}/fold_${i}/progress.png"
  # Create the target directory if it doesn't exist
  mkdir -p "${TARGET_PATH}"

  # Copy the debug.json file to the target directory
  cp "${SOURCE_PATH}" "${TARGET_PATH}"
  # Copy the progress.png file to the target directory
  cp "${SOURCE_PROGRESS_PATH}" "${TARGET_PATH}"
  
done
