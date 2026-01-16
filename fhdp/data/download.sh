#!/bin/bash

# Download script for nuScence dataset
# Note: Full nuScence dataset requires registration and authentication

# Create output directory
mkdir -p nuscenes
cd nuscenes

# Instructions for manual download
cat << EOF
========================================
NuScence Dataset Download Instructions
========================================

The full nuScence dataset requires registration and acceptance of terms.
Please follow these steps:

1. Visit the nuScence website: https://www.nuscenes.org/download
2. Register for an account
3. Accept the terms of service
4. Download the following files:
   - v1.0-trainval_meta.tgz (Metadata)
   - v1.0-trainval_sample_data.tgz (Sample data)
   - v1.0-trainval_sweeps.tgz (Sweeps)
   - v1.0-trainval_maps.tgz (Maps)
   - v1.0-trainval_annotation.tgz (Annotations - optional)
   - v1.0-trainval_panoptic.tgz (Panoptic annotations - optional)

5. Place all downloaded files in this directory: $(pwd)

6. Run this script again to extract the files

========================================
EOF

# Check if any files have been downloaded
if [ -n "$(ls -A *.tgz 2>/dev/null)" ]; then
    echo "
Found downloaded files. Extracting..."
    
    # Extract all files
    extraction_failed=false
    
    for FILE in *.tgz; do
        echo "Extracting: $FILE"
        if ! tar -xzf "$FILE"; then
            echo "Error extracting: $FILE"
            extraction_failed=true
        fi
    done
    
    if $extraction_failed; then
        echo "
WARNING: Some files failed to extract. Please check the error messages above."
        echo "This may be due to incomplete downloads or corrupted files."
        echo "Try re-downloading the problematic files."
    else
        echo "
All files extracted successfully!"
        echo "NuScence dataset ready at: $(pwd)"
    fi
else
    echo "
No .tgz files found. Please download the dataset manually first."
    exit 1
fi
