from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
import os
import re

def download_data(file_list, repo_id, repo_type, local_dir):
    for file in file_list:
        hf_hub_download(
            repo_id=repo_id, 
            filename=file, 
            repo_type=repo_type, 
            local_dir=local_dir
        )

# This approach uses snapshot_download which supports include/exclude patterns
def download_with_patterns(repo_id, repo_type, local_dir, include_patterns=None, exclude_patterns=None):
    """
    Download files from a Hugging Face repository that match the include patterns.
    
    Args:
        repo_id: ID of the repository
        repo_type: Type of the repository (e.g., "dataset", "model")
        local_dir: Local directory to save files to
        include_patterns: List of glob patterns to include
        exclude_patterns: List of glob patterns to exclude
    """
    print(f"Downloading files from {repo_id} with:")
    if include_patterns:
        print(f"  Include patterns: {include_patterns}")
    if exclude_patterns:
        print(f"  Exclude patterns: {exclude_patterns}")
        
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            allow_patterns=include_patterns,
            ignore_patterns=exclude_patterns
        )
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error downloading files: {e}")
        print("Falling back to manual method...")
        
        # If snapshot_download fails, use our custom implementation
        download_data_with_pattern(
            repo_id=repo_id, 
            repo_type=repo_type, 
            local_dir=local_dir,
            include_pattern="|".join(include_patterns) if include_patterns else None
        )

# Our existing pattern-matching function remains as a fallback
def download_data_with_pattern(repo_id, repo_type, local_dir, include_pattern=None):
    """
    Download files from a Hugging Face repository that match the include pattern.
    
    Args:
        repo_id: ID of the repository
        repo_type: Type of the repository (e.g., "dataset", "model")
        local_dir: Local directory to save files to
        include_pattern: Regex pattern to filter files
    """
    # First get all files in the repo
    all_files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
    
    # Filter files based on pattern
    if include_pattern:
        pattern = re.compile(include_pattern)
        matching_files = [file for file in all_files if pattern.search(file)]
        print(f"Found {len(matching_files)} files matching pattern '{include_pattern}':")
        for file in matching_files:
            print(f"- {file}")
    else:
        matching_files = all_files
        print(f"No pattern specified. Found {len(all_files)} files in repository.")
    
    # Ask for confirmation before downloading
    if matching_files:
        confirmation = input(f"\nDownload these {len(matching_files)} files? (y/n): ")
        if confirmation.lower() == 'y':
            for file in matching_files:
                print(f"Downloading: {file}")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    repo_type=repo_type,
                    local_dir=local_dir
                )
            print("Download completed.")
        else:
            print("Download cancelled.")

# Example usage
if __name__ == "__main__":
    # Example using the newer snapshot_download with pattern matching
    download_with_patterns(
        repo_id="OpenDILabCommunity/LMDrive",
        repo_type="dataset",
        local_dir="/mnt/data2/",
        include_patterns=["*Town01*", "*Town06*", "*Town10*"]
    )
    
    # Or alternatively, using the CLI-style approach
    # This requires a newer version of huggingface_hub
    # import subprocess
    # cmd = [
    #     "huggingface-cli", "download",
    #     "OpenDILabCommunity/LMDrive",
    #     "--repo-type", "dataset",
    #     "--include", "*Town01*", "--include", "*Town06*", "--include", "*Town10*",
    #     "--local-dir", "FedDrive/src_data/"
    # ]
    # subprocess.run(cmd)