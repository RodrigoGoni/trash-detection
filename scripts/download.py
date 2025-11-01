import kagglehub
import os

# Define the desired relative path
target_path = "../data/raw"

os.makedirs(target_path, exist_ok=True)

# Download the latest version to the specified path
path = kagglehub.dataset_download(
    "kneroma/tacotrashdataset"
)

print(f"Path to dataset files: {path}")
