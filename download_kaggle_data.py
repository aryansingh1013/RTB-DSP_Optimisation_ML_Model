import kagglehub
import os

print("Downloading dataset...")
try:
    path = kagglehub.dataset_download("lastsummer/ipinyou")
    print("Path to dataset files:", path)
    
    # List files in the directory
    print("\nFiles in download directory:")
    for root, dirs, files in os.walk(path):
        for file in files:
            print(os.path.join(root, file))
except Exception as e:
    print(f"Error downloading dataset: {e}")
