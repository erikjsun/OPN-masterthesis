"""
Quick test script to verify smart caching implementation works.
This will attempt to load a small subset of data twice and show the speed difference.
"""

import os
import json
import time
import shutil
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from main import FullyExtractedBlobDataset

# Load environment variables
load_dotenv()

AZURE_STORAGE_URL = os.getenv('AZURE_STORAGE_URL')
AZURE_STORAGE_KEY = os.getenv('AZURE_STORAGE_KEY')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME')

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)['main']
    PREPROCESSEDDATA_FOLDERNAME = config['paths']['preprocessed_folder']

# Clean cache before test
if os.path.exists('local_cache'):
    print("Cleaning old cache...")
    shutil.rmtree('local_cache')
print()

# Initialize Azure Blob Service Client
blob_service_client = BlobServiceClient(account_url=AZURE_STORAGE_URL, credential=AZURE_STORAGE_KEY)
container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

# List preprocessed .pth files
print("Listing available .pth files...")
all_pth_blobs = [
    blob.name for blob in container_client.list_blobs(name_starts_with=PREPROCESSEDDATA_FOLDERNAME)
    if blob.name.endswith('.pth')
]
print(f"Found {len(all_pth_blobs)} .pth files")

# Test with just 3 files (will cache up to ~2.2GB worth)
test_files = all_pth_blobs[:3]
print(f"\nTesting with {len(test_files)} files:")
for f in test_files:
    print(f"  - {f}")

# First load (should download and cache)
print("\n" + "="*60)
print("FIRST LOAD (should download from Azure and cache)")
print("="*60)
start = time.time()
dataset1 = FullyExtractedBlobDataset(blob_service_client, AZURE_CONTAINER_NAME, test_files)
end = time.time()
first_load_time = end - start
print(f"\nFirst load completed in {first_load_time:.2f} seconds")
print(f"Loaded {len(dataset1)} samples")

# Delete the dataset to free memory
del dataset1

# Second load (should use cache)
print("\n" + "="*60)
print("SECOND LOAD (should use cache)")
print("="*60)
start = time.time()
dataset2 = FullyExtractedBlobDataset(blob_service_client, AZURE_CONTAINER_NAME, test_files)
end = time.time()
second_load_time = end - start
print(f"\nSecond load completed in {second_load_time:.2f} seconds")
print(f"Loaded {len(dataset2)} samples")

# Show improvement
print("\n" + "="*60)
print("RESULTS")
print("="*60)
speedup = first_load_time / second_load_time if second_load_time > 0 else float('inf')
print(f"First load:  {first_load_time:.2f}s")
print(f"Second load: {second_load_time:.2f}s")
print(f"Speedup:     {speedup:.1f}x faster")
print(f"Time saved:  {first_load_time - second_load_time:.2f}s")

# Check cache size
if os.path.exists('local_cache'):
    cache_size_bytes = sum(
        os.path.getsize(os.path.join('local_cache', f))
        for f in os.listdir('local_cache')
        if os.path.isfile(os.path.join('local_cache', f))
    )
    cache_size_gb = cache_size_bytes / (1024**3)
    print(f"Cache size:  {cache_size_gb:.2f} GB")
