import os
import requests
from tqdm import tqdm
import tarfile

DATA_DIR = "dakshina_dataset_v1.0"
TAR_FILE = "daksh.tar"
DOWNLOAD_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"

def download_file(url, filename):
    """Download with progress bar"""
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
        desc=f"📥 Downloading {filename}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        colour='green'
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

def extract_tar(tar_path, extract_path="."):
    """Extract with progress feedback"""
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="📦 Extracting", unit="file", colour="cyan"):
            tar.extract(member, path=extract_path)

# Main logic
if not os.path.isdir(DATA_DIR):
    if not os.path.isfile(TAR_FILE):
        download_file(DOWNLOAD_URL, TAR_FILE)
    else:
        print(f"✅ {TAR_FILE} already exists. Skipping download.")
    
    print("📂 Extracting tarball...")
    extract_tar(TAR_FILE)
    print("✅ Extraction complete.")
else:
    print(f"✅ {DATA_DIR} already exists. Skipping download and extraction.")
