#!/usr/bin/env python3
"""
Fast multi-threaded downloader for Hugging Face weights using HTTP Range requests.
"""
import concurrent.futures
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "weights"

HF_URL = "https://huggingface.co/why0722/hcd-circuit-weights/resolve/main/weights_yolo_parseq.zip"

def get_content_info(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        final_url = resp.geturl()
        total_size = int(resp.headers.get("content-length", 0))
    return final_url, total_size

def download_range(url, start, end, out_path, chunk_idx):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start}-{end}"
    })
    attempts = 0
    while attempts < 5:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                expected = end - start + 1
                if len(data) != expected:
                    raise IOError(f"Chunk {chunk_idx}: Expected {expected} bytes, got {len(data)}")
                with open(out_path, "wb") as f:
                    f.write(data)
                return True
        except Exception as e:
            attempts += 1
            time.sleep(1.5 * attempts)
    raise RuntimeError(f"Failed to download chunk {chunk_idx} ({start}-{end}) after 5 attempts")

def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    zip_dest = WEIGHTS_DIR / "weights_yolo_parseq.zip"
    
    print("[1/3] Resolving URL and file size...")
    final_url, total_size = get_content_info(HF_URL)
    print(f"Total size: {total_size} bytes ({total_size / (1024*1024):.1f} MB)")
    
    num_threads = 16
    chunk_size = (total_size + num_threads - 1) // num_threads
    chunks = []
    for i in range(num_threads):
        start = i * chunk_size
        end = min(total_size - 1, (i + 1) * chunk_size - 1)
        if start <= end:
            chunks.append((i, start, end, WEIGHTS_DIR / f"part_{i}.tmp"))

    print(f"[2/3] Downloading {len(chunks)} chunks in parallel with {num_threads} threads...")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(download_range, final_url, c[1], c[2], c[3], c[0]): c[0]
            for c in chunks
        }
        done = 0
        for f in concurrent.futures.as_completed(futures):
            idx = futures[f]
            f.result()
            done += 1
            elapsed = time.time() - t0
            pct = done * 100 // len(chunks)
            print(f"  Chunk {idx+1}/{len(chunks)} finished ({pct}% completed, {elapsed:.1f}s elapsed)")

    print("[3/3] Assembling chunks into final zip...")
    with open(zip_dest, "wb") as out_f:
        for c in chunks:
            part_path = c[3]
            with open(part_path, "rb") as in_f:
                out_f.write(in_f.read())
            part_path.unlink()

    print(f"Saved {zip_dest.name} ({zip_dest.stat().st_size} bytes).")
    print("Extracting weights...")
    with zipfile.ZipFile(zip_dest, "r") as z:
        has_weights_root = any(name.startswith("weights/") for name in z.namelist())
        extract_to = REPO_ROOT if has_weights_root else WEIGHTS_DIR
        z.extractall(extract_to)

    print("Unpacking complete!")
    try:
        zip_dest.unlink()
    except Exception:
        pass
    print("DONE!")

if __name__ == "__main__":
    main()
