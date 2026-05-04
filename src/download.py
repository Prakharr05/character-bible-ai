"""
Download videos from a CSV of URLs using yt-dlp.
Usage: python src/download.py
"""
import csv
import subprocess
from pathlib import Path

VIDEOS_CSV = "videos.csv"
OUTPUT_DIR = Path("data/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_video(url: str, video_id: str) -> Path | None:
    output_path = OUTPUT_DIR / f"{video_id}.%(ext)s"
    cmd = [
        "yt-dlp",
        "-f", "best[height<=720]/best",   # cap at 720p for speed
        "-o", str(output_path),
        "--no-playlist",
        "--cookies-from-browser", "edge",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # find the actual downloaded file
        for f in OUTPUT_DIR.glob(f"{video_id}.*"):
            return f
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {url}: {e.stderr[:200]}")
        return None


def main():
    with open(VIDEOS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            url = row["url"]
            if any(OUTPUT_DIR.glob(f"{video_id}.*")):
                print(f"Skipping {video_id} (already downloaded)")
                continue
            print(f"Downloading {video_id}: {row['title']}")
            path = download_video(url, video_id)
            if path:
                print(f"  -> {path}")


if __name__ == "__main__":
    main()
