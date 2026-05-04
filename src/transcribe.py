"""
Transcribe downloaded videos using OpenAI's hosted Whisper API.
Faster, cleaner, no local PyTorch headache. ~$0.006/minute of audio.
"""
import csv
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

VIDEOS_CSV = "videos.csv"
VIDEOS_DIR = Path("data/videos")
TRANSCRIPTS_DIR = Path("data/transcripts")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI Whisper API has a 25MB file limit. We trust 720p videos under
# ~5 min stay well under that. For longer videos, you'd extract audio
# with ffmpeg first.
MAX_BYTES = 25 * 1024 * 1024


def transcribe_file(video_path: Path) -> dict:
    """Send file to OpenAI Whisper API."""
    if video_path.stat().st_size > MAX_BYTES:
        print(f"  WARNING: {video_path.name} exceeds 25MB; may fail.")

    with open(video_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",   # gives us segments + timestamps
        )
    # result is a pydantic object; convert to dict
    return result.model_dump()


def transcribe_all():
    with open(VIDEOS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

            if transcript_path.exists():
                print(f"Skipping {video_id} (already transcribed)")
                continue

            video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
            if not video_files:
                print(f"No video file for {video_id}")
                continue

            print(f"Transcribing {video_id}: {row['title']}...")
            try:
                result = transcribe_file(video_files[0])
            except Exception as e:
                print(f"  Failed: {e}")
                continue

            output = {
                "video_id": video_id,
                "url": row["url"],
                "title": row["title"],
                "character": row["character"],
                "language": result.get("language"),
                "text": result.get("text", ""),
                "segments": [
                    {"start": s["start"], "end": s["end"], "text": s["text"]}
                    for s in result.get("segments", [])
                ],
            }
            transcript_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  -> {transcript_path}")


if __name__ == "__main__":
    transcribe_all()