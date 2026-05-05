"""
Character Bible Extraction Pipeline.

For each character, we:
  1. Aggregate all their tagged transcripts.
  2. Run a structured extraction prompt to pull catchphrases, vocab,
     opinions, relationships, backstory, brands, aesthetic.
  3. Store the result in Postgres + push embeddings of every line to ChromaDB.

This is the *brain* of the system.
"""
import os
import json
from pathlib import Path
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

from db import SessionLocal, CharacterBible, Video, init_db

load_dotenv()
client = OpenAI()

TRANSCRIPTS_DIR = Path("data/transcripts")
BIBLES_DIR = Path("data/bibles")
BIBLES_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


# --------------------------------------------------------------------------
# The extraction prompt. This is the single most important piece of code.
# --------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are a script analyst building a character bible
for an Indian comedy creator's recurring character.

The character is: {character_name}

Below is every piece of dialogue this character has spoken across multiple
videos, with video titles for context. Your job is to extract a structured
character bible by reading like a careful screenwriter.

Be SPECIFIC and EVIDENCE-BASED. Do not invent traits. Only include things
explicitly supported by the dialogue. If something appears only once, mark
it as "single-source" in your output.

Return ONLY valid JSON in exactly this schema:

{{
  "catchphrases": [
    {{"phrase": "haaye", "frequency": "high|medium|single-source", "example_video": "video title"}}
  ],
  "vocabulary_fingerprint": {{
    "signature_words": ["words this character uses that others wouldn't"],
    "speech_patterns": ["e.g., mixes Hindi-English in a specific way"],
    "words_they_never_use": ["inferred from absence - e.g., never uses 'literally'"]
  }},
  "opinions": [
    {{"topic": "marriage", "stance": "what they think", "evidence": "quote or paraphrase"}}
  ],
  "relationships": [
    {{"name": "fictional son Rohan", "relation": "son", "established_in": "video title"}}
  ],
  "backstory_facts": [
    {{"fact": "lives in South Delhi", "evidence": "quote or paraphrase"}}
  ],
  "brands_referenced": [
    {{"brand": "Estee Lauder", "context": "praised|criticised|mentioned", "video": "title"}}
  ],
  "aesthetic_markers": {{
    "social_class": "e.g., upper-class Delhi",
    "tone": "e.g., haughty, judgmental, gossipy",
    "emotional_default": "e.g., outraged at the lower classes"
  }},
  "summary": "2-3 sentences capturing who this character fundamentally is"
}}

Here is the dialogue corpus:

{dialogue_corpus}
"""


def build_dialogue_corpus(character: str) -> str:
    """Aggregate all transcripts tagged to this character."""
    chunks = []
    for transcript_file in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        data = json.loads(transcript_file.read_text(encoding="utf-8"))
        if data.get("character") != character:
            continue
        chunks.append(f"=== Video: {data['title']} ===\n{data['text']}\n")
    return "\n".join(chunks)


def extract_bible(character: str) -> dict:
    """Run the LLM extraction for one character."""
    print(f"\nBuilding bible for {character}...")
    corpus = build_dialogue_corpus(character)
    if not corpus:
        print(f"  No transcripts found for {character}")
        return {}

    print(f"  Corpus length: {len(corpus)} chars")

    prompt = EXTRACTION_PROMPT.format(
        character_name=character,
        dialogue_corpus=corpus,
    )

    response = client.chat.completions.create(
        model="gpt-4o",          # use 4o for quality on extraction
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,         # low temp = deterministic extraction
        response_format={"type": "json_object"},
    )

    bible = json.loads(response.choices[0].message.content)

    # save to disk
    bible_path = BIBLES_DIR / f"{character}.json"
    bible_path.write_text(json.dumps(bible, indent=2, ensure_ascii=False))
    print(f"  -> Saved to {bible_path}")

    return bible


def store_in_db(character: str, bible: dict):
    """Persist the bible to Postgres."""
    session = SessionLocal()
    existing = session.query(CharacterBible).filter_by(character=character).first()

    fields = dict(
        catchphrases=json.dumps(bible.get("catchphrases", [])),
        vocabulary=json.dumps(bible.get("vocabulary_fingerprint", {})),
        opinions=json.dumps(bible.get("opinions", [])),
        relationships=json.dumps(bible.get("relationships", [])),
        backstory=json.dumps(bible.get("backstory_facts", [])),
        brands_referenced=json.dumps(bible.get("brands_referenced", [])),
        aesthetic=json.dumps(bible.get("aesthetic_markers", {})),
    )

    if existing:
        for k, v in fields.items():
            setattr(existing, k, v)
    else:
        session.add(CharacterBible(character=character, **fields))
    session.commit()
    session.close()


def embed_dialogue(character: str):
    """Push every dialogue segment to ChromaDB for semantic search."""
    collection = chroma_client.get_or_create_collection(name=f"dialogue_{character}")

    docs, ids, metas = [], [], []
    for transcript_file in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        data = json.loads(transcript_file.read_text(encoding="utf-8"))
        if data.get("character") != character:
            continue
        for i, seg in enumerate(data.get("segments", [])):
            text = seg["text"].strip()
            if len(text) < 10:
                continue
            doc_id = f"{data['video_id']}_{i}"
            docs.append(text)
            ids.append(doc_id)
            metas.append({
                "video_id": data["video_id"],
                "title": data["title"],
                "start": seg["start"],
                "end": seg["end"],
            })

    if docs:
        # upsert handles re-runs
        collection.upsert(documents=docs, ids=ids, metadatas=metas)
        print(f"  Embedded {len(docs)} segments for {character}")


def main():
    init_db()
    characters = ["billi_maasi", "west_delhi_girl"]   # extend as you tag more
    for character in characters:
        bible = extract_bible(character)
        if bible:
            store_in_db(character, bible)
            embed_dialogue(character)


if __name__ == "__main__":
    main()
