# Character Bible AI

A writing co-pilot for **Kusha Kapila**'s character universe, using OpenAI, ChromaDB, and N8N.

## The Problem

Kusha Kapila isn't a typical influencer. She runs what's essentially a fictional universe of nine-plus recurring characters — Billi Maasi, Coco Bhaiya, Naina, Sunita, Zaalim Aunty, and others — each with their own voice, vocabulary, and worldview built over eight years of content. As she scales into Bollywood, Netflix shows, and her own shapewear brand UnderNeat, that entire character bible only exists in her head.

This causes three concrete failure modes:

1. **Continuity drift** — catchphrases and backstory facts quietly contradicting past videos.
2. **Brand collisions** — new partnerships clashing with brands a character previously endorsed.
3. **Creative scaling bottleneck** — she can't write longer-form content without rewatching her own videos.

## The Solution

A four-layer system that turns Kusha's content archive into a living, queryable character bible.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Ingestion (N8N)                                       │
│  Daily scrape of Instagram + YouTube → yt-dlp download          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: Character Attribution                                 │
│  Vision (GPT-4o) + Voice embeddings (pyannote) + Language (LLM) │
│  → Fused confidence score → Auto-tag or flag for review         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: Bible Extraction                                      │
│  Whisper transcription → GPT-4o structured extraction           │
│  → Postgres (canon facts) + ChromaDB (semantic dialogue search) │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Writing Co-Pilot (Streamlit)                          │
│  • Authenticity Scorer — would this character say this?         │
│  • Continuity Checker — flag contradictions with canon          │
│  • Free-form Q&A — ask anything about a character               │
└─────────────────────────────────────────────────────────────────┘
```

## What's Built in This Demo

This repo is a **vertical slice** focused on showing the system working end-to-end for two characters: **Billi Maasi** and **Coco Bhaiya**.

| Layer | Demo | Production |
|---|---|---|
| Ingestion | Manual CSV of 10-12 videos | N8N workflow (`n8n_workflow.json`) |
| Character attribution | Manual tagging in CSV | LLM + vision + voice fusion |
| Bible extraction | ✅ Live, working | Same |
| Writing co-pilot UI | ✅ Live, working | Same |

The auto-attribution layer is documented as the production vision but skipped for the 12-hour build — manual tagging gets us to a demo-able product faster without losing the AI showcase.

## Stack

- **OpenAI GPT-4o** — bible extraction, authenticity scoring, continuity checking, Q&A
- **OpenAI Whisper (local)** — transcription with auto language detection (handles Hinglish)
- **ChromaDB** — semantic search over every dialogue line
- **PostgreSQL + SQLAlchemy** — structured canon storage
- **Streamlit** — co-pilot UI
- **N8N** — production ingestion orchestration
- **yt-dlp** — video download

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY

# 3. Initialise the database
python src/db.py

# 4. Fill videos.csv with URLs and character tags

# 5. Run the pipeline
python src/download.py        # downloads videos
python src/transcribe.py      # whisper transcription
python src/extract_bible.py   # builds character bibles

# 6. Launch the UI
streamlit run src/app.py
```

## Demo Walkthrough

### Authenticity Scorer
Type `"haaye Coco bhaiya, lets go to Goa"` and the AI responds with a score and reasoning grounded in past dialogue — e.g., flagging that Billi Maasi never says "lets go to Goa" because her established persona only travels by private jet.

### Continuity Checker
Paste a draft script. The AI cross-references it against the canonical bible and flags every contradiction with a severity rating, citing the specific past video that establishes the canon.

### Q&A
Ask `"What does Coco Bhaiya think about cricket?"` — the AI answers using semantic search over every line that character has ever spoken, citing source videos.

## Why This Is the Right Solution for Kusha (Specifically)

This isn't a generic creator tool. It's tailored to her exact situation:

- **Multi-character creators are rare.** Most influencers maintain one persona. Kusha maintains nine. This system has more value for her than any other Indian creator.
- **Career inflection point.** She's moving from short-form sketch comedy into film, TV, and her own brand. The cost of forgetting her own canon is now measured in lost brand deals and broken IP, not just fan complaints.
- **Same problem TV writers' rooms have already solved with bibles.** This system gives her, as a one-woman studio, the same infrastructure that backed long-running TV franchises.

## Beyond Kusha

The architecture generalises to any character-driven Indian creator: Bhuvan Bam (BB Ki Vines), Zakir Khan's stage characters, regional sketch creators. Solve it for Kusha — the most complex universe in this space — and you have a template for an emerging class of Indian creator IP.

---

Built in 12 hours as a problem-statement-to-prototype assignment.
