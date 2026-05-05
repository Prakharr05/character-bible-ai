"""
Character Bible AI - Writing Co-Pilot
Streamlit interface with three core features.
"""
import os
import json
import streamlit as st
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

from db import SessionLocal, CharacterBible

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_bible(character: str) -> dict:
    session = SessionLocal()
    row = session.query(CharacterBible).filter_by(character=character).first()
    session.close()
    if not row:
        return {}
    return {
        "catchphrases": json.loads(row.catchphrases or "[]"),
        "vocabulary": json.loads(row.vocabulary or "{}"),
        "opinions": json.loads(row.opinions or "[]"),
        "relationships": json.loads(row.relationships or "[]"),
        "backstory": json.loads(row.backstory or "[]"),
        "brands": json.loads(row.brands_referenced or "[]"),
        "aesthetic": json.loads(row.aesthetic or "{}"),
    }


def get_relevant_dialogue(character: str, query: str, n: int = 6) -> list[dict]:
    """Semantic search across the character's past dialogue."""
    try:
        collection = chroma_client.get_collection(name=f"dialogue_{character}")
        results = collection.query(query_texts=[query], n_results=n)
        return [
            {"text": doc, "meta": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
    except Exception:
        return []


# --------------------------------------------------------------------------
# Feature 1: Authenticity Scorer
# --------------------------------------------------------------------------
AUTHENTICITY_PROMPT = """You are a script consultant for an Indian comedy
creator. Evaluate whether the draft line below is authentic to the
character's established voice.

CHARACTER BIBLE:
{bible}

RELEVANT PAST DIALOGUE (semantically similar to the draft):
{relevant}

DRAFT LINE:
"{draft}"

Score the line on authenticity from 1-10. Then explain in 2-3 sentences
exactly why, citing specific evidence from the bible or past dialogue
(quote past lines where useful). If under 7, suggest a rewrite that
matches the character's voice better.

Return JSON in this exact format:
{{
  "score": 7,
  "reasoning": "...",
  "evidence": ["specific quote or fact"],
  "suggested_rewrite": "..." or null
}}
"""


def score_authenticity(character: str, draft: str) -> dict:
    bible = load_bible(character)
    relevant = get_relevant_dialogue(character, draft, n=6)
    relevant_str = "\n".join(f"- \"{r['text']}\" (from {r['meta']['title']})" for r in relevant)

    prompt = AUTHENTICITY_PROMPT.format(
        bible=json.dumps(bible, indent=2, ensure_ascii=False),
        relevant=relevant_str or "(none found)",
        draft=draft,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# --------------------------------------------------------------------------
# Feature 2: Continuity Checker
# --------------------------------------------------------------------------
CONTINUITY_PROMPT = """You are a continuity editor for an Indian comedy
creator's character universe. Your job is to find contradictions between
a new draft script and the character's established canon.

CHARACTER BIBLE (canon):
{bible}

NEW DRAFT SCRIPT:
{draft}

Find every contradiction. For each, explain:
  - What the draft says
  - What canon says
  - Severity (low/medium/high)

Also flag risky points where the draft introduces something brand-new
that wasn't in canon (could be fine, but creator should know).

Return JSON:
{{
  "contradictions": [
    {{
      "draft_says": "...",
      "canon_says": "...",
      "evidence": "specific bible item or video",
      "severity": "high"
    }}
  ],
  "new_canon_introduced": [
    {{"element": "...", "note": "..."}}
  ],
  "verdict": "1-2 sentences overall"
}}
"""


def check_continuity(character: str, draft: str) -> dict:
    bible = load_bible(character)
    prompt = CONTINUITY_PROMPT.format(
        bible=json.dumps(bible, indent=2, ensure_ascii=False),
        draft=draft,
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# --------------------------------------------------------------------------
# Feature 3: Free-form Q&A
# --------------------------------------------------------------------------
QA_PROMPT = """You are answering questions about an Indian comedy creator's
character. Answer ONLY based on the bible and past dialogue below. If the
answer isn't established in canon, say so clearly.

CHARACTER BIBLE:
{bible}

RELEVANT PAST DIALOGUE:
{relevant}

QUESTION: {question}

Answer in 2-4 sentences. Cite specific videos or bible items where useful.
"""


def answer_question(character: str, question: str) -> str:
    bible = load_bible(character)
    relevant = get_relevant_dialogue(character, question, n=8)
    relevant_str = "\n".join(f"- \"{r['text']}\" (from {r['meta']['title']})" for r in relevant)

    prompt = QA_PROMPT.format(
        bible=json.dumps(bible, indent=2, ensure_ascii=False),
        relevant=relevant_str or "(no semantically similar dialogue found)",
        question=question,
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Character Bible AI", page_icon="ðŸ“š", layout="wide")
st.title("ðŸ“š Character Bible AI")
st.caption("A writing co-pilot for Kusha Kapila's character universe")

with st.sidebar:
    st.header("Character")
    character = st.selectbox(
        "Select character",
        ["billi_maasi", "west_delhi_girl"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    bible = load_bible(character)
    if bible:
        st.success(f"Bible loaded: {len(bible.get('catchphrases', []))} catchphrases, "
                   f"{len(bible.get('opinions', []))} opinions, "
                   f"{len(bible.get('brands', []))} brands tracked")
        with st.expander("View bible"):
            st.json(bible)
    else:
        st.warning("No bible found. Run `python src/extract_bible.py` first.")

tab1, tab2, tab3 = st.tabs(["âœï¸ Authenticity Scorer", "ðŸ” Continuity Checker", "ðŸ’¬ Ask About Character"])

# Tab 1: Authenticity Scorer
with tab1:
    st.subheader("Would this character actually say this?")
    st.markdown("Type a draft line. The AI scores it against the character's established voice.")
    draft = st.text_area("Draft line", height=100, key="auth_draft",
                         placeholder="haaye Coco bhaiya, lets go to Goa")
    if st.button("Score authenticity", type="primary", key="auth_btn"):
        if draft.strip():
            with st.spinner("Analysing..."):
                result = score_authenticity(character, draft)
            score = result.get("score", 0)
            color = "green" if score >= 7 else "orange" if score >= 5 else "red"
            st.markdown(f"### Authenticity: :{color}[{score}/10]")
            st.markdown(f"**Why:** {result.get('reasoning', '')}")
            if result.get("evidence"):
                st.markdown("**Evidence:**")
                for e in result["evidence"]:
                    st.markdown(f"- {e}")
            if result.get("suggested_rewrite"):
                st.markdown(f"**Suggested rewrite:** _{result['suggested_rewrite']}_")

# Tab 2: Continuity Checker
with tab2:
    st.subheader("Check a script against established canon")
    st.markdown("Paste a draft script. The AI flags contradictions with past videos.")
    script = st.text_area("Draft script", height=200, key="cont_script",
                          placeholder="Paste a multi-line script here...")
    if st.button("Check continuity", type="primary", key="cont_btn"):
        if script.strip():
            with st.spinner("Cross-referencing canon..."):
                result = check_continuity(character, script)
            contradictions = result.get("contradictions", [])
            if contradictions:
                st.error(f"Found {len(contradictions)} contradiction(s)")
                for c in contradictions:
                    sev = c.get("severity", "medium")
                    sev_emoji = {"high": "ðŸ”´", "medium": "ðŸŸ¡", "low": "ðŸŸ¢"}.get(sev, "âšª")
                    st.markdown(f"{sev_emoji} **{sev.upper()}**")
                    st.markdown(f"- Draft says: _{c.get('draft_says', '')}_")
                    st.markdown(f"- Canon says: _{c.get('canon_says', '')}_")
                    st.markdown(f"- Evidence: {c.get('evidence', '')}")
                    st.markdown("---")
            else:
                st.success("No contradictions found âœ“")

            new_items = result.get("new_canon_introduced", [])
            if new_items:
                st.info("New canon being introduced:")
                for n in new_items:
                    st.markdown(f"- **{n.get('element', '')}** â€” {n.get('note', '')}")

            st.markdown(f"**Verdict:** {result.get('verdict', '')}")

# Tab 3: Q&A
with tab3:
    st.subheader("Ask anything about this character's canon")
    question = st.text_input("Question", key="qa_question",
                             placeholder="What does Coco Bhaiya think about cricket?")
    if st.button("Ask", type="primary", key="qa_btn"):
        if question.strip():
            with st.spinner("Searching canon..."):
                answer = answer_question(character, question)
            st.markdown(answer)

st.markdown("---")
st.caption("Powered by GPT-4o + ChromaDB | N8N orchestration in production")
