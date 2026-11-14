# NewsCheck (Automated News Corroboration System)

> **NewsCheck** is a fully automated, conservative news integrity pipeline that ingests reports from diverse independent sources, detects syndication/wire duplication, clusters articles by event, cross-checks claims via Natural Language Inference (NLI), applies mathematical corroboration scoring, and publishes a static site — with **zero human intervention**.

The system strictly adheres to the principle: **Never claim a story is "true" or "verified"**. Instead, it reports:
`"Corroborated by N independent sources"` alongside transparent claim agreement and dispute tables.

---

## 1. Architecture & Pipeline

```
  1. Ingest (RSS / feedparser, robots.txt, rate limits, domain whitelist)
       ↓
  2. Extract (trafilatura body extraction, content hashing, paywall fallback)
       ↓
  3. Deduplicate (MinHash LSH + wire-service syndication detection: PTI, ANI, Reuters, AP, AFP)
       ↓
  4. Cluster & Rank (Multilingual SBERT + HDBSCAN, tier-weighted scoring, top 10 selection)
       ↓
  5. Claim Extraction (Verbatim source-sentence anchoring via Ollama / Gemini / Rule-based)
       ↓
  6. Pairwise Cross-Check (NLI model: AGREED, DISPUTED, SINGLE_SOURCE)
       ↓
  7. Corroboration Scoring (Tier weights, primary source bonus, conflict penalty, recency decay)
       ↓
  8. Safety Gates & Sensitive Classifier (Suppresses narrative for sensitive topics / <3 sources)
       ↓
  9. Constrained Generation & Entailment Filter (Sentences MUST be entailed by source evidence)
       ↓
  10. Atomic Build & Rollback (Jinja2 static site swap, failure alerts via Telegram / Email)
```

---

## 2. Integrity & Safety Gates

1. **Wire Deduplication**: Ten outlets publishing the exact same wire story count as **one** confirmation, not ten.
2. **Minimum Corroboration**: Stories with fewer than 3 independent sources are labeled `Unconfirmed` and do not receive generated narrative summaries.
3. **Developing Badge**: Stories under 2 hours old are flagged `Developing` and re-evaluated each run.
4. **Sensitive Topic Gate**: Stories involving crimes naming private individuals, medical/health claims, communal/religious violence, suicide, minors, sexual violence, or active conflicts **never** publish generated narrative — only the verbatim headline, claim comparison table, and direct source links.
5. **Defamation & Attribution**: No assertions against private individuals without attribution. All quotes are strictly capped at under 15 words and linked to the source.
6. **Entailment Filter**: Every generated sentence is checked against stored source passages via NLI. If fewer than 2 sentences survive entailment, the story falls back to headline + claim table only.
7. **Atomic Rollback**: If any stage errors out, the pipeline preserves the last-good published build and fires an alert.

---

## 3. Quickstart & Local Setup

### Prerequisites
- Python 3.12 or 3.13
- Free and open-source dependencies (no paid services required)

```bash
# Clone the repository
git clone https://github.com/newscheck/newscheck.git
cd newscheck

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install package in editable mode with development tools
pip install -e ".[dev]"

# Copy environment template (all keys are optional)
cp .env.example .env
```

### CLI Commands

```bash
# Verify configured feeds (30 curated India English sources by default)
newscheck show-sources

# Initialize the SQLite database schema
newscheck init-db

# Run the complete end-to-end pipeline
newscheck run

# Build the static site manually from database
newscheck build
```

---

## 4. Testing & Verification

```bash
# Run the complete test suite (35+ unit & integration tests)
pytest -v

# Run linting and code formatting
ruff check src tests

# Run strict type checking
mypy src
```

---

## 5. Automated Production Deployment

The project includes a ready-to-run GitHub Actions workflow (`.github/workflows/pipeline.yml`):
- Runs every **60 minutes** on a cron schedule.
- Uses **concurrency locks** to guarantee single-writer SQLite safety.
- Builds the static site atomically into `public/`.
- Deploys automatically to **GitHub Pages**.
- Commits updated run metadata back to the repository.

---

## 6. How to Enhance & Scale

- **Local LLMs (Ollama)**: Install [Ollama](https://ollama.com) and pull `llama3.1:8b`. Set `llm_provider = "ollama"` in `settings.toml`. The pipeline automatically prompts Ollama for structured claim extraction and validates source sentences.
- **GPU Acceleration**: When running on a machine with CUDA/ROCm, PyTorch automatically moves the embedding and NLI models to the GPU for faster clustering and cross-checking.
- **Multilingual Support**: `paraphrase-multilingual-MiniLM-L12-v2` already clusters Hindi, Kannada, and English. Multilingual claim extraction can be enabled by adding translated prompts to the LLM extractor.
- **Paid / Specialized Feeds**: Add commercial wire feeds (e.g. Bloomberg, Factiva) directly to `sources.yaml` with `tier: 1`.
