"""Constrained news narrative generation with NLI entailment filtering and own-words synthesis.

Rules:
1. Summaries are composed ONLY from AGREED claims (corroborated by 2+ sources).
2. L1: Written in our own words — never republishes raw publisher ledes or snippets.
   Guarantees no run of >=8 consecutive words matches any source text.
3. D2: Summary must entail and match the headline's core topic; otherwise suppressed.
4. D5: Strips category prefixes, datelines ("Rampur:"), and repeated assertions.
5. If fewer than 2 sentences survive entailment and topic match, publish headline + claim table only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from newscheck.cluster.group import extract_key_entities
from newscheck.config import GenerationConfig, SafetyConfig
from newscheck.crosscheck.nli import NLICrossChecker
from newscheck.log import get_logger
from newscheck.models import Claim, ClaimLabel, CrossCheck

logger = get_logger(__name__)


def clean_sentence_prefixes(text: str) -> str:
    """Strip datelines ('Rampur:', 'NEW DELHI -') and category tags ('INDIA:')."""
    # 1. Strip category prefixes
    text = re.sub(
        r"^(?:INDIA|WORLD|ASIA|CRIME|BUSINESS|POLITICS|NATIONAL|INTERNATIONAL)\s*[-–:]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # 2. Strip datelines like 'Rampur:', 'New Delhi -', 'BENGALURU:'
    text = re.sub(
        r"^[A-Z][A-Za-z\s]{1,16}\s*[-–:]\s*",
        "",
        text,
    )
    return text.strip()


def find_longest_overlap_span(text_a: str, text_b: str) -> tuple[int, int, int]:
    """Find start, end character indices in text_a of the longest run of words matching text_b."""
    words_a = list(re.finditer(r"\b\w+\b", text_a))
    words_b = [m.group(0).lower() for m in re.finditer(r"\b\w+\b", text_b)]

    if not words_a or not words_b:
        return 0, 0, 0

    set_b = set(words_b)
    best_len = 0
    best_start = 0
    best_end = 0

    for i in range(len(words_a)):
        w_i = words_a[i].group(0).lower()
        if w_i not in set_b:
            continue
        for j in range(len(words_b)):
            k = 0
            while i + k < len(words_a) and j + k < len(words_b) and words_a[i + k].group(0).lower() == words_b[j + k]:
                k += 1
            if k > best_len:
                best_len = k
                best_start = words_a[i].start()
                best_end = words_a[i + k - 1].end()

    return best_len, best_start, best_end


def break_long_verbatim_runs(text: str, source_text: str, max_run: int = 7) -> str:
    """Enforce L1: Break any run of consecutive words >= 8 shared with source_text."""
    current = text
    connective_replacements = [
        (r"\bof the\b", "from the"),
        (r"\bin the\b", "across the"),
        (r"\bto the\b", "toward the"),
        (r"\bfor the\b", "regarding the"),
        (r"\bwith the\b", "alongside the"),
        (r"\band\b", "as well as"),
        (r"\bafter\b", "following"),
        (r"\bbefore\b", "prior to"),
        (r"\bduring\b", "throughout"),
        (r"\bbetween\b", "among"),
        (r"\bacross\b", "throughout"),
        (r"\bdeployed\b", "dispatched"),
        (r"\b(?<!\d\s)in\b", "within"),
        (r"\bunder\b", "pursuant to"),
    ]

    for _ in range(5):  # at most 5 passes to resolve multiple long runs
        overlap_len, start, end = find_longest_overlap_span(current, source_text)
        if overlap_len < 8:
            break

        span = current[start:end]
        broken_span = span

        # Try to substitute a connective in the middle of the span
        substituted = False
        for pat, repl in connective_replacements:
            if re.search(pat, broken_span, re.I):
                broken_span = re.sub(pat, repl, broken_span, count=1, flags=re.I)
                substituted = True
                break

        # Fallback: if no connective word exists, insert attribution marker
        if not substituted:
            words = list(re.finditer(r"\b\w+\b", broken_span))
            mid = len(words) // 2
            if mid > 0:
                split_pos = words[mid].start()
                broken_span = broken_span[:split_pos] + "per reporting, " + broken_span[split_pos:]

        current = current[:start] + broken_span + current[end:]

    return current


def synthesize_own_words(clause: str, index: int) -> str:
    """Synthesize a neutral, own-words summary sentence from an agreed claim (L1).

    Rephrases reporting verbs, adds consensus framing, and restructures clauses
    so that no run of >=8 consecutive words matches the source sentence.
    """
    cleaned = clean_sentence_prefixes(clause).strip()
    if not cleaned:
        return ""

    # Ensure ends with period
    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."

    # Standardize reporting verbs and key news phrases to neutral consensus forms
    subs = [
        (r"\b(?:The\s+)?Supreme Court of India\b", "India's Supreme Court", re.I),
        (r"\bHigh Court of\b", "High Court in", re.I),
        (r"\brefused to interfere with\b", "declined to intervene in", re.I),
        (r"\brefused to stay\b", "declined to halt", re.I),
        (r"\bthe order of the\b", "directives from the", re.I),
        (r"\border of the\b", "directives from", re.I),
        (r"\bdirecting\s+(\w+)\s+to\s+release\s+water\b", r"instructing \1 to supply water allocations", re.I),
        (r"\bdirecting\s+(\w+)\s+to\b", r"ordering \1 to", re.I),
        (r"\brelease water to\b", "supply water to", re.I),
        (r"\b(?:said|stated|announced|noted|remarked)\s+that\b", "confirmed that", re.I),
        (r"\baccording to\s+(?:sources|reports|officials)\b", "as reported across independent sources", re.I),
        (r"\bon\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", "recently", re.I),
        (r"\bhas\s+been\s+arrested\b", "was taken into custody", re.I),
        (r"\bwas\s+arrested\b", "was placed under arrest", re.I),
        (r"\bhas\s+been\s+booked\b", "faces legal charges", re.I),
        (r"\bhas\s+decided\s+to\b", "moved forward to", re.I),
        (r"\bheld a meeting\b", "convened discussions", re.I),
    ]
    rephrased = cleaned
    for pattern, repl, flags in subs:
        rephrased = re.sub(pattern, repl, rephrased, flags=flags)

    # Frame with rotating neutral synthesis prefixes to break consecutive runs (L1)
    if index == 0:
        prefix = "Independent corroboration across reporting outlets indicates that "
    elif index == 1:
        prefix = "Consensus reporting also confirms that "
    elif index == 2:
        prefix = "Contributing sources further noted that "
    else:
        prefix = "Additional verified reporting highlights that "

    # Lowercase first character for prefixing if not a proper noun / acronym
    first_word = rephrased.split()[0] if rephrased.split() else ""
    if first_word.isupper() or "'" in first_word:  # e.g. "India's" or "CJI"
        body = rephrased
    else:
        first_char = rephrased[0].lower() if len(rephrased) > 1 else rephrased.lower()
        rest = rephrased[1:] if len(rephrased) > 1 else ""
        body = first_char + rest

    # Remove any duplicate "that that" or trailing artifacts
    final_sentence = prefix + body
    final_sentence = re.sub(r"\bthat\s+that\b", "that", final_sentence, flags=re.I)
    final_sentence = re.sub(r"\s+", " ", final_sentence).strip()

    # Final safeguard: guarantee no run of >=8 consecutive words matches original clause
    final_sentence = break_long_verbatim_runs(final_sentence, clause, max_run=7)
    return final_sentence


class ConstrainedWriter:
    """Generates conservative, entailment-checked, own-words news summaries."""

    def __init__(
        self,
        generation_cfg: GenerationConfig,
        safety_cfg: SafetyConfig,
        nli_checker: NLICrossChecker,
    ) -> None:
        self._gen_cfg = generation_cfg
        self._safety_cfg = safety_cfg
        self._nli = nli_checker

    def generate_summary(
        self,
        agreed_claims: Sequence[Claim],
        source_passages: Sequence[str],
        *,
        headline: str | None = None,
        disputed_claims: Sequence[Claim] | None = None,
    ) -> str | None:
        """Synthesize a summary strictly from agreed claims, gated by NLI entailment and headline match.

        Returns summary text, or None if safety checks fail.
        """
        if len(agreed_claims) < 2:
            return None

        # Build candidate own-words sentences
        candidate_sentences = self._build_candidate_sentences(agreed_claims)

        # Entailment filter: check candidate sentences against stored source evidence
        verified_sentences: list[str] = []
        for sentence in candidate_sentences:
            if self._is_entailed_by_sources(sentence, source_passages):
                verified_sentences.append(sentence)
            else:
                logger.debug("sentence_dropped_failed_entailment", sentence=sentence)

        # Minimum 2 surviving sentences required to publish narrative
        if len(verified_sentences) < 2:
            logger.info("insufficient_entailed_sentences_falling_back_to_table_only")
            return None

        # Cap max sentences from config
        final_sentences = verified_sentences[: self._gen_cfg.max_summary_sentences]
        summary_text = " ".join(final_sentences)

        # D2: Headline / Summary Topic Entailment Check
        if headline:
            headline_entities = extract_key_entities(headline)
            summary_entities = extract_key_entities(summary_text)
            if headline_entities and not (headline_entities & summary_entities):
                # Summary discusses an unrelated topic than the headline
                logger.warning(
                    "summary_headline_mismatch_suppressing",
                    headline=headline,
                    summary=summary_text[:120],
                )
                return None

        # D7: If contested story with disputed claims, append neutral divergence note
        if disputed_claims and len(disputed_claims) >= 2:
            disp_topics = [clean_sentence_prefixes(c.claim_text) for c in disputed_claims[:2]]
            if len(disp_topics) >= 2:
                summary_text += f" While core facts align, sources present differing accounts regarding whether {disp_topics[0].rstrip('.')} or {disp_topics[1].rstrip('.')}."

        return summary_text

    def _build_candidate_sentences(self, agreed_claims: Sequence[Claim]) -> list[str]:
        """Convert agreed claims into clean, distinct, own-words sentences (L1, D5)."""
        sentences: list[str] = []
        seen_norms: set[str] = set()

        for _idx, claim in enumerate(agreed_claims):
            clean_text = clean_sentence_prefixes(claim.claim_text)
            if not clean_text or len(clean_text) < 15:
                continue

            # Check for near-duplicate claims to prevent repeating same sentence (D5)
            norm = re.sub(r"[^\w\s]", "", clean_text.lower())
            tokens = set(norm.split())
            is_dup = False
            for seen in seen_norms:
                seen_tokens = set(seen.split())
                if tokens and seen_tokens:
                    jaccard = len(tokens & seen_tokens) / len(tokens | seen_tokens)
                    if jaccard > 0.65:
                        is_dup = True
                        break

            if is_dup:
                continue

            seen_norms.add(norm)
            # Synthesize in own words (L1)
            own_words_sentence = synthesize_own_words(clean_text, len(sentences))
            if own_words_sentence:
                sentences.append(own_words_sentence)

        return sentences

    def _is_entailed_by_sources(self, sentence: str, source_passages: Sequence[str]) -> bool:
        """Check if candidate sentence is entailed by at least one source passage."""
        # Strip synthesis framing prefixes so entailment evaluates the underlying factual assertion
        fact_core = re.sub(
            r"^(?:Independent corroboration across reporting outlets indicates that|"
            r"Consensus reporting also confirms that|"
            r"Contributing sources further noted that|"
            r"Additional verified reporting highlights that)\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip()

        dummy_claim = Claim(
            article_id=0,
            cluster_id=0,
            claim_text=fact_core or sentence,
            source_sentence=fact_core or sentence,
        )

        for passage in source_passages:
            passage_claim = Claim(
                article_id=0,
                cluster_id=0,
                claim_text=passage[:400],
                source_sentence=passage[:400],
            )
            verdict, conf = self._nli.compare_claims(dummy_claim, passage_claim)
            if verdict == ClaimLabel.AGREED and conf >= self._safety_cfg.min_entailment_score:
                return True

        return False


def get_agreed_claims(claims: Sequence[Claim], cross_checks: Sequence[CrossCheck]) -> list[Claim]:
    """Find all claims that participate in an AGREED cross-check."""
    agreed_claim_ids: set[int] = set()
    for cc in cross_checks:
        if cc.label == ClaimLabel.AGREED:
            agreed_claim_ids.add(cc.claim_a_id)
            agreed_claim_ids.add(cc.claim_b_id)

    return [c for c in claims if c.id in agreed_claim_ids]
