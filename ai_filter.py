"""
ai_filter.py
============
Groq (llama-3.3-70b-versatile) scoring and tier-based filtering.

Migration note (2026-06-18)
---------------------------
Replaced Google Gemini SDK with Groq to bypass free-tier 429 (Resource
Exhausted) rate limits.  The Groq free tier offers significantly higher
requests-per-minute on llama-3.3-70b-versatile.

Pipeline
--------
1. Assemble a plain-text prompt from all raw job dicts (batched to stay
   within token limits, default batch size = 10).
2. Send to ``llama-3.3-70b-versatile`` via the Groq Chat Completions API.
3. Strip any accidental markdown fences from the response.
4. Parse and validate the returned JSON through ``JobEvaluationList`` (Pydantic).
5. Apply tier filter:
     - Keep ALL  ``Tier_A_Strong`` jobs.
     - Keep      ``Tier_B_Fuzzy``  jobs only if ``match_score >= MIN_TIER_B_SCORE``.
     - Discard   ``Tier_C_None``   jobs unconditionally.

Fallback (keyword-based)
------------------------
If the Groq API call fails (rate-limit, network error, bad JSON, etc.) for a
given batch, a local rule-based keyword classifier is used so the pipeline
never crashes.  Keyword-matched jobs are assigned conservative scores and
flagged in their ``logic_summary``.

Error handling
--------------
A ``ValueError`` is raised only if *both* the Groq call and the keyword
fallback fail for the same batch, which in practice should never happen.
``main.py`` catches ``ValueError`` and aborts with a log entry.
"""

import json
import logging
import os
import re
import time

from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    MIN_TIER_B_SCORE,
    TIER_A,
    TIER_B,
    TIER_C,
)
from models import JobEvaluation, JobEvaluationList

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Jobs per Groq request — keep lower than Gemini batch to stay within
# Groq free-tier context window and token-per-minute limits.
_BATCH_SIZE: int = 10

# Seconds to sleep between successive Groq API calls (free-tier courtesy).
_INTER_BATCH_DELAY: float = 2.0

# ---------------------------------------------------------------------------
# Keyword sets for the local fallback classifier
# ---------------------------------------------------------------------------

_TIER_A_KEYWORDS: frozenset[str] = frozenset([
    "fastapi", "flask", "django", "rest api", "restful", "backend engineer",
    "backend developer", "python developer", "python backend", "api development",
    "fresher", "entry level", "entry-level", "0-1 year", "0 to 1 year",
    "internship", "intern", "graduate trainee", "junior developer",
    "junior engineer", "jwt", "authentication", "sqlalchemy", "postgresql",
    "mysql", "fastapi developer",
])

_TIER_B_KEYWORDS: frozenset[str] = frozenset([
    "python", "sql", "software engineer", "software developer", "full stack",
    "fullstack", "cloud", "aws", "azure", "gcp", "devops", "backend",
    "1-2 year", "1 to 2 year", "associate engineer", "associate developer",
    "api", "microservices", "docker", "kubernetes", "data engineer",
    "machine learning", "ml engineer",
])

_DISCARD_KEYWORDS: frozenset[str] = frozenset([
    "3+ years", "3 years", "4 years", "5 years", "senior", "lead", "manager",
    "director", "architect", "principal", "exclusively java", "spring boot",
    "exclusively node", "frontend only", "react developer", "angular developer",
    "vue developer", "ui developer", "graphic designer", "product manager",
    "hr", "human resources", "sales", "business development",
])

# Junk / automated-alert phrases — jobs whose title or body contains any of
# these are LinkedIn notifications, feed digests, etc., not real postings.
_JUNK_BLACKLIST: list[str] = [
    "alert", "has been created", "feed", "notification", "job alert",
    "your search", "digest", "unsubscribe", "email preference",
    "manage alerts", "forwarded this email",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Pre-compiled word-boundary patterns for fallback keyword matching.
# Using \b prevents substring false positives (e.g. "hr" matching "shred",
# "ml" matching "html").  Patterns are compiled once at import time.
_TIER_A_PATTERNS: list[re.Pattern] = [
    re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in _TIER_A_KEYWORDS
]
_TIER_B_PATTERNS: list[re.Pattern] = [
    re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in _TIER_B_KEYWORDS
]
_DISCARD_PATTERNS: list[re.Pattern] = [
    re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in _DISCARD_KEYWORDS
]

def _build_prompt(batch: list[dict]) -> str:
    """
    Build a plain-text prompt optimised for Llama 3.3.

    The model is commanded to return a raw JSON array with **no** markdown
    code fences and **no** conversational text — just the JSON.

    Each element in the output array must contain exactly these keys:
      id, title, company, tier, match_score, reason, clean_apply_url

    Tiers:
      Tier_A_Strong  — strong Python backend / fresher match
      Tier_B_Fuzzy   — adjacent / partial match or 1-2 yr experience
      Tier_C_None    — discard (3+ yrs, wrong stack, non-engineering)
    """
    header = (
        "You are an expert IT technical recruiter evaluating job postings for an "
        "entry-level Python Backend Engineer (0-1 year experience). "
        "The candidate's skills: Python, FastAPI, REST APIs, SQL, JWT, Authentication.\n\n"
        "TARGET ROLES: Backend Engineer, Python Developer, Python Backend Developer, "
        "Backend Intern, Software Engineer (Python).\n\n"
        "CLASSIFICATION RULES:\n"
        "  Tier_A_Strong : Explicitly requires Python backend (FastAPI/Django/Flask/REST) "
        "AND targets freshers/interns/0-1 yr candidates.\n"
        "  Tier_B_Fuzzy  : General software roles with Python/SQL, OR entry-level roles "
        "where backend is part of the stack, OR roles asking 1-2 yrs where a strong "
        "fresher can still apply.\n"
        "  Tier_C_None   : Requires 3+ years, is purely frontend, uses an exclusively "
        "different stack (Java/Spring Boot only, Node.js only), or is non-engineering. "
        "Also classify automated notification emails (e.g. 'job alert', 'your alert has "
        "been created', digest summaries) as Tier_C_None.\n\n"
        "STRICT OUTPUT FORMAT — respond with a raw JSON array ONLY.\n"
        "Do NOT wrap it in ```json code fences.\n"
        "Do NOT add any explanation, greeting, or text outside the JSON.\n"
        "Each element must have exactly these keys:\n"
        "  id            (integer — same as the job number below)\n"
        "  title         (string  — job title extracted from the posting)\n"
        "  company       (string  — company name)\n"
        "  tier          (string  — one of: Tier_A_Strong | Tier_B_Fuzzy | Tier_C_None)\n"
        "  match_score   (integer — MUST be between 1 and 100 inclusive. "
        "NEVER output 0. If a job is completely irrelevant or is an automated "
        "notification, assign match_score = 1, not 0.)\n"
        "  reason        (string  — one concise sentence explaining the tier/score)\n"
        "  clean_apply_url (string — application URL with tracking params stripped)\n\n"
        "JOB POSTINGS:\n"
    )

    job_blocks: list[str] = []
    for i, job in enumerate(batch, start=1):
        apply_url = job.get("clean_url") or job.get("apply_url", "")
        text_snippet = job.get("text", "")[:1500]  # trim to keep token count low
        job_blocks.append(
            f"--- Job #{i} ---\n"
            f"Subject/Title : {job.get('title', 'Unknown')}\n"
            f"Apply URL     : {apply_url}\n"
            f"Body Text     :\n{text_snippet}\n"
        )

    return header + "\n".join(job_blocks)


def _strip_markdown_fences(raw: str) -> str:
    """
    Remove markdown code fences (```json ... ``` or ``` ... ```) that the
    model may emit despite being instructed not to.
    """
    # Remove leading/trailing whitespace first
    raw = raw.strip()
    # Pattern: optional language tag after opening fence
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _call_groq(client: Groq, prompt: str) -> str:
    """
    Send *prompt* to the Groq Chat Completions endpoint and return the raw
    response text.

    Raises
    ------
    Exception
        Re-raises any exception from the Groq SDK so the caller can decide
        whether to fall back to keyword classification.
    """
    logger.debug("Sending request to Groq model '%s'.", GROQ_MODEL)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.1,   # near-deterministic tier assignments
    )
    return chat_completion.choices[0].message.content


def _parse_groq_response(raw_text: str, batch: list[dict]) -> list[JobEvaluation]:
    """
    Parse and Pydantic-validate the JSON returned by Groq.

    Returns a list of ``JobEvaluation`` objects on success, or raises
    ``ValueError`` with a descriptive message on failure.
    """
    cleaned = _strip_markdown_fences(raw_text)
    logger.debug("Groq response (stripped, first 500 chars): %s", cleaned[:500])

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Groq returned non-JSON response: {exc}\n"
            f"Raw output (first 400 chars): {raw_text[:400]}"
        ) from exc

    # The model may return a bare array or an object wrapping one.
    if isinstance(parsed, dict):
        # Try common wrapper keys: "jobs", "results", "data"
        for key in ("jobs", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            raise ValueError(
                f"Groq returned a JSON object but no recognisable array key found. "
                f"Keys present: {list(parsed.keys())}"
            )

    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected a JSON array from Groq, got {type(parsed).__name__}."
        )

    # Normalise field names: the model might use 'reason' instead of
    # 'logic_summary' and 'tier' instead of 'match_tier'.
    normalised: list[dict] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue

        # Clamp match_score to [1, 100] — prevents Pydantic crash if the
        # model returns 0 despite being told not to.
        raw_score = item.get("match_score", 50)
        try:
            raw_score = int(raw_score)
        except (TypeError, ValueError):
            raw_score = 50
        clamped_score = max(1, min(100, raw_score))

        normalised.append({
            "job_title":       item.get("title", item.get("job_title", "Unknown")),
            "company":         item.get("company", "Unknown"),
            "match_tier":      item.get("tier", item.get("match_tier", TIER_C)),
            "match_score":     clamped_score,
            "logic_summary":   item.get("reason", item.get("logic_summary", "")),
            "clean_apply_url": item.get("clean_apply_url",
                                        batch[idx].get("clean_url", "")
                                        if idx < len(batch) else ""),
        })

    try:
        evaluation_list = JobEvaluationList.model_validate({"jobs": normalised})
    except Exception as exc:
        raise ValueError(
            f"Pydantic validation failed for Groq response: {exc}"
        ) from exc

    return evaluation_list.jobs


def _keyword_fallback(batch: list[dict]) -> list[JobEvaluation]:
    """
    Local rule-based classifier used when the Groq API call fails.

    Inspects ``title`` and ``text`` fields for known keywords and assigns
    a conservative tier and score.  Automated alert/notification emails are
    detected via ``_JUNK_BLACKLIST`` and silently skipped.

    This ensures the pipeline always produces *some* output even during
    prolonged API outages, without polluting the report with junk.
    """
    logger.warning(
        "Groq API unavailable for this batch — running keyword fallback classifier "
        "on %d job(s).",
        len(batch),
    )
    results: list[JobEvaluation] = []

    for job in batch:
        title_lower = job.get("title", "").lower()
        desc_lower = job.get("text", "").lower()
        combined = title_lower + " " + desc_lower
        apply_url = job.get("clean_url") or job.get("apply_url", "")

        # ── Filter out junk / automated alert emails ──────────────────
        is_junk = any(
            junk in title_lower or junk in desc_lower
            for junk in _JUNK_BLACKLIST
        )
        if is_junk:
            logger.info(
                "  🗑️ JUNK  Skipping automated alert/notification: %s",
                job.get("title", "?"),
            )
            continue

        # ── Check discard signals (word-boundary matching) ─────────────
        if any(pat.search(combined) for pat in _DISCARD_PATTERNS):
            tier = TIER_C
            score = 20
            reason = "[Keyword Fallback] Discard keywords detected (3+ yrs / senior / wrong stack)."
        elif any(pat.search(combined) for pat in _TIER_A_PATTERNS):
            tier = TIER_A
            score = 72   # conservative — we don't have AI confidence here
            reason = "[Keyword Fallback] Strong Python backend / fresher keywords matched."
        elif any(pat.search(combined) for pat in _TIER_B_PATTERNS):
            tier = TIER_B
            score = 72   # matches MIN_TIER_B_SCORE so fuzzy jobs survive Layer 4 during API outages
            reason = "[Keyword Fallback] General Python/software keywords matched."
        else:
            tier = TIER_C
            score = 1
            reason = "[Keyword Fallback] No relevant keywords found."

        try:
            results.append(
                JobEvaluation(
                    job_title=job.get("title", "Unknown"),
                    company=job.get("source", "Unknown"),
                    match_tier=tier,       # type: ignore[arg-type]
                    match_score=score,
                    logic_summary=reason,
                    clean_apply_url=apply_url,
                )
            )
        except Exception as exc:
            logger.warning("Keyword fallback skipped one job due to validation error: %s", exc)

    logger.info(
        "Keyword fallback produced %d evaluation(s) for batch.", len(results)
    )
    return results


def _apply_tier_filter(evaluations: list[JobEvaluation]) -> list[JobEvaluation]:
    """
    Return only jobs that pass the tier threshold:
    - All  Tier_A_Strong
    - Tier_B_Fuzzy with match_score >= MIN_TIER_B_SCORE
    - Tier_C_None is always discarded
    """
    kept: list[JobEvaluation] = []

    for job in evaluations:
        if job.match_tier == TIER_A:
            kept.append(job)
            logger.info(
                "  ✅ KEEP [%s | score=%d] %s @ %s",
                job.match_tier, job.match_score, job.job_title, job.company,
            )
        elif job.match_tier == TIER_B and job.match_score >= MIN_TIER_B_SCORE:
            kept.append(job)
            logger.info(
                "  ✅ KEEP [%s | score=%d] %s @ %s",
                job.match_tier, job.match_score, job.job_title, job.company,
            )
        else:
            logger.info(
                "  ❌ DROP [%s | score=%d] %s @ %s",
                job.match_tier, job.match_score, job.job_title, job.company,
            )

    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_jobs(raw_jobs: list[dict]) -> list[JobEvaluation]:
    """
    Score and filter *raw_jobs* using Groq (llama-3.3-70b-versatile).

    Parameters
    ----------
    raw_jobs : list[dict]
        Jobs that have already been deduplicated (output of ``dedup.filter_new``).

    Returns
    -------
    list[JobEvaluation]
        Jobs that passed the tier filter, ready for email delivery.

    Raises
    ------
    ValueError
        Only if both the Groq API call and the keyword fallback fail for the
        same batch — practically unreachable.
    """
    if not raw_jobs:
        logger.info("No jobs to evaluate.")
        return []

    if not GROQ_API_KEY:
        logger.error(
            "GROQ_API_KEY is not set. Cannot initialise Groq client. "
            "Falling back to keyword classifier for all jobs."
        )
        all_evals = _keyword_fallback(raw_jobs)
        return _apply_tier_filter(all_evals)

    client = Groq(api_key=GROQ_API_KEY)
    all_kept: list[JobEvaluation] = []
    total_batches = -(-len(raw_jobs) // _BATCH_SIZE)  # ceiling division

    for batch_start in range(0, len(raw_jobs), _BATCH_SIZE):
        batch = raw_jobs[batch_start: batch_start + _BATCH_SIZE]
        batch_num = (batch_start // _BATCH_SIZE) + 1

        logger.info(
            "Sending batch %d/%d (%d job(s)) to Groq [%s] …",
            batch_num, total_batches, len(batch), GROQ_MODEL,
        )

        evaluations: list[JobEvaluation] = []

        try:
            prompt = _build_prompt(batch)
            raw_response = _call_groq(client, prompt)
            evaluations = _parse_groq_response(raw_response, batch)
            logger.info(
                "Batch %d/%d: received %d valid evaluation(s) from Groq.",
                batch_num, total_batches, len(evaluations),
            )

        except json.JSONDecodeError as exc:
            logger.warning(
                "Batch %d/%d: JSON parse error (%s). Activating keyword fallback.",
                batch_num, total_batches, exc,
            )
            evaluations = _keyword_fallback(batch)

        except ValueError as exc:
            logger.warning(
                "Batch %d/%d: Pydantic / format error (%s). Activating keyword fallback.",
                batch_num, total_batches, exc,
            )
            evaluations = _keyword_fallback(batch)

        except Exception as exc:  # noqa: BLE001
            # Catches Groq rate-limit (429), network errors, auth errors, etc.
            logger.warning(
                "Batch %d/%d: Groq API error (%s: %s). Activating keyword fallback.",
                batch_num, total_batches, type(exc).__name__, exc,
            )
            evaluations = _keyword_fallback(batch)

        # Apply tier filter to this batch's evaluations
        logger.info(
            "Batch %d/%d: applying tier filter to %d evaluation(s) …",
            batch_num, total_batches, len(evaluations),
        )
        kept = _apply_tier_filter(evaluations)
        all_kept.extend(kept)

        # Polite inter-batch delay — respects Groq free-tier rate limits
        # without the 65-second penalty of the previous Gemini implementation.
        if batch_start + _BATCH_SIZE < len(raw_jobs):
            logger.debug(
                "Sleeping %.1fs before next batch to respect Groq rate limits.",
                _INTER_BATCH_DELAY,
            )
            time.sleep(_INTER_BATCH_DELAY)

    logger.info(
        "AI filtering complete: %d/%d job(s) passed the tier threshold.",
        len(all_kept), len(raw_jobs),
    )
    return all_kept
