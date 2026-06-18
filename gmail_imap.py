"""
gmail_imap.py
=============
Stream A — Gmail IMAP ingestion and post-delivery flag management.

Security contract
-----------------
- This module ONLY ever calls ``imap.select(GMAIL_FOLDER)`` where
  ``GMAIL_FOLDER = "Daily-Jobs"``.  No other mailbox is opened.
- Messages are fetched read-only during ingestion; the ``READONLY`` flag is
  passed to ``imap.select`` during the fetch phase so no accidental mutation
  occurs.
- ``mark_as_read()`` opens a *separate* authenticated connection and sets
  ``+FLAGS \\Seen`` on the exact UID list provided by the caller.  It is
  called exclusively from ``main.py`` **after** a successful SMTP send.
- No message is ever deleted or moved.
"""

from email.message import Message
import imaplib
import logging
from email.header import decode_header, make_header

from bs4 import BeautifulSoup

from config import (
    GMAIL_APP_PASSWORD,
    GMAIL_FOLDER,
    GMAIL_USER_EMAIL,
    IMAP_HOST,
    IMAP_PORT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_mime_header(raw: str | bytes | None) -> str:
    """Safely decode a MIME-encoded email header value to plain text."""
    if raw is None:
        return ""
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return str(raw)


def _extract_text_and_urls(msg: email.message.Message) -> tuple[str, list[str]]:
    """
    Walk a parsed email message and return:
    - plain/concatenated text extracted from HTML parts (via BeautifulSoup)
    - all unique href URLs found in <a> tags inside those HTML parts
    """
    text_parts: list[str] = []
    urls: list[str] = []

    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type not in ("text/html", "text/plain"):
            continue

        payload = part.get_payload(decode=True)
        if not payload:
            continue

        charset = part.get_content_charset() or "utf-8"
        try:
            decoded = payload.decode(charset, errors="replace")
        except (LookupError, ValueError):
            decoded = payload.decode("utf-8", errors="replace")

        if content_type == "text/html":
            soup = BeautifulSoup(decoded, "html.parser")
            text_parts.append(soup.get_text(separator=" ", strip=True))
            for tag in soup.find_all("a", href=True):
                href = tag["href"].strip()
                if href.startswith("http") and href not in urls:
                    urls.append(href)
        else:
            text_parts.append(decoded.strip())

    return " ".join(text_parts), urls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_unread_jobs() -> tuple[list[dict], list[bytes]]:
    """
    Connect to Gmail via IMAP, open **only** the ``Daily-Jobs`` folder, and
    return all unread messages as structured dicts alongside their raw UIDs.

    Returns
    -------
    raw_jobs : list[dict]
        Each dict contains:
          - ``"title"``     : subject line of the alert email
          - ``"text"``      : plain text extracted from the HTML body
          - ``"apply_url"`` : the first plausible job-application URL found
          - ``"all_urls"``  : full list of URLs extracted from the email
    email_uids : list[bytes]
        Raw IMAP UIDs to be passed to ``mark_as_read()`` after a successful
        SMTP send.  Empty list if no unread messages exist.
    """
    raw_jobs: list[dict] = []
    email_uids: list[bytes] = []

    logger.info("Connecting to IMAP: %s:%d", IMAP_HOST, IMAP_PORT)
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)

    try:
        imap.login(GMAIL_USER_EMAIL, GMAIL_APP_PASSWORD)
        logger.info("IMAP login successful.")

        # ── Security boundary: ONLY Daily-Jobs, opened READONLY ──────────
        status, _ = imap.select(f'"{GMAIL_FOLDER}"', readonly=True)
        if status != "OK":
            logger.error(
                "Failed to select mailbox '%s'. Status: %s", GMAIL_FOLDER, status
            )
            return [], []

        logger.info("Opened mailbox '%s' (readonly). Searching for UNREAD...", GMAIL_FOLDER)
        status, data = imap.search(None, "(UNSEEN)")
        if status != "OK" or not data or not data[0]:
            logger.info("No unread messages found in '%s'.", GMAIL_FOLDER)
            return [], []

        uid_list: list[bytes] = data[0].split()
        logger.info("Found %d unread message(s).", len(uid_list))

        for uid in uid_list:
            status, msg_data = imap.fetch(uid, "(RFC822)")
            if status != "OK" or not msg_data:
                logger.warning("Failed to fetch UID %s; skipping.", uid)
                continue

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            subject = _decode_mime_header(msg.get("Subject", ""))
            text, urls = _extract_text_and_urls(msg)

            # Pick the first URL that looks like a job-application link.
            # Heuristic: prefer URLs containing keywords; fall back to first URL.
            apply_url = ""
            keywords = ("apply", "job", "career", "internshala", "linkedin",
                        "naukri", "indeed", "lever", "greenhouse", "workday")
            for url in urls:
                if any(kw in url.lower() for kw in keywords):
                    apply_url = url
                    break
            if not apply_url and urls:
                apply_url = urls[0]

            if not apply_url:
                logger.debug("No apply URL found in message UID=%s; skipping.", uid)
                continue

            raw_jobs.append(
                {
                    "title": subject,
                    "text": text[:4000],  # cap to avoid huge Gemini prompts
                    "apply_url": apply_url,
                    "all_urls": urls,
                }
            )
            email_uids.append(uid)

    finally:
        try:
            imap.logout()
        except Exception:
            pass

    logger.info(
        "IMAP ingestion complete: %d job(s) extracted from %d unread email(s).",
        len(raw_jobs),
        len(email_uids),
    )
    return raw_jobs, email_uids


def mark_as_read(uids: list[bytes]) -> None:
    """
    Open a fresh IMAP connection (read-write) and mark the given *uids* as
    ``\\Seen`` inside the ``Daily-Jobs`` folder.

    This is called **only** from ``main.py`` after a confirmed successful SMTP
    send.  It is never called on any other folder.

    Parameters
    ----------
    uids : list[bytes]
        The raw UID byte-strings returned by ``fetch_unread_jobs()``.
    """
    if not uids:
        logger.info("No UIDs to mark as read; skipping.")
        return

    logger.info("Marking %d message(s) as read in '%s'.", len(uids), GMAIL_FOLDER)
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)

    try:
        imap.login(GMAIL_USER_EMAIL, GMAIL_APP_PASSWORD)

        # ── Security boundary: ONLY Daily-Jobs, read-write this time ─────
        status, _ = imap.select(f'"{GMAIL_FOLDER}"', readonly=False)
        if status != "OK":
            logger.error(
                "Could not open '%s' for writing flags. UIDs NOT marked.",
                GMAIL_FOLDER,
            )
            return

        uid_str = b",".join(uids)
        status, _ = imap.store(uid_str, "+FLAGS", "\\Seen")
        if status == "OK":
            logger.info("Successfully marked %d message(s) as read.", len(uids))
        else:
            logger.error("store() returned non-OK status: %s", status)

    finally:
        try:
            imap.logout()
        except Exception:
            pass
