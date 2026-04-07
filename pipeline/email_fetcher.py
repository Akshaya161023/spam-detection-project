import os
import sys
import base64
import json
import requests
import yaml
from bs4 import BeautifulSoup
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.predict import predict

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDS_PATH  = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH  = os.path.join(BASE_DIR, "token.json")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FETCH_LIMIT = config.get("email_fetch_limit", 20)
SCOPES      = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_BASE  = "https://gmail.googleapis.com/gmail/v1/users/me"


# ── Auth ─────────────────────────────────────────────────────
def get_credentials():
    """Get valid OAuth2 credentials, refreshing or re-authenticating as needed."""
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                print("Token refreshed.")
            except Exception as e:
                print(f"Token refresh failed: {e}. Re-authenticating...")
                creds = None

        if not creds:
            if not os.path.exists(CREDS_PATH):
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDS_PATH}"
                )
            flow  = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=8080)
            print("Login successful.")

        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
        print(f"Token saved.")

    return creds


# ── Requests session with retry ───────────────────────────────
def get_session(creds):
    """Create a requests session with auth header and retry logic."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {creds.token}",
        "Content-Type":  "application/json"
    })

    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    return session


# ── Gmail API calls using requests ───────────────────────────
def list_messages(session, max_results=20):
    """List messages from inbox."""
    url    = f"{GMAIL_BASE}/messages"
    params = {
        "maxResults": max_results,
        "q":          "in:inbox"
    }
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("messages", [])


def get_message(session, msg_id):
    """Get full message by ID."""
    url  = f"{GMAIL_BASE}/messages/{msg_id}"
    resp = session.get(url, params={"format": "full"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Helpers ───────────────────────────────────────────────────
def decode_body(payload):
    """Recursively decode email body from base64."""
    body = ""
    if "parts" in payload:
        for part in payload["parts"]:
            body += decode_body(part)
    else:
        data = payload.get("body", {}).get("data", "")
        if data:
            decoded = base64.urlsafe_b64decode(data + "==")
            body    = decoded.decode("utf-8", errors="ignore")
    return body


def clean_html(raw):
    """Strip HTML tags."""
    if "<" in raw:
        return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    return raw.strip()


def get_header(headers, name):
    """Extract a header value by name."""
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return "Unknown"


# ── Main fetch function ───────────────────────────────────────
def fetch_emails(session, max_results=None):
    """Fetch and parse emails from Gmail."""
    if max_results is None:
        max_results = FETCH_LIMIT

    messages = list_messages(session, max_results)

    if not messages:
        print("No messages found.")
        return []

    emails = []
    for msg in messages:
        try:
            full    = get_message(session, msg["id"])
            headers = full["payload"].get("headers", [])

            sender  = get_header(headers, "From")
            subject = get_header(headers, "Subject")
            date    = get_header(headers, "Date")
            body    = clean_html(decode_body(full["payload"]))

            emails.append({
                "id":      msg["id"],
                "sender":  sender,
                "subject": subject,
                "date":    date,
                "body":    body[:2000]
            })
        except Exception as e:
            print(f"Skipping message {msg['id']}: {e}")
            continue

    print(f"Fetched {len(emails)} emails.")
    return emails


# ── Fetch + predict ───────────────────────────────────────────
def fetch_and_predict(max_results=None):
    """Full pipeline: auth → fetch → predict → return results."""
    creds   = get_credentials()
    session = get_session(creds)
    emails  = fetch_emails(session, max_results)

    results = []
    for email in emails:
        try:
            text       = f"{email.get('subject', '')} {email.get('body', '')}"
            prediction = predict(text)

            results.append({
                "id":               email.get("id", ""),
                "sender":           email.get("sender", "Unknown"),
                "subject":          email.get("subject", "No subject"),
                "date":             email.get("date", "Unknown"),
                "body_preview":     email.get("body", "")[:200],
                "label":            prediction.get("label", "unknown"),
                "is_spam":          prediction.get("is_spam", False),
                "spam_confidence":  prediction.get("spam_confidence", 0.0),
                "ham_confidence":   prediction.get("ham_confidence", 0.0),
                "threshold_used":   prediction.get("threshold_used", 0.6),
            })
        except Exception as e:
            results.append({
                "id":               email.get("id", "error"),
                "sender":           email.get("sender", "Unknown"),
                "subject":          email.get("subject", "No subject"),
                "date":             email.get("date", "Unknown"),
                "body_preview":     "Could not process.",
                "label":            "unknown",
                "is_spam":          False,
                "spam_confidence":  0.0,
                "ham_confidence":   0.0,
                "threshold_used":   0.6,
            })
            print(f"Prediction error: {e}")

    return results


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Gmail connection with requests library...")
    try:
        creds   = get_credentials()
        session = get_session(creds)

        # Test profile fetch
        resp    = session.get(f"{GMAIL_BASE}/profile", timeout=30)
        profile = resp.json()
        print(f"Connected as  : {profile.get('emailAddress')}")
        print(f"Total messages: {profile.get('messagesTotal')}")

        # Fetch and predict
        results = fetch_and_predict(max_results=5)
        print(f"\nFetched {len(results)} emails:")
        for r in results:
            status = "SPAM" if r["is_spam"] else "HAM"
            print(f"  [{status}] {r['subject'][:50]} — {r['spam_confidence']*100:.1f}%")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()