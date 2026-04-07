import sys
sys.path.append(".")

print("Step 1: Testing internet...")
import urllib.request
try:
    urllib.request.urlopen("https://www.google.com", timeout=10)
    print("  OK - Internet works")
except Exception as e:
    print(f"  FAIL - {e}")

print("Step 2: Testing Google APIs...")
try:
    urllib.request.urlopen("https://oauth2.googleapis.com", timeout=10)
    print("  OK - Google APIs reachable")
except Exception as e:
    print(f"  FAIL - {e}")

print("Step 3: Testing Gmail API endpoint...")
try:
    urllib.request.urlopen("https://www.googleapis.com/gmail/v1/", timeout=10)
    print("  OK - Gmail API endpoint reachable")
except Exception as e:
    print(f"  FAIL - {e}")

print("Step 4: Testing credentials.json...")
import os, json
if os.path.exists("credentials.json"):
    with open("credentials.json") as f:
        data = json.load(f)
    keys = list(data.get("installed", data.get("web", {})).keys())
    print(f"  OK - credentials.json found, keys: {keys}")
else:
    print("  FAIL - credentials.json NOT found in project root")

print("Step 5: Testing token.json...")
if os.path.exists("token.json"):
    with open("token.json") as f:
        token = json.load(f)
    print(f"  OK - token.json found")
    print(f"  Expiry: {token.get('expiry', 'unknown')}")
    print(f"  Has refresh_token: {'refresh_token' in token}")
else:
    print("  WARN - token.json not found (will be created on first login)")

print("Step 6: Testing Gmail API auth...")
try:
    from pipeline.email_fetcher import get_gmail_service
    service = get_gmail_service()
    print("  OK - Gmail service created successfully")

    profile = service.users().getProfile(userId="me").execute()
    print(f"  OK - Connected as: {profile.get('emailAddress')}")
    print(f"  Total messages: {profile.get('messagesTotal')}")
except Exception as e:
    print(f"  FAIL - {e}")

print("\nDone. Share the output above so we can identify the exact problem.")