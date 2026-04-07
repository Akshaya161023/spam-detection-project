import socket
import urllib.request
import ssl

print("=" * 50)
print("NETWORK DIAGNOSTIC")
print("=" * 50)

tests = [
    ("Google homepage",     "https://www.google.com"),
    ("Gmail OAuth",         "https://oauth2.googleapis.com"),
    ("Google APIs",         "https://www.googleapis.com"),
    ("Gmail API",           "https://gmail.googleapis.com"),
]

for name, url in tests:
    try:
        urllib.request.urlopen(url, timeout=10)
        print(f"  PASS  {name}")
    except urllib.error.HTTPError as e:
        print(f"  PASS  {name} (HTTP {e.code} — server reached)")
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        {e}")

print()
print("Checking proxy settings...")
import urllib.request as req
proxies = req.getproxies()
if proxies:
    print(f"  Proxy detected: {proxies}")
else:
    print("  No proxy configured")

print()
print("Checking SSL...")
try:
    context = ssl.create_default_context()
    with socket.create_connection(("www.googleapis.com", 443), timeout=10) as sock:
        with context.wrap_socket(sock, server_hostname="www.googleapis.com") as ssock:
            print(f"  SSL OK — {ssock.version()}")
except Exception as e:
    print(f"  SSL FAIL — {e}")