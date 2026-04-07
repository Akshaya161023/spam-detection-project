from pipeline.predict import predict

# ── 5 test cases ─────────────────────────────────────────────
test_cases = [
    {
        "type"  : "OBVIOUS SPAM",
        "text"  : "WINNER!! You have been selected for a $1000 Walmart gift card. Call now to claim your FREE prize!"
    },
    {
        "type"  : "SPAM (phishing)",
        "text"  : "Urgent: Your bank account has been suspended. Verify your details immediately at http://secure-login.xyz"
    },
    {
        "type"  : "NORMAL (friend)",
        "text"  : "Hey, are you coming to the meeting tomorrow at 10am? Let me know if you need the Zoom link."
    },
    {
        "type"  : "NORMAL (work)",
        "text"  : "Please find attached the quarterly report for your review. Let me know if you have any questions."
    },
    {
        "type"  : "BORDERLINE (newsletter)",
        "text"  : "Get 20% off your next purchase! Limited time offer for our valued subscribers only."
    },
]

# ── Run and print results ─────────────────────────────────────
print("=" * 60)
print("  SPAM DETECTOR — PREDICTION TEST")
print("=" * 60)

for i, case in enumerate(test_cases, 1):
    result = predict(case["text"])
    status = "SPAM" if result["is_spam"] else "HAM"
    bar    = int(result["spam_confidence"] * 20)  # scale to 20 chars
    visual = "[" + "#" * bar + "-" * (20 - bar) + "]"

    print(f"\nTest {i} — {case['type']}")
    print(f"  Text       : {case['text'][:65]}...")
    print(f"  Result     : {status}")
    print(f"  Spam conf  : {visual} {result['spam_confidence']*100:.1f}%")
    print(f"  Ham conf   : {result['ham_confidence']*100:.1f}%")
    print(f"  Threshold  : {result['threshold_used']}")

print("\n" + "=" * 60)
print("All tests complete.")
print("=" * 60)