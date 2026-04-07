import os
import joblib
import yaml
from pipeline.preprocess import preprocess

# ── 1. Set base paths ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
config_path = os.path.join(BASE_DIR, "config.yaml")

# ── 2. Load config safely ────────────────────────────────────
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
except FileNotFoundError:
    print("⚠️ config.yaml not found, using default values")
    config = {}

THRESHOLD = config.get("threshold", 0.6)

# ── 3. Load model and vectorizer ─────────────────────────────
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ── 4. Prediction function ───────────────────────────────────
def predict(text: str) -> dict:
    clean = preprocess(text)
    vec = vectorizer.transform([clean])

    probs = model.predict_proba(vec)[0]
    ham_confidence = float(probs[0])
    spam_confidence = float(probs[1])

    is_spam = spam_confidence > THRESHOLD

    return {
        "is_spam": is_spam,
        "spam_confidence": spam_confidence,
        "ham_confidence": ham_confidence,
        "threshold_used": THRESHOLD
    }


# ── 5. Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    test = "Congratulations! You won a FREE iPhone. Click now!"
    result = predict(test)

    print("\nTest Text:", test)
    print("Prediction:", result)