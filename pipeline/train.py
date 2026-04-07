import pandas as pd
import joblib
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.preprocess import preprocess

# ── 1. Load ───────────────────────────────────────────────────
df = pd.read_csv("data/spam.csv", encoding="latin-1")
print(f"Total rows: {len(df)}")
print(df["label"].value_counts())

# ── 2. Preprocess ─────────────────────────────────────────────
print("\nPreprocessing...")
df["cleaned"] = df["message"].apply(preprocess)
df["label_num"] = df["label"].map({"spam": 1, "ham": 0})
df = df.dropna(subset=["label_num"])

# ── 3. Split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["label_num"],
    test_size=0.2, random_state=42,
    stratify=df["label_num"]
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. TF-IDF with better parameters ─────────────────────────
vectorizer = TfidfVectorizer(
    max_features=8000,       # more features
    ngram_range=(1, 2),      # unigrams + bigrams catches phrases
    min_df=1,                # include rare but important words
    sublinear_tf=True        # dampen high frequency terms
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── 5. Train with tuned alpha ─────────────────────────────────
# Lower alpha = less smoothing = model trusts data more
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test_vec)

print("\n── Evaluation ────────────────────────────────")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print("\nFull Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# ── 7. Save ───────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model,      "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("Saved model.pkl and vectorizer.pkl")