import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    # Handle None / NaN / non-string values
    if not isinstance(text, str):
        return ""
    if text.strip() == "":
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # 3. Tokenize
    tokens = text.split()

    # 4. Remove stopwords + stem
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]

    return " ".join(tokens)


if __name__ == "__main__":
    sample = "Congratulations! You've WON a FREE prize. Call NOW: 1800-555-0000!!!"
    print("Original:", sample)
    print("Cleaned :", preprocess(sample))
    print("NaN test:", preprocess(float("nan")))
    print("None test:", preprocess(None))