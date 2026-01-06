# test_langdetect.py
from langdetect import detect, DetectorFactory

# Set seed for consistent results
DetectorFactory.seed = 0

test_texts = [
    "Hello, how are you today?",
    "नमस्ते, आप कैसे हैं?",
    "హలో, మీరు ఎలా ఉన్నారు?",
    "Bonjour, comment allez-vous?"
]

for text in test_texts:
    try:
        lang = detect(text)
        print(f"Text: {text[:30]}...")
        print(f"Detected language: {lang}")
        print("-" * 50)
    except Exception as e:
        print(f"Error detecting language for '{text[:30]}...': {e}")