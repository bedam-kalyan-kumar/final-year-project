# app/model.py
import os
from groq import Groq
from dotenv import load_dotenv
import json
import re

load_dotenv()

class GroqModelManager:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in .env")

        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"
        
        # Language code to language name mapping
        self.lang_names = {
            "en": "English",
            "hi": "Hindi",
            "te": "Telugu", 
            "ta": "Tamil",
            "kn": "Kannada",
            "fr": "French"
        }

    def predict_both(self, text, num_words=3, num_sentences=3, lang="en"):
        """
        One SINGLE API call that returns BOTH:
         - next words
         - next sentences
         With language-specific predictions
        """
        
        language_name = self.lang_names.get(lang, "English")
        
        # Create language-specific prompt
        if lang == "te":
            # Telugu specific prompt
            prompt = f"""
మీరు ఒక టెక్స్ట్ ప్రిడిక్షన్ మోడల్.
ఇచ్చిన ఇన్పుట్ టెక్స్ట్ కోసం, ఈ కింది రెండింటినీ జనరేట్ చేయండి:

1) {num_words} సంభావ్య తర్వాతి పదాలు (తెలుగులో)
2) {num_words} సంభావ్య తర్వాతి వాక్యాలు (తెలుగులో)

నియమాలు:
- ఇన్పుట్ టెక్స్ట్‌ను పునరావృతం చేయకండి.
- డూప్లికేట్ అంశాలను ఉత్పత్తి చేయకండి.
- క్రింది JSON ఫార్మాట్‌లో అవుట్‌పుట్ ఇవ్వండి:
{{
  "words": [...],
  "sentences": [...]
}}

ఇన్పుట్ టెక్స్ట్: "{text}"
"""
        elif lang == "hi":
            # Hindi specific prompt
            prompt = f"""
आप एक टेक्स्ट प्रेडिक्शन मॉडल हैं।
दिए गए इनपुट टेक्स्ट के लिए, निम्नलिखित दोनों जनरेट करें:

1) {num_words} संभावित अगले शब्द (हिंदी में)
2) {num_words} संभावित अगले वाक्य (हिंदी में)

नियम:
- इनपुट टेक्स्ट को दोहराएं नहीं।
- डुप्लिकेट आइटम न बनाएं।
- निम्नलिखित JSON फॉर्मेट में आउटपुट दें:
{{
  "words": [...],
  "sentences": [...]
}}

इनपुट टेक्स्ट: "{text}"
"""
        else:
            # English/other languages prompt
            prompt = f"""
You are a predictive text model.
Given input text below, generate BOTH:

1) {num_words} likely next single words in {language_name}
2) {num_words} possible next sentences in {language_name}

Rules:
- Do NOT repeat the input text.
- Do NOT produce duplicate items.
- Output strictly in JSON object like this:
{{
  "words": [...],
  "sentences": [...]
}}

Input text: "{text}"
"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250 if lang not in ["en", "fr"] else 200,
                temperature=0.7,
                top_p=0.9,
            )
        except Exception as e:
            print(f"Model request failed for {lang}:", e)
            return [], []

        # Extract content
        try:
            content = resp.choices[0].message.content
        except Exception:
            try:
                content = getattr(resp.choices[0], 'text', '')
            except Exception:
                content = ""

        if not content:
            return [], []

        # Try to extract JSON
        words = []
        sentences = []
        
        try:
            # Look for JSON pattern
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                words = data.get("words", []) or data.get("word", [])
                sentences = data.get("sentences", []) or data.get("sentence", [])
                
                # Ensure lists
                if not isinstance(words, list):
                    words = [str(words)]
                if not isinstance(sentences, list):
                    sentences = [str(sentences)]
            else:
                # Fallback: look for numbered lists or bullet points
                lines = content.split('\n')
                collecting_words = False
                collecting_sentences = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for word section
                    if 'word' in line.lower() or 'पद' in line or 'పద' in line:
                        collecting_words = True
                        collecting_sentences = False
                        continue
                    elif 'sentence' in line.lower() or 'वाक्य' in line or 'వాక్య' in line:
                        collecting_words = False
                        collecting_sentences = True
                        continue
                    elif line.startswith(('1.', '2.', '3.', '-', '•', '*')):
                        # Remove numbering/bullets
                        clean_line = re.sub(r'^\d+\.\s*|^[-•*]\s*', '', line)
                        if collecting_words and len(words) < num_words:
                            words.append(clean_line)
                        elif collecting_sentences and len(sentences) < num_sentences:
                            sentences.append(clean_line)
                    elif collecting_words and len(words) < num_words and len(line.split()) <= 3:
                        words.append(line)
                    elif collecting_sentences and len(sentences) < num_sentences:
                        sentences.append(line)
                
                # If still empty, use first few non-empty lines
                if not words and not sentences:
                    valid_lines = [l for l in lines if l and len(l.split()) > 0]
                    words = valid_lines[:num_words]
                    sentences = valid_lines[num_words:num_words+num_sentences]
                    
        except Exception as e:
            print(f"Failed to parse model output for {lang}:", e)
            # Last resort: split content
            tokens = content.split()
            words = tokens[:num_words]
            sentences = [' '.join(tokens[i:i+5]) for i in range(num_words, min(len(tokens), num_words+15), 5)]
        
        # Clean and filter
        words = [w.strip(' "\'.,;:!?') for w in words if w and w.strip()]
        sentences = [s.strip(' "\'.,;:!?') for s in sentences if s and s.strip()]
        
        # Remove duplicates while preserving order
        words = list(dict.fromkeys(words))
        sentences = list(dict.fromkeys(sentences))
        
        return words[:num_words], sentences[:num_sentences]

_model_manager = None

def get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = GroqModelManager()
    return _model_manager