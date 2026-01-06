# app/spellchecker.py
import os
import string
from symspellpy import SymSpell, Verbosity

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# naming convention: app/data/{lang}_word_frequency.txt
DEFAULT_LANG = "en"

class SpellChecker:
    def __init__(self, max_edit_distance=2, prefix_length=7):
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        # Keep a cache of loaded SymSpell per language
        self._symspell_map = {}  # lang -> SymSpell instance

    def _freq_path_for_lang(self, lang: str):
        lang = (lang or DEFAULT_LANG).lower()
        fname = f"{lang}_word_frequency.txt"
        return os.path.join(DATA_DIR, fname)

    def _load_symspell_for_lang(self, lang: str):
        lang = (lang or DEFAULT_LANG).lower()
        
        if lang in self._symspell_map:
            return self._symspell_map[lang]
        
        freq_path = self._freq_path_for_lang(lang)
        
        if not os.path.exists(freq_path):
            print(f"Dictionary not found for {lang}: {freq_path}")
            self._symspell_map[lang] = None
            return None
        
        sym = SymSpell(
            max_dictionary_edit_distance=self.max_edit_distance,
            prefix_length=self.prefix_length
        )
        
        try:
            # Try to load dictionary with UTF-8 encoding
            with open(freq_path, "r", encoding="utf-8") as f:
                line_count = sym.load_dictionary(f, term_index=0, count_index=1)
            
            if line_count > 0:
                print(f"Successfully loaded {line_count} words for {lang}")
                self._symspell_map[lang] = sym
                return sym
            else:
                print(f"No words loaded for {lang}")
                self._symspell_map[lang] = None
                return None
                
        except Exception as e:
            print(f"Error loading dictionary for {lang}: {str(e)}")
            self._symspell_map[lang] = None
            return None

    def correct_text(self, text: str, lang: str = DEFAULT_LANG, max_results=2) -> str:
        """
        Try to correct using language-specific dictionary.
        If no dictionary, return original text.
        """
        if not text or not isinstance(text, str):
            return text
        
        # For short texts or single words
        if len(text.split()) <= 1:
            return self.correct_word(text, lang)
        
        sym = self._load_symspell_for_lang(lang)
        if sym is None:
            # no-op if we don't have a dictionary
            print(f"No dictionary for {lang}, returning original text")
            return text
        
        try:
            # Use compound lookup for multi-word contexts
            suggestions = sym.lookup_compound(
                text, 
                max_edit_distance=self.max_edit_distance
            )
            if suggestions:
                return suggestions[0].term
        except Exception as e:
            print(f"Error in correct_text for {lang}: {str(e)}")
        
        return text

    def correct_word(self, word: str, lang: str = DEFAULT_LANG) -> str:
        """Correct a single word"""
        if not word or len(word) < 2:
            return word
        
        sym = self._load_symspell_for_lang(lang)
        if sym is None:
            return word
        
        try:
            # Look up the word
            suggestions = sym.lookup(
                word, 
                Verbosity.CLOSEST,
                max_edit_distance=self.max_edit_distance
            )
            
            if suggestions:
                # Return the best suggestion
                return suggestions[0].term
        except Exception as e:
            print(f"Error correcting word '{word}' for {lang}: {str(e)}")
        
        return word

    def get_available_languages(self):
        """Get list of available language dictionaries"""
        available = []
        for lang in ['en', 'hi', 'te', 'ta', 'kn', 'fr']:
            path = self._freq_path_for_lang(lang)
            if os.path.exists(path):
                available.append(lang)
        return available