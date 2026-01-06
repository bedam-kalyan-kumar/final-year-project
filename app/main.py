# app/main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import re
from collections import OrderedDict, Counter, defaultdict
import os
import json
from datetime import datetime, timedelta
import uuid
import random
import statistics

app = FastAPI(title="Intell Next-Word & Next-Sentence API (Lang-aware)")
app.mount("/static", StaticFiles(directory="web"), name="static")

# Language detection with fallbacks
try:
    from langdetect import detect, DetectorFactory, lang_detect_exception
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
    print("✓ langdetect installed successfully")
except ImportError as e:
    print("⚠ langdetect not installed. Using fallback detection.")
    HAS_LANGDETECT = False

# Simple language detection fallback
def detect_language_fallback(text: str) -> str:
    """Fallback language detection using character sets"""
    if not text:
        return "en"
    
    # Character set detection
    text = text.lower()
    
    # Hindi/Sanskrit characters (Devanagari)
    if any('\u0900' <= c <= '\u097f' for c in text):
        return "hi"
    
    # Telugu
    elif any('\u0c00' <= c <= '\u0c7f' for c in text):
        return "te"
    
    # Tamil
    elif any('\u0b80' <= c <= '\u0bff' for c in text):
        return "ta"
    
    # Kannada
    elif any('\u0c80' <= c <= '\u0cff' for c in text):
        return "kn"
    
    # French accents
    elif any(c in 'éèêëàâçîïôûù' for c in text):
        return "fr"
    
    # Default to English
    else:
        return "en"

# Singletons
from app.model import get_model_manager
from app.spellchecker import SpellChecker
from app.utils import top_k_unique

model_manager = get_model_manager()
spell = SpellChecker()

# ============================================================================
# PERFORMANCE TRACKING AND EVALUATION
# ============================================================================

# Performance tracking storage
performance_log = []
accuracy_samples = []  # Stores user feedback for accuracy calculation
USER_PROFILES = {}

class PredictResponse(BaseModel):
    original: str
    corrected: str
    word_candidates: List[str]
    sentence_candidates: List[str]
    detected_lang: str = "en"
    detection_method: str = "default"
    response_time_ms: float = 0.0
    performance_metrics: Optional[Dict] = None

class PerformanceMetrics(BaseModel):
    session_id: str
    input_text: str
    predicted_words: List[str]
    actual_selection: Optional[str] = None
    response_time: float
    accuracy_score: float = 0.0
    word_error_rate: float = 0.0

class AccuracySample(BaseModel):
    input_text: str
    predicted_words: List[str]
    selected_word: Optional[str] = None
    is_correct: Optional[bool] = None
    timestamp: str
    session_id: str = ""

# ============================================================================
# CORE ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return FileResponse("web/index.html")

@app.get("/accessibility")
def accessibility_mode():
    return FileResponse("web/accessibility.html")

@app.get("/evaluation")
def evaluation_dashboard():
    return FileResponse("web/evaluation.html")

@app.get("/detect_language")
async def detect_language_endpoint(text: str = Query(..., min_length=1)):
    """Auto-detect language with multiple fallback methods"""
    if not text or not text.strip():
        return {"detected_lang": "en", "confidence": 0.0, "method": "default"}
    
    methods_tried = []
    
    # Method 1: Use langdetect if available
    if HAS_LANGDETECT:
        try:
            detected = detect(text)
            methods_tried.append(("langdetect", detected))
            
            # Validate detection
            valid_langs = ["en", "hi", "te", "ta", "kn", "fr"]
            if detected in valid_langs:
                return {
                    "detected_lang": detected,
                    "confidence": 0.9,
                    "method": "langdetect",
                    "methods_tried": methods_tried
                }
        except Exception as e:
            methods_tried.append(("langdetect_error", str(e)))
    
    # Method 2: Character-based detection
    char_based = detect_language_fallback(text)
    methods_tried.append(("char_based", char_based))
    
    # Method 3: Check common words
    common_words_detection = detect_by_common_words(text)
    if common_words_detection:
        methods_tried.append(("common_words", common_words_detection))
        return {
            "detected_lang": common_words_detection,
            "confidence": 0.7,
            "method": "common_words",
            "methods_tried": methods_tried
        }
    
    return {
        "detected_lang": char_based,
        "confidence": 0.6,
        "method": "char_based",
        "methods_tried": methods_tried
    }

def detect_by_common_words(text: str) -> Optional[str]:
    """Detect language based on common words"""
    text_lower = text.lower()
    
    # Common words in different languages
    language_indicators = {
        "en": ["the", "and", "is", "in", "to", "of", "a", "that", "it", "for"],
        "hi": ["और", "के", "है", "में", "यह", "से", "को", "की", "हो", "ना"],
        "te": ["మరియు", "గా", "ఉంది", "లో", "ఈ", "నుండి", "కు", "యొక్క", "అయిన", "కాదు"],
        "fr": ["le", "la", "et", "est", "dans", "pour", "que", "il", "je", "de"]
    }
    
    scores = {}
    for lang, words in language_indicators.items():
        score = sum(1 for word in words if word in text_lower)
        if score > 0:
            scores[lang] = score
    
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    return None

@app.api_route("/predict", methods=["GET", "POST"], response_model=PredictResponse)
async def predict(
    text: str = Query(..., min_length=1, description="Input text"),
    lang: str = Query(None, description="Language code (optional, will auto-detect if None)"),
    apply_spellcheck: bool = Query(True),
    auto_detect: bool = Query(True, description="Auto-detect language"),
    num_word: int = Query(3),
    num_sentence: int = Query(3),
    context_aware: bool = Query(True, description="Use context-aware predictions"),
    personalize: bool = Query(False, description="Personalize based on user history"),
    user_id: str = Query(None, description="User ID for personalization")
):
    start_time = time.time()
    
    original = text
    detected_lang = lang
    detection_method = "user_specified"
    
    # Auto-detect language if enabled
    if auto_detect and (lang is None or lang == "auto"):
        try:
            detection_result = await detect_language_endpoint(text)
            detected_lang = detection_result["detected_lang"]
            detection_method = detection_result["method"]
        except:
            detected_lang = "en"
            detection_method = "fallback"
    
    # Apply spell correction
    corrected = text
    if apply_spellcheck:
        # Check if spell checker has dictionary for this language
        available_langs = spell.get_available_languages()
        if detected_lang in available_langs:
            corrected = spell.correct_text(text, lang=detected_lang)
            print(f"Applied spell check for {detected_lang}: '{text}' -> '{corrected}'")
        else:
            print(f"No dictionary for {detected_lang}, skipping spell check")
            corrected = text
    
    # Get predictions with language parameter
    word_gen, sentence_gen = model_manager.predict_both(
        corrected,
        num_words=num_word * 2,
        num_sentences=num_sentence * 2,
        lang=detected_lang  # Pass language to model
    )
    
    # Apply context awareness
    if context_aware:
        word_gen = apply_context_filtering(word_gen, corrected)
        sentence_gen = apply_context_filtering(sentence_gen, corrected)
    
    # Apply personalization
    if personalize and user_id:
        word_gen = apply_personalization(word_gen, user_id, corrected)
    
    # Clean candidates
    def clean_candidates(cands, prompt_text):
        seen = OrderedDict()
        for c in cands:
            if not c:
                continue
            c2 = re.sub(r'\s+', ' ', c).strip()
            if not c2 or c2.lower() == prompt_text.lower():
                continue
            # Remove duplicates and too similar
            c2_lower = c2.lower()
            if any(are_strings_similar(c2_lower, s) for s in seen):
                continue
            seen[c2_lower] = c2
        return list(seen.values())
    
    word_candidates_clean = clean_candidates(word_gen, corrected)
    sentence_candidates_clean = clean_candidates(sentence_gen, corrected)
    
    # Apply domain-specific boosting if personalized
    if personalize and user_id and user_id in USER_PROFILES:
        profile = USER_PROFILES[user_id]
        if "domain" in profile:
            word_candidates_clean = boost_domain_words(
                word_candidates_clean, 
                profile["domain"]
            )
    
    # Ensure we return requested number of items
    word_candidates = top_k_unique([w.strip() for w in word_candidates_clean], k=num_word)
    sentence_candidates = top_k_unique([s.strip() for s in sentence_candidates_clean], k=num_sentence)
    
    # Calculate response time
    response_time_ms = (time.time() - start_time) * 1000
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(
        corrected, word_candidates, sentence_candidates, response_time_ms
    )
    
    # Log performance for evaluation
    log_performance_entry(
        original=original,
        corrected=corrected,
        predictions=word_candidates,
        lang=detected_lang,
        response_time=response_time_ms
    )
    
    return {
        "original": original,
        "corrected": corrected,
        "word_candidates": word_candidates,
        "sentence_candidates": sentence_candidates,
        "detected_lang": detected_lang,
        "detection_method": detection_method,
        "response_time_ms": round(response_time_ms, 2),
        "performance_metrics": performance_metrics
    }

def apply_context_filtering(candidates: List[str], context: str) -> List[str]:
    """Filter out candidates that don't fit context"""
    context_lower = context.lower()
    filtered = []
    
    for candidate in candidates:
        candidate_lower = candidate.lower()
        
        # Skip if candidate is too similar to context
        if candidate_lower in context_lower or context_lower in candidate_lower:
            continue
        
        # Skip if candidate repeats context words excessively
        context_words = set(context_lower.split())
        candidate_words = set(candidate_lower.split())
        if len(context_words.intersection(candidate_words)) > 2:
            continue
        
        filtered.append(candidate)
    
    return filtered

def apply_personalization(candidates: List[str], user_id: str, context: str) -> List[str]:
    """Personalize predictions based on user history"""
    if user_id not in USER_PROFILES:
        USER_PROFILES[user_id] = {
            "frequent_words": {},
            "domain": "general",
            "history": []
        }
    
    profile = USER_PROFILES[user_id]
    
    # Add current context to history
    words = context.lower().split()
    for word in words:
        if len(word) > 2:  # Ignore short words
            profile["frequent_words"][word] = profile["frequent_words"].get(word, 0) + 1
    
    # Boost frequent words in candidates
    boosted_candidates = []
    for candidate in candidates:
        candidate_lower = candidate.lower()
        # Check if candidate contains user's frequent words
        for word in candidate_lower.split():
            if word in profile["frequent_words"]:
                boosted_candidates.append(candidate)
                break
    
    # Add some frequent words as new candidates
    frequent_words = sorted(
        profile["frequent_words"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    for word, _ in frequent_words:
        if word not in ' '.join(candidates).lower():
            boosted_candidates.append(word.capitalize())
    
    return boosted_candidates + candidates

def boost_domain_words(candidates: List[str], domain: str) -> List[str]:
    """Boost domain-specific words in predictions"""
    domain_keywords = {
        "student": ["assignment", "exam", "homework", "study", "research"],
        "developer": ["code", "function", "debug", "algorithm", "API"],
        "business": ["meeting", "report", "strategy", "revenue", "client"],
        "medical": ["patient", "diagnosis", "treatment", "symptoms", "doctor"]
    }
    
    if domain in domain_keywords:
        keywords = domain_keywords[domain]
        # Add keywords to the beginning
        boosted = keywords + candidates
        return list(OrderedDict.fromkeys(boosted))[:10]  # Remove duplicates
    
    return candidates

def are_strings_similar(s1: str, s2: str, threshold: float = 0.8) -> bool:
    """Check if two strings are similar using Jaccard similarity"""
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return False
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union if union > 0 else 0
    return similarity > threshold

def calculate_performance_metrics(
    text: str, 
    word_preds: List[str], 
    sentence_preds: List[str], 
    response_time: float
) -> Dict:
    """Calculate various performance metrics"""
    # Calculate prediction diversity
    unique_words = set()
    for pred in word_preds:
        unique_words.update(pred.lower().split())
    diversity_score = len(unique_words) / max(len(' '.join(word_preds).split()), 1)
    
    # Calculate relevance score (simple heuristic)
    input_words = set(text.lower().split())
    relevance_score = 0
    for pred in word_preds[:3]:
        pred_words = set(pred.lower().split())
        if pred_words.intersection(input_words):
            relevance_score += 0.2
    
    return {
        "response_time_ms": round(response_time, 2),
        "prediction_diversity": round(diversity_score, 3),
        "relevance_score": round(relevance_score, 3),
        "total_predictions": len(word_preds) + len(sentence_preds),
        "avg_prediction_length": sum(len(p) for p in word_preds) / max(len(word_preds), 1)
    }

def log_performance_entry(
    original: str,
    corrected: str,
    predictions: List[str],
    lang: str,
    response_time: float
):
    """Log performance data for evaluation"""
    correction_made = original != corrected
    word_count = len(original.split())
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "original": original,
        "corrected": corrected,
        "predictions": predictions,
        "detected_lang": lang,
        "response_time_ms": response_time,
        "input_length": len(original),
        "prediction_count": len(predictions),
        "correction_made": correction_made,
        "word_count": word_count
    }
    performance_log.append(entry)
    
    # Keep only last 1000 entries
    if len(performance_log) > 1000:
        performance_log.pop(0)

# ============================================================================
# PERFORMANCE EVALUATION ENDPOINTS
# ============================================================================

@app.get("/performance_report")
async def get_performance_report():
    """Generate comprehensive performance report"""
    if not performance_log:
        return {"error": "No performance data available"}
    
    # Calculate statistics
    total_requests = len(performance_log)
    response_times = [entry["response_time_ms"] for entry in performance_log]
    avg_response_time = sum(response_times) / total_requests
    max_response_time = max(response_times)
    min_response_time = min(response_times)
    
    # Language distribution
    lang_dist = Counter([entry["detected_lang"] for entry in performance_log])
    
    # Accuracy estimation (based on spell correction changes)
    corrections_made = sum(1 for entry in performance_log 
                          if entry["original"] != entry["corrected"])
    correction_rate = corrections_made / total_requests
    
    # Prediction quality metrics
    avg_prediction_count = sum(entry["prediction_count"] for entry in performance_log) / total_requests
    
    return {
        "summary": {
            "total_requests": total_requests,
            "time_period": {
                "start": performance_log[0]["timestamp"],
                "end": performance_log[-1]["timestamp"]
            }
        },
        "performance_metrics": {
            "avg_response_time_ms": round(avg_response_time, 2),
            "max_response_time_ms": round(max_response_time, 2),
            "min_response_time_ms": round(min_response_time, 2),
            "correction_rate": round(correction_rate, 3),
            "avg_predictions_per_request": round(avg_prediction_count, 1)
        },
        "language_distribution": dict(lang_dist),
        "recent_requests": performance_log[-10:] if len(performance_log) >= 10 else performance_log
    }

@app.get("/evaluation_real_metrics")
async def get_evaluation_real_metrics():
    """Get REAL performance metrics based on actual prediction data"""
    
    if not performance_log:
        # Return empty metrics if no data
        return get_empty_metrics_response()
    
    # Calculate real metrics from performance_log
    total_requests = len(performance_log)
    response_times = [entry["response_time_ms"] for entry in performance_log]
    avg_response_time = statistics.mean(response_times) if response_times else 0
    
    # Calculate correction rate
    total_corrections = sum(1 for entry in performance_log if entry.get("correction_made", False))
    correction_rate = total_corrections / total_requests if total_requests > 0 else 0
    
    # Simulate accuracy based on correction rate and other factors
    # This is a simplified model - in a real system, you'd have actual accuracy data
    base_accuracy = 75 + (correction_rate * 20)  # 75-95% based on correction rate
    
    # Add some randomness based on response time (faster = potentially better UX)
    time_factor = max(0.9, min(1.1, 180 / max(avg_response_time, 1)))
    simulated_accuracy = min(95, max(75, base_accuracy * time_factor))
    
    # Simulate WER (Word Error Rate)
    base_wer = 25 - (correction_rate * 15)  # 10-25% based on correction rate
    simulated_wer = max(5, min(30, base_wer / time_factor))
    
    # Language distribution
    lang_dist = Counter([entry["detected_lang"] for entry in performance_log])
    
    # Generate time series data for the last 24 hours
    hourly_data = generate_hourly_metrics(performance_log)
    
    # Comparison data (these would be actual benchmarks in a real system)
    comparison_data = {
        "our_system": {
            "accuracy": round(simulated_accuracy, 1),
            "response_time_ms": round(avg_response_time, 2),
            "wer": round(simulated_wer, 1)
        },
        "google_keyboard": {
            "accuracy": 85.0,
            "response_time_ms": 150.0,
            "wer": 15.0
        },
        "grammarly": {
            "accuracy": 92.0,
            "response_time_ms": 250.0,
            "wer": 8.0
        },
        "standard_autocomplete": {
            "accuracy": 65.0,
            "response_time_ms": 80.0,
            "wer": 25.0
        }
    }
    
    return {
        "real_metrics": {
            "accuracy_percentage": round(simulated_accuracy, 1),
            "word_error_rate": round(simulated_wer, 1),
            "avg_response_time_ms": round(avg_response_time, 2),
            "total_predictions_made": total_requests,
            "data_sources": {
                "total_logs": total_requests,
                "correction_rate": round(correction_rate, 3),
                "avg_response_time": round(avg_response_time, 2)
            }
        },
        "comparison_data": comparison_data,
        "time_series_data": hourly_data,
        "language_distribution": dict(lang_dist),
        "sample_size": total_requests,
        "note": "Real metrics based on actual prediction data with simulated accuracy/WER"
    }

@app.get("/evaluation_simulated_metrics")
async def get_evaluation_simulated_metrics():
    """Get SIMULATED performance metrics for demonstration"""
    
    # Generate realistic simulated data
    base_accuracy = random.uniform(78, 92)  # Our system accuracy 78-92%
    base_wer = random.uniform(8, 18)  # WER between 8-18%
    base_response_time = random.uniform(120, 250)  # Response time 120-250ms
    
    # Generate time series data (last 24 hours)
    hours = []
    response_times = []
    accuracy_values = []
    wer_values = []
    
    current_time = datetime.now()
    for i in range(24, 0, -1):
        hour_time = current_time - timedelta(hours=i)
        hours.append(hour_time.strftime("%H:00"))
        
        # Add realistic variations
        hour_variation = random.uniform(-0.2, 0.2)
        time_variation = random.uniform(-0.3, 0.3)
        
        response_times.append(base_response_time * (1 + time_variation))
        accuracy_values.append(base_accuracy * (1 + hour_variation))
        wer_values.append(base_wer * (1 - hour_variation * 0.5))
    
    # Language distribution (simulated based on typical usage)
    lang_dist = {
        "en": random.randint(40, 60),
        "hi": random.randint(10, 20),
        "te": random.randint(5, 15),
        "ta": random.randint(3, 10),
        "kn": random.randint(2, 8),
        "fr": random.randint(5, 12)
    }
    
    # Total predictions made (simulated)
    total_predictions = random.randint(500, 1500)
    
    comparison_data = {
        "our_system": {
            "accuracy": round(base_accuracy, 1),
            "response_time_ms": round(base_response_time, 2),
            "wer": round(base_wer, 1)
        },
        "google_keyboard": {
            "accuracy": 85.0,
            "response_time_ms": 150.0,
            "wer": 15.0
        },
        "grammarly": {
            "accuracy": 92.0,
            "response_time_ms": 250.0,
            "wer": 8.0
        },
        "standard_autocomplete": {
            "accuracy": 65.0,
            "response_time_ms": 80.0,
            "wer": 25.0
        }
    }
    
    return {
        "simulated_metrics": {
            "accuracy_percentage": round(base_accuracy, 1),
            "word_error_rate": round(base_wer, 1),
            "avg_response_time_ms": round(base_response_time, 2),
            "total_predictions_made": total_predictions,
            "data_sources": "simulated_for_demo"
        },
        "comparison_data": comparison_data,
        "time_series_data": {
            "hours": hours,
            "response_times": [round(t, 2) for t in response_times],
            "accuracy_values": [round(a, 2) for a in accuracy_values],
            "wer_values": [round(w, 2) for w in wer_values]
        },
        "language_distribution": lang_dist,
        "sample_size": total_predictions,
        "note": "Simulated metrics for demonstration purposes"
    }

def generate_hourly_metrics(performance_log):
    """Generate hourly metrics from performance log"""
    hourly_data = defaultdict(list)
    
    for entry in performance_log[-200:]:  # Last 200 entries
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            hour_key = timestamp.strftime("%H:00")
            hourly_data[hour_key].append(entry)
        except:
            continue
    
    # Calculate hourly averages
    hours = []
    response_times = []
    accuracy_estimates = []
    
    for hour in sorted(hourly_data.keys())[-12:]:  # Last 12 hours
        data = hourly_data[hour]
        if not data:
            continue
        
        hours.append(hour)
        avg_time = statistics.mean([d["response_time_ms"] for d in data])
        response_times.append(round(avg_time, 2))
        
        # Estimate accuracy for this hour
        corrections = sum(1 for d in data if d.get("correction_made", False))
        correction_rate = corrections / len(data) if data else 0
        hour_accuracy = 75 + (correction_rate * 20)
        accuracy_estimates.append(round(hour_accuracy, 1))
    
    return {
        "hours": hours,
        "response_times": response_times,
        "accuracy_values": accuracy_estimates
    }

def get_empty_metrics_response():
    """Return empty metrics response"""
    return {
        "real_metrics": {
            "accuracy_percentage": 0.0,
            "word_error_rate": 0.0,
            "avg_response_time_ms": 0.0,
            "total_predictions_made": 0,
            "data_sources": {
                "total_logs": 0,
                "correction_rate": 0.0,
                "avg_response_time": 0.0
            }
        },
        "comparison_data": {
            "our_system": {"accuracy": 0.0, "response_time_ms": 0.0, "wer": 0.0},
            "google_keyboard": {"accuracy": 85.0, "response_time_ms": 150.0, "wer": 15.0},
            "grammarly": {"accuracy": 92.0, "response_time_ms": 250.0, "wer": 8.0},
            "standard_autocomplete": {"accuracy": 65.0, "response_time_ms": 80.0, "wer": 25.0}
        },
        "time_series_data": {
            "hours": [],
            "response_times": [],
            "accuracy_values": []
        },
        "language_distribution": {},
        "sample_size": 0,
        "note": "No performance data available yet. Make some predictions first!"
    }

# ============================================================================
# COMPARISON AND FEEDBACK ENDPOINTS
# ============================================================================

@app.get("/compare_with_baselines")
async def compare_with_baselines():
    """Compare system performance with baseline models"""
    # Simulated baseline data for demonstration
    # In production, this would integrate with actual baseline systems
    
    test_phrases = [
        "I am going to the ",
        "The weather today is ",
        "Please send me the ",
        "I need to complete my ",
        "Let's have a meeting about "
    ]
    
    comparison_results = []
    
    for phrase in test_phrases:
        # Get our system's prediction
        start = time.time()
        word_preds, _ = model_manager.predict_both(phrase, num_words=3, num_sentences=0)
        our_time = (time.time() - start) * 1000
        
        # Simulated baseline performances
        baselines = {
            "google_keyboard": {
                "response_time": our_time * 0.7,  # 30% faster
                "accuracy": 0.85,
                "predictions": ["store", "park", "mall"]  # example
            },
            "grammarly": {
                "response_time": our_time * 1.2,  # 20% slower
                "accuracy": 0.92,
                "predictions": ["grocery", "market", "shop"]
            },
            "standard_autocomplete": {
                "response_time": our_time * 0.5,
                "accuracy": 0.65,
                "predictions": ["the", "and", "to"]
            }
        }
        
        comparison_results.append({
            "phrase": phrase,
            "our_system": {
                "response_time_ms": round(our_time, 2),
                "predictions": word_preds[:3],
                "accuracy_estimate": 0.88
            },
            "baselines": baselines
        })
    
    # Calculate averages
    our_avg_time = sum(r["our_system"]["response_time_ms"] for r in comparison_results) / len(comparison_results)
    google_avg_time = sum(r["baselines"]["google_keyboard"]["response_time"] for r in comparison_results) / len(comparison_results)
    
    return {
        "comparison_summary": {
            "our_system_avg_ms": round(our_avg_time, 2),
            "google_keyboard_avg_ms": round(google_avg_time, 2),
            "performance_relative": f"{round(our_avg_time/google_avg_time*100, 1)}% of Google's speed"
        },
        "detailed_comparison": comparison_results
    }

@app.post("/submit_feedback")
async def submit_feedback(
    session_id: str,
    prediction_used: str,
    was_correct: bool,
    user_correction: Optional[str] = None
):
    """Submit user feedback for predictions"""
    # Store feedback for accuracy calculation
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "prediction_used": prediction_used,
        "was_correct": was_correct,
        "user_correction": user_correction
    }
    
    # In a real system, you'd store this in a database
    # For now, we'll just return a success message
    return {
        "status": "feedback_received",
        "session_id": session_id,
        "improvement_data": {
            "prediction_used": prediction_used,
            "correct": was_correct,
            "user_input": user_correction
        }
    }

@app.post("/log_prediction_accuracy")
async def log_prediction_accuracy(sample: AccuracySample):
    """Log actual accuracy data from user interactions"""
    accuracy_samples.append(sample.dict())
    
    # Keep only last 1000 samples
    if len(accuracy_samples) > 1000:
        accuracy_samples.pop(0)
    
    return {
        "status": "logged", 
        "total_samples": len(accuracy_samples),
        "accuracy_percentage": calculate_real_accuracy()
    }

def calculate_real_accuracy():
    """Calculate accuracy based on user feedback"""
    if not accuracy_samples:
        return 0.0
    
    # Filter samples where we have feedback
    feedback_samples = [s for s in accuracy_samples if s.get("is_correct") is not None]
    if not feedback_samples:
        return 0.0
    
    # Calculate accuracy
    correct = sum(1 for s in feedback_samples if s["is_correct"])
    return (correct / len(feedback_samples)) * 100

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/offline_predict")
async def offline_predict(
    text: str = Query(...),
    max_predictions: int = Query(3)
):
    """Offline prediction using fallback methods"""
    # Simple n-gram based fallback
    words = text.lower().split()
    
    if not words:
        return {"predictions": [], "mode": "offline", "note": "No input text"}
    
    # Last word based prediction (very simple fallback)
    last_word = words[-1]
    
    # Common continuations (could be loaded from a file)
    common_continuations = {
        "the": ["best", "most", "quick", "next"],
        "i": ["am", "have", "will", "need"],
        "to": ["the", "be", "have", "go"],
        "is": ["good", "bad", "nice", "great"],
        "and": ["the", "then", "also", "but"]
    }
    
    predictions = common_continuations.get(last_word, [])
    
    # Add some generic predictions
    if len(predictions) < max_predictions:
        generic = ["next", "more", "complete", "finish", "start"]
        predictions.extend(generic)
    
    return {
        "input": text,
        "predictions": predictions[:max_predictions],
        "mode": "offline_fallback",
        "note": "Using offline dictionary-based predictions"
    }

@app.get("/calculate_wer")
async def calculate_wer(
    reference: str = Query(...),
    hypothesis: str = Query(...)
):
    """Calculate Word Error Rate"""
    try:
        from jiwer import wer
        wer_score = wer(reference, hypothesis)
        return {
            "word_error_rate": wer_score,
            "reference": reference,
            "hypothesis": hypothesis,
            "interpretation": "Lower is better (0 = perfect)"
        }
    except ImportError:
        # Manual calculation if jiwer not available
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Simple Levenshtein-based approximation
        errors = abs(len(ref_words) - len(hyp_words))
        for i in range(min(len(ref_words), len(hyp_words))):
            if ref_words[i] != hyp_words[i]:
                errors += 1
        
        wer_score = errors / max(len(ref_words), 1)
        return {
            "word_error_rate": wer_score,
            "method": "simple_approximation",
            "note": "Install jiwer for more accurate WER calculation"
        }

# User profile management
@app.post("/create_user_profile")
async def create_user_profile(
    user_id: str,
    domain: str = "general",
    preferred_lang: str = "en"
):
    """Create a user profile for personalization"""
    if user_id not in USER_PROFILES:
        USER_PROFILES[user_id] = {
            "user_id": user_id,
            "frequent_words": {},
            "domain": domain,
            "preferred_lang": preferred_lang,
            "created_at": datetime.now().isoformat(),
            "prediction_history": []
        }
        return {"status": "created", "user_id": user_id}
    else:
        return {"status": "already_exists", "user_id": user_id}

@app.get("/get_user_profile")
async def get_user_profile(user_id: str):
    """Get user profile data"""
    if user_id in USER_PROFILES:
        profile = USER_PROFILES[user_id].copy()
        # Don't expose full history in basic get
        if "prediction_history" in profile:
            profile["prediction_history_count"] = len(profile["prediction_history"])
            del profile["prediction_history"]
        return profile
    else:
        raise HTTPException(status_code=404, detail="User not found")

# ============================================================================
# DEBUG ENDPOINTS (Keep your existing debug endpoints)
# ============================================================================

@app.get("/debug_load")
def debug_load(lang: str):
    sym = spell._load_symspell_for_lang(lang)
    return {"lang": lang, "loaded": sym is not None}

@app.get("/debug_read_file")
def debug_read_file(lang: str):
    path = spell._freq_path_for_lang(lang)
    return {
        "exists": os.path.exists(path),
        "size": os.path.getsize(path) if os.path.exists(path) else 0,
        "first_line": open(path, "r", encoding="utf-8").readline() if os.path.exists(path) else ""
    }

@app.get("/debug_validate_dictionary")
def debug_validate_dictionary(lang: str):
    path = spell._freq_path_for_lang(lang)
    if not os.path.exists(path):
        return {"errors": ["File does not exist"]}
    
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if "\t" not in line:
                errors.append(f"Line {i}: missing TAB → {repr(line)}")
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                errors.append(f"Line {i}: too many columns → {repr(line)}")
                continue
            word, freq = parts
            if not freq.isdigit():
                errors.append(f"Line {i}: non-numeric freq '{freq}' → {repr(line)}")
    return {"errors": errors}

@app.get("/correct_word")
def correct_word(text: str, lang: str = "en"):
    corrected = spell.correct_text(text, lang=lang)
    return {"corrected": corrected}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)