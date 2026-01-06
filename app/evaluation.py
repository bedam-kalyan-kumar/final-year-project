# app/evaluation.py
import time
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
import statistics

@dataclass
class TestCase:
    input_text: str
    expected_next_words: List[str]
    expected_sentences: List[str]
    language: str
    difficulty: str  # easy, medium, hard

class Evaluator:
    def __init__(self):
        self.test_cases = self.load_test_cases()
        self.results = []
    
    def load_test_cases(self) -> List[TestCase]:
        """Load test cases for evaluation"""
        test_cases = [
            TestCase(
                input_text="I am going to the",
                expected_next_words=["store", "market", "park"],
                expected_sentences=["store to buy groceries.", "park for a walk."],
                language="en",
                difficulty="easy"
            ),
            TestCase(
                input_text="The quick brown fox jumps over the",
                expected_next_words=["lazy", "sleeping", "fence"],
                expected_sentences=["lazy dog.", "fence and runs away."],
                language="en",
                difficulty="medium"
            ),
            # Add more test cases...
        ]
        return test_cases
    
    def evaluate_prediction_accuracy(self, predictions: List[str], expected: List[str]) -> float:
        """Calculate accuracy of predictions"""
        if not expected:
            return 0.0
        
        matches = sum(1 for pred in predictions[:3] if pred in expected)
        return matches / min(3, len(expected))
    
    def run_evaluation(self, model_manager) -> Dict:
        """Run comprehensive evaluation"""
        metrics = {
            "accuracy_scores": [],
            "response_times": [],
            "language_performance": defaultdict(list)
        }
        
        for test_case in self.test_cases:
            start_time = time.time()
            
            # Get predictions
            word_preds, sentence_preds = model_manager.predict_both(
                test_case.input_text,
                num_words=3,
                num_sentences=2
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Calculate accuracy
            word_accuracy = self.evaluate_prediction_accuracy(
                word_preds, test_case.expected_next_words
            )
            
            # Store metrics
            metrics["accuracy_scores"].append(word_accuracy)
            metrics["response_times"].append(response_time)
            metrics["language_performance"][test_case.language].append(word_accuracy)
        
        # Calculate summary statistics
        metrics["avg_accuracy"] = statistics.mean(metrics["accuracy_scores"]) * 100
        metrics["avg_response_time"] = statistics.mean(metrics["response_times"])
        metrics["accuracy_by_difficulty"] = self.calculate_by_difficulty()
        
        return metrics
    
    def calculate_by_difficulty(self) -> Dict:
        """Calculate accuracy by difficulty level"""
        difficulty_scores = defaultdict(list)
        
        for test_case, accuracy in zip(self.test_cases, self.results):
            difficulty_scores[test_case.difficulty].append(accuracy)
        
        return {
            diff: statistics.mean(scores) * 100 
            for diff, scores in difficulty_scores.items()
        }
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        metrics = self.run_evaluation()
        
        report = {
            "evaluation_summary": {
                "total_test_cases": len(self.test_cases),
                "average_accuracy": f"{metrics['avg_accuracy']:.2f}%",
                "average_response_time": f"{metrics['avg_response_time']:.2f}ms",
                "performance_by_language": {
                    lang: f"{statistics.mean(scores)*100:.2f}%"
                    for lang, scores in metrics["language_performance"].items()
                }
            },
            "comparison_with_baselines": {
                "our_system": metrics["avg_accuracy"],
                "google_keyboard": 85.3,  # Example baseline
                "grammarly": 89.7,  # Example baseline
                "standard_keyboard": 72.1
            },
            "recommendations": [
                "Increase training data for low-frequency words",
                "Optimize response time for mobile devices",
                "Add more context-aware features"
            ]
        }
        
        return report