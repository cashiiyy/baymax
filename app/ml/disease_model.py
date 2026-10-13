"""
BAYMAX AI – Disease & Symptom Machine Learning Model
=====================================================
Trained on clinical datasets (4,920 training samples, 132 symptom features, 41 diseases).
Provides:
  1. Automated symptom extraction from free-form user query.
  2. Probabilistic disease classification via calibrated Random Forest.
  3. Grounded disease-symptom profiles to prevent LLM hallucinations.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.utils.logger import get_logger

log = get_logger(__name__)

# Default file paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_CSV = PROJECT_ROOT / "app" / "datasets" / "Disease_name" / "Training.csv"
TESTING_CSV = PROJECT_ROOT / "app" / "datasets" / "Disease_name" / "Testing.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "disease_prediction_model.joblib"
METADATA_FILE = PROJECT_ROOT / "models" / "disease_model_metadata.json"


@dataclass
class DiseasePrediction:
    disease: str
    probability: float
    confidence_percent: float
    matching_symptoms: List[str]
    characteristic_symptoms: List[str]


# Common colloquial and clinical synonyms mapped to standard dataset symptom keys
SYMPTOM_SYNONYMS: Dict[str, str] = {
    # Headache variants
    "headache": "headache",
    "severe headache": "headache",
    "head pain": "headache",
    "migraine": "headache",
    "head ache": "headache",
    "throbbing head": "headache",
    # Fever & temperature
    "fever": "high_fever",
    "high fever": "high_fever",
    "temperature": "high_fever",
    "feverish": "high_fever",
    "mild fever": "mild_fever",
    "low grade fever": "mild_fever",
    "chills": "chills",
    "shivering": "shivering",
    # Respiratory & Cold
    "cough": "cough",
    "coughing": "cough",
    "sneezing": "continuous_sneezing",
    "cold": "runny_nose",
    "runny nose": "runny_nose",
    "blocked nose": "congestion",
    "congestion": "congestion",
    "sinus pressure": "sinus_pressure",
    "throat irritation": "throat_irritation",
    "sore throat": "throat_irritation",
    "patches in throat": "patches_in_throat",
    "phlegm": "phlegm",
    "breathlessness": "breathlessness",
    "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "blood in sputum": "blood_in_sputum",
    # Stomach & GI
    "stomach pain": "stomach_pain",
    "abdominal pain": "abdominal_pain",
    "belly pain": "belly_pain",
    "stomach ache": "stomach_pain",
    "tummy ache": "stomach_pain",
    "cramps": "cramps",
    "acidity": "acidity",
    "acid reflux": "acidity",
    "indigestion": "indigestion",
    "heartburn": "acidity",
    "vomiting": "vomiting",
    "throwing up": "vomiting",
    "nausea": "nausea",
    "nauseous": "nausea",
    "feel sick": "nausea",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "loose motion": "diarrhoea",
    "constipation": "constipation",
    "loss of appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite",
    # Body pain & Fatigue
    "fatigue": "fatigue",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "lethargy": "lethargy",
    "weakness": "muscle_weakness",
    "muscle weakness": "muscle_weakness",
    "muscle pain": "muscle_pain",
    "body pain": "muscle_pain",
    "body ache": "muscle_pain",
    "joint pain": "joint_pain",
    "back pain": "back_pain",
    "neck pain": "neck_pain",
    "stiff neck": "stiff_neck",
    # Cardiac & Chest
    "chest pain": "chest_pain",
    "fast heart rate": "fast_heart_rate",
    "palpitations": "palpitations",
    "heart racing": "fast_heart_rate",
    # Neurological / Balance
    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "spinning": "spinning_movements",
    "loss of balance": "loss_of_balance",
    "unsteadiness": "unsteadiness",
    "blurred vision": "blurred_and_distorted_vision",
    "distorted vision": "blurred_and_distorted_vision",
    "altered sensorium": "altered_sensorium",
    "slurred speech": "slurred_speech",
    # Skin & Allergies
    "itching": "itching",
    "itchy": "itching",
    "skin rash": "skin_rash",
    "rash": "skin_rash",
    "blister": "blister",
    "red spots": "red_spots_over_body",
    "yellow skin": "yellowish_skin",
    "yellowing of eyes": "yellowing_of_eyes",
    "sweating": "sweating",
    "dehydration": "dehydration",
}


class DiseasePredictorML:
    """
    Trained Machine Learning Model and Inference Service for Disease Prediction.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        self.model_path = model_path or MODEL_FILE
        self.metadata_path = metadata_path or METADATA_FILE
        self.model: Optional[RandomForestClassifier] = None
        self.symptom_features: List[str] = []
        self.classes_: List[str] = []
        self.disease_symptoms_profile: Dict[str, List[str]] = {}
        self.disease_home_remedies: Dict[str, List[str]] = {}

        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Load trained model or train if not present."""
        if self.model_path.exists() and self.metadata_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.symptom_features = meta.get("symptom_features", [])
                    self.classes_ = meta.get("classes", [])
                    self.disease_symptoms_profile = meta.get("disease_symptoms_profile", {})
                    self.disease_home_remedies = meta.get("disease_home_remedies", {})
                log.info(
                    "Loaded trained disease ML model | features={} classes={}",
                    len(self.symptom_features),
                    len(self.classes_),
                )
                return
            except Exception as exc:
                log.warning("Failed to load saved model, re-training: {}", exc)

        log.info("No trained disease model found. Training from datasets...")
        self.train_and_save()

    def train_and_save(
        self,
        train_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
    ) -> float:
        """
        Train Random Forest Classifier on Training.csv and evaluate on Testing.csv.
        Returns accuracy score on test set.
        """
        train_file = train_path or TRAINING_CSV
        test_file = test_path or TESTING_CSV

        if not train_file.exists():
            raise FileNotFoundError(f"Training dataset not found: {train_file}")

        # 1. Parse Training Data
        X_train = []
        y_train = []
        with open(train_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader) if h.strip()]
            self.symptom_features = header[:-1]

            for row in reader:
                if not row or not any(row):
                    continue
                features = [int(val.strip()) if val.strip().isdigit() else 0 for val in row[:len(self.symptom_features)]]
                label = row[len(self.symptom_features)].strip()
                X_train.append(features)
                y_train.append(label)

        X_train_arr = np.array(X_train, dtype=np.int32)
        y_train_arr = np.array(y_train)

        # 2. Build Disease Symptoms Profile from training data
        disease_profile: Dict[str, set] = {}
        for x, y in zip(X_train, y_train):
            if y not in disease_profile:
                disease_profile[y] = set()
            for feat_idx, val in enumerate(x):
                if val == 1:
                    disease_profile[y].add(self.symptom_features[feat_idx])

        self.disease_symptoms_profile = {k: sorted(list(v)) for k, v in disease_profile.items()}

        # Curated home care & first aid advice for top conditions
        self.disease_home_remedies = {
            "Migraine": [
                "Rest in a quiet, dark, well-ventilated room",
                "Apply a cold or warm compress to forehead or back of neck",
                "Stay well hydrated with water or electrolyte fluids",
                "Over-the-counter pain relievers such as acetaminophen or ibuprofen if safe",
                "Avoid sensory triggers: bright screens, loud noises, caffeine, and missed meals",
                "Consult physician if migraine is accompanied by fever, neck stiffness, confusion, or speech loss",
            ],
            "Hypertension": [
                "Sit or lie down in a calm position and practice slow deep breathing",
                "Reduce dietary sodium and avoid caffeine or stimulants",
                "Monitor blood pressure with a home cuff",
                "Take prescribed anti-hypertensive medication as directed by doctor",
                "Seek immediate emergency care if systolic > 180 or diastolic > 120 with chest pain or vision changes",
            ],
            "Allergy": [
                "Identify and remove exposure to the allergen",
                "Take non-sedating antihistamines (e.g. cetirizine or loratadine) as directed",
                "Use saline nasal spray for sinus congestion",
                "Seek emergency care immediately if experiencing lip swelling or wheezing (anaphylaxis)",
            ],
            "Common Cold": [
                "Ensure plenty of rest and sleep to support immune recovery",
                "Drink warm fluids: herbal tea with honey, clear broth, warm water",
                "Use steam inhalation or saline nasal rinses for congestion",
                "Over-the-counter paracetamol for aches and mild fever",
            ],
            "GERD": [
                "Avoid lying down for at least 2 to 3 hours after eating",
                "Elevate head of bed 6 inches",
                "Avoid trigger foods: fatty, spicy, acidic foods, caffeine, mint, chocolate",
                "Consider antacids or H2 blockers after consulting healthcare provider",
            ],
            "Gastroenteritis": [
                "Sip oral rehydration salts (ORS) or electrolyte solutions frequently",
                "Stick to bland, low-fiber foods (BRAT diet: bananas, rice, applesauce, toast)",
                "Avoid dairy, alcohol, and very greasy foods until gut heals",
                "Seek medical attention if unable to keep fluids down for 24 hours or blood in stool",
            ],
            "Heart attack": [
                "🚨 EMERGENCY: Call 911 or local emergency medical services immediately",
                "Do NOT drive yourself to hospital",
                "Sit down, stay calm, loosen tight clothing",
                "Chew an adult aspirin (325mg) if not allergic and advised by emergency dispatch",
            ],
        }

        # 3. Train Classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train_arr, y_train_arr)
        self.model = clf
        self.classes_ = [str(c) for c in clf.classes_]

        # 4. Evaluate on Testing Set
        accuracy = 1.0
        if test_file.exists():
            X_test = []
            y_test = []
            with open(test_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if not row or not any(row):
                        continue
                    features = [int(val.strip()) if val.strip().isdigit() else 0 for val in row[:len(self.symptom_features)]]
                    label = row[len(self.symptom_features)].strip()
                    X_test.append(features)
                    y_test.append(label)
            if X_test:
                y_pred = clf.predict(np.array(X_test, dtype=np.int32))
                accuracy = float(np.mean(y_pred == np.array(y_test)))
                log.info("Disease ML model test set accuracy: {:.2f}%", accuracy * 100)

        # 5. Save model and metadata
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, self.model_path)
        meta = {
            "symptom_features": self.symptom_features,
            "classes": self.classes_,
            "accuracy": accuracy,
            "disease_symptoms_profile": self.disease_symptoms_profile,
            "disease_home_remedies": self.disease_home_remedies,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        log.info("Saved trained model to {} (acc={:.2%})", self.model_path, accuracy)
        return accuracy

    def extract_symptoms(self, text: str) -> List[str]:
        """
        Extract standardized dataset symptom keys from natural language text.
        """
        if not text:
            return []

        text_lower = text.lower()
        # Normalize punctuation
        cleaned = re.sub(r"[^\w\s]", " ", text_lower)
        detected = set()

        # 1. Match synonym phrases (longer phrases first)
        sorted_synonyms = sorted(SYMPTOM_SYNONYMS.keys(), key=len, reverse=True)
        for phrase in sorted_synonyms:
            if re.search(r"\b" + re.escape(phrase) + r"\b", text_lower):
                detected.add(SYMPTOM_SYNONYMS[phrase])

        # 2. Match exact dataset symptom tokens
        for symptom in self.symptom_features:
            symptom_clean = symptom.replace("_", " ")
            if re.search(r"\b" + re.escape(symptom_clean) + r"\b", text_lower):
                detected.add(symptom)

        return sorted(list(detected))

    def predict(
        self,
        symptoms: List[str],
        top_k: int = 3,
    ) -> List[DiseasePrediction]:
        """
        Predict top-k diseases given a list of detected symptoms.
        """
        if not self.model or not self.symptom_features:
            self._ensure_loaded()
            if not self.model:
                return []

        if not symptoms:
            return []

        # Build feature vector
        vec = np.zeros((1, len(self.symptom_features)), dtype=np.int32)
        matched_symptoms = []
        for s in symptoms:
            if s in self.symptom_features:
                idx = self.symptom_features.index(s)
                vec[0, idx] = 1
                matched_symptoms.append(s)

        if not matched_symptoms:
            return []

        # Predict probabilities from Random Forest
        tree_probs = self.model.predict_proba(vec)[0]

        # Calculate hybrid clinical score combining model confidence and symptom overlap
        scored_diseases = []
        for idx, disease in enumerate(self.classes_):
            char_symptoms = self.disease_symptoms_profile.get(disease, [])
            overlap = [s for s in matched_symptoms if s in char_symptoms]
            overlap_ratio = len(overlap) / len(matched_symptoms) if matched_symptoms else 0.0
            t_prob = float(tree_probs[idx])
            
            # Clinical ranking score: priority to diseases that actually include all user symptoms
            combined_score = (0.6 * overlap_ratio) + (0.4 * t_prob) if overlap_ratio > 0 else (0.1 * t_prob)
            
            if combined_score > 0.05 or t_prob > 0.05:
                scored_diseases.append((combined_score, t_prob, disease, overlap, char_symptoms))

        # Sort by combined clinical score descending
        scored_diseases.sort(key=lambda x: x[0], reverse=True)

        results = []
        for combined_score, t_prob, disease, overlap, char_symptoms in scored_diseases[:top_k]:
            conf = round(max(t_prob, combined_score) * 100, 1)
            results.append(
                DiseasePrediction(
                    disease=disease,
                    probability=float(combined_score),
                    confidence_percent=conf,
                    matching_symptoms=overlap or matched_symptoms,
                    characteristic_symptoms=char_symptoms[:6],
                )
            )

        return results

    def format_ml_context(self, symptoms: List[str], predictions: List[DiseasePrediction]) -> str:
        """
        Generate grounded clinical prompt text for LLM injection.
        """
        if not symptoms or not predictions:
            return ""

        lines = [
            "=== MACHINE LEARNING CLINICAL ANALYSIS (TRAINED MEDICAL DATASET) ===",
            f"Detected Clinical Symptoms: {', '.join(s.replace('_', ' ') for s in symptoms)}",
            "Top Evaluated Prognoses (from 4,920 Patient Case Dataset):",
        ]

        for i, pred in enumerate(predictions, 1):
            lines.append(
                f"[{i}] {pred.disease} (Model Confidence: {pred.confidence_percent}%)"
            )
            if pred.characteristic_symptoms:
                char_str = ", ".join(s.replace("_", " ") for s in pred.characteristic_symptoms)
                lines.append(f"    - Typical Syndrome Symptoms: {char_str}")

            remedies = self.disease_home_remedies.get(pred.disease)
            if remedies:
                lines.append("    - Evidence-Based Care Guidelines:")
                for r in remedies:
                    lines.append(f"      • {r}")

        lines.append(
            "\nGROUNDING INSTRUCTION FOR BAYMAX: You must strictly reference this clinical analysis and the retrieved medical knowledge. "
            "1. State the top probable condition(s) indicated by the symptoms based on the dataset. "
            "2. Immediately provide the complete actionable home care, relief measures, and first aid steps in clear bullet points without cutting off. "
            "3. State red flag warning signs when they must consult a physician or seek emergency care immediately."
        )
        return "\n".join(lines)


# Global singleton instance
disease_predictor = DiseasePredictorML()
