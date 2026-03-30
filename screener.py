import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from src.preprocessor import preprocess, preprocess_to_string
from src.feature_extractor import TFIDFExtractor

SKILL_CATEGORIES = {
    "programming": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go",
        "rust", "scala", "kotlin", "swift", "r", "matlab",
    ],
    "ml_ai": [
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "transformers",
        "llm", "neural network", "reinforcement learning",
    ],
    "data": [
        "sql", "pandas", "numpy", "spark", "hadoop", "kafka",
        "data pipeline", "etl", "data warehouse", "tableau", "power bi",
    ],
    "cloud": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ci/cd", "devops", "microservices",
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "agile", "scrum", "project management",
    ],
}


def extract_keywords(text: str, skill_dict: dict = SKILL_CATEGORIES) -> dict:
    
    text_lower = text.lower()
    matched = {}
    for category, skills in skill_dict.items():
        found = [s for s in skills if re.search(r"\b" + re.escape(s) + r"\b", text_lower)]
        if found:
            matched[category] = found
    return matched


def keyword_score(resume_text: str, required_skills: list[str]) -> dict:
   
    text_lower = resume_text.lower()
    matched = [s for s in required_skills if re.search(r"\b" + re.escape(s.lower()) + r"\b", text_lower)]
    missing = [s for s in required_skills if s not in matched]
    score = round(len(matched) / len(required_skills) * 100, 1) if required_skills else 0
    return {"score": score, "matched": matched, "missing": missing}

class ResumeRanker:

    def __init__(self):
        self.extractor = TFIDFExtractor(max_features=8000)

    def rank(
        self,
        job_description: str,
        resumes: list[str],
        resume_ids: list[str] | None = None,
    ) -> pd.DataFrame:
       
        if resume_ids is None:
            resume_ids = [f"resume_{i+1}" for i in range(len(resumes))]

        # Preprocess all documents
        jd_clean = preprocess_to_string(job_description)
        resumes_clean = [preprocess_to_string(r) for r in resumes]

        # Fit on JD + resumes combined so vocabulary is shared
        all_docs = [jd_clean] + resumes_clean
        matrix = self.extractor.fit_transform(all_docs)

        jd_vec = matrix[0]
        resume_vecs = matrix[1:]

        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(jd_vec, resume_vecs)[0]

        df = pd.DataFrame({
            "resume_id": resume_ids,
            "similarity_score": np.round(scores * 100, 2),
        }).sort_values("similarity_score", ascending=False).reset_index(drop=True)
        df.index += 1  # rank starts at 1
        df.index.name = "rank"
        return df

class ResumeClassifier:
    
    def __init__(self):
        self.extractor = TFIDFExtractor(max_features=5000)
        self.model = LogisticRegression(max_iter=1000, C=1.0)
        self._trained = False

    def train(self, texts: list[str], labels: list[int]) -> str:
        
        processed = [preprocess_to_string(t) for t in texts]
        X = self.extractor.fit_transform(processed)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        self._trained = True

        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred, target_names=["Not Suitable", "Suitable"])

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Predict suitability for new resumes.

        Returns list of dicts with 'label' and 'confidence'.
        """
        if not self._trained:
            raise RuntimeError("Train the model first using .train()")
        processed = [preprocess_to_string(t) for t in texts]
        X = self.extractor.transform(processed)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return [
            {
                "label": "Suitable" if p == 1 else "Not Suitable",
                "confidence": round(float(probs[i][p]) * 100, 1),
            }
            for i, p in enumerate(preds)
        ]

    def save(self, model_path="models/classifier.pkl", vec_path="models/clf_vectorizer.pkl"):
        Path("models").mkdir(exist_ok=True)
        joblib.dump(self.model, model_path)
        self.extractor.save(vec_path)

    def load(self, model_path="models/classifier.pkl", vec_path="models/clf_vectorizer.pkl"):
        self.model = joblib.load(model_path)
        self.extractor.load(vec_path)
        self._trained = True
