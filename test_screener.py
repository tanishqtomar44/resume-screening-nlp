import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from src.preprocessor import clean_text, tokenize, remove_stopwords, lemmatize, preprocess
from src.screener import keyword_score, extract_keywords
from src.feature_extractor import TFIDFExtractor, BagOfWordsExtractor

class TestPreprocessor(unittest.TestCase):

    def test_clean_text_lowercases(self):
        self.assertEqual(clean_text("Hello World"), "hello world")

    def test_clean_text_removes_punctuation(self):
        result = clean_text("Hello, World! #NLP")
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)
        self.assertNotIn("#", result)

    def test_clean_text_removes_urls(self):
        result = clean_text("visit https://example.com for details")
        self.assertNotIn("https", result)

    def test_tokenize_returns_list(self):
        tokens = tokenize("machine learning python")
        self.assertIsInstance(tokens, list)
        self.assertIn("machine", tokens)

    def test_remove_stopwords(self):
        tokens = ["i", "am", "a", "data", "scientist"]
        result = remove_stopwords(tokens)
        self.assertNotIn("i", result)
        self.assertNotIn("a", result)
        self.assertIn("data", result)

    def test_lemmatize(self):
        tokens = ["running", "models", "libraries"]
        result = lemmatize(tokens)
        self.assertIn("model", result)
        self.assertIn("library", result)

    def test_preprocess_pipeline(self):
        text = "I have 5 years of experience in Machine Learning and Deep Learning."
        result = preprocess(text)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        # All tokens should be lowercase
        for token in result:
            self.assertEqual(token, token.lower())

class TestKeywordScreener(unittest.TestCase):

    SAMPLE_RESUME = """
    Experienced Python developer with 4 years in machine learning.
    Proficient in TensorFlow, PyTorch, SQL, and AWS.
    Strong communication and teamwork skills.
    """

    def test_keyword_score_full_match(self):
        result = keyword_score(self.SAMPLE_RESUME, ["python", "machine learning", "sql"])
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(len(result["missing"]), 0)

    def test_keyword_score_partial_match(self):
        result = keyword_score(self.SAMPLE_RESUME, ["python", "kubernetes"])
        self.assertGreater(result["score"], 0)
        self.assertIn("kubernetes", result["missing"])

    def test_keyword_score_no_match(self):
        result = keyword_score(self.SAMPLE_RESUME, ["cobol", "fortran"])
        self.assertEqual(result["score"], 0.0)

    def test_extract_keywords(self):
        skills = extract_keywords(self.SAMPLE_RESUME)
        self.assertIn("programming", skills)
        self.assertIn("python", skills["programming"])

class TestFeatureExtractor(unittest.TestCase):

    DOCS = [
        "python machine learning deep learning",
        "sql database data warehouse etl",
        "docker kubernetes cloud aws devops",
    ]

    def test_tfidf_fit_transform_shape(self):
        extractor = TFIDFExtractor(max_features=100)
        matrix = extractor.fit_transform(self.DOCS)
        self.assertEqual(matrix.shape[0], 3)

    def test_tfidf_transform_after_fit(self):
        extractor = TFIDFExtractor(max_features=100)
        extractor.fit_transform(self.DOCS)
        new_matrix = extractor.transform(["python sql"])
        self.assertEqual(new_matrix.shape[0], 1)

    def test_bow_feature_names(self):
        extractor = BagOfWordsExtractor(max_features=50)
        extractor.fit_transform(self.DOCS)
        names = extractor.get_feature_names()
        self.assertIn("python", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
