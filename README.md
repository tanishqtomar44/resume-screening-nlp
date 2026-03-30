# 🧠 Resume Screening System — NLP-Powered

A Python-based resume screening tool that uses NLP techniques (TF-IDF, keyword matching,
Word2Vec) to rank candidates against a job description automatically.

---

## 📁 Project Structure

```
resume_screener/
├── main.py                        # Entry point — run screening from CLI
├── requirements.txt               # All Python dependencies
├── .gitignore
├── config/
│   └── skills.json                # Required skills for keyword matching
├── data/
│   ├── job_description.txt        # Your target job description
│   └── sample_resumes/            # Drop resume files here (.pdf / .docx / .txt)
├── src/
│   ├── __init__.py
│   ├── preprocessor.py            # Text extraction + NLP cleaning pipeline
│   ├── feature_extractor.py       # TF-IDF, BoW, Word2Vec feature builders
│   └── screener.py                # Scoring, ranking, ML classifier
├── models/                        # Saved vectorizers / classifiers (auto-created)
├── results/                       # Output CSVs (auto-created)
└── tests/
    └── test_screener.py           # Unit tests
```

---

## 🚀 Step-by-Step Development Guide

### Step 1 — Project Setup

**1.1 Create the project directory and open in VS Code:**
```bash
mkdir resume_screener
cd resume_screener
code .
```

**1.2 Create a virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**1.3 Install required libraries:**
```bash
pip install -r requirements.txt
```

**1.4 Download NLTK data (one-time setup):**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

**1.5 Download spaCy English model (optional, for advanced NER):**
```bash
python -m spacy download en_core_web_sm
```

---

### Step 2 — Data Acquisition & Preprocessing

#### How to acquire resume data
- **Sample PDFs**: Download from Kaggle's "Resume Dataset" or use your own.
- **Text files**: Export from Word or copy-paste resume content for quick testing.
- Place all resume files in `data/sample_resumes/`.

#### Text Extraction (`src/preprocessor.py`)

The `extract_text()` function auto-detects file type:

```python
from src.preprocessor import extract_text

text = extract_text("data/sample_resumes/jane_doe.pdf")
```

Supported formats: `.pdf` (via pdfplumber), `.docx` (via python-docx), `.txt`

#### NLP Preprocessing Pipeline

```python
from src.preprocessor import preprocess

raw = "Experienced Machine Learning Engineer with 5 years in Python, TensorFlow."
tokens = preprocess(raw)
# → ['experience', 'machine', 'learning', 'engineer', '5', 'year', 'python', 'tensorflow']
```

Pipeline steps:
1. **Lowercase** — `"Machine Learning"` → `"machine learning"`
2. **Remove URLs, punctuation** — via regex
3. **Tokenize** — NLTK `word_tokenize()`
4. **Remove stop words** — NLTK `stopwords.words('english')`
5. **Lemmatize** — `WordNetLemmatizer` (`"running"` → `"run"`)

---

### Step 3 — Feature Extraction (`src/feature_extractor.py`)

#### TF-IDF (recommended default)

```python
from src.feature_extractor import TFIDFExtractor

extractor = TFIDFExtractor(max_features=5000, ngram_range=(1, 2))
matrix = extractor.fit_transform(list_of_preprocessed_strings)
# matrix is a sparse matrix of shape (n_docs, n_features)
```

**Why TF-IDF?** It rewards terms that are frequent in a document but rare across the corpus —
perfect for identifying distinctive skills and experiences.

#### Bag-of-Words

```python
from src.feature_extractor import BagOfWordsExtractor

bow = BagOfWordsExtractor(max_features=3000)
matrix = bow.fit_transform(documents)
feature_names = bow.get_feature_names()  # list of vocabulary terms
```

#### Word2Vec Embeddings (optional — semantic similarity)

```python
from src.feature_extractor import Word2VecExtractor
from src.preprocessor import preprocess

token_lists = [preprocess(doc) for doc in raw_documents]
w2v = Word2VecExtractor(vector_size=100)
w2v.fit(token_lists)
embeddings = w2v.transform(token_lists)   # shape: (n_docs, 100)
```

---

### Step 4 — Screening Logic (`src/screener.py`)

#### A. TF-IDF Cosine Similarity Ranking

Ranks all resumes by how similar they are to the job description vector.

```python
from src.screener import ResumeRanker

ranker = ResumeRanker()
results = ranker.rank(
    job_description="...",          # raw JD text
    resumes=["...", "...", "..."],   # list of raw resume texts
    resume_ids=["alice.pdf", "bob.docx", "carol.txt"]
)
print(results)
#  rank  resume_id    similarity_score
#  1     alice.pdf    78.34
#  2     carol.txt    61.20
#  3     bob.docx     32.10
```

#### B. Keyword / Skill Matching

```python
from src.screener import keyword_score

required = ["python", "machine learning", "sql", "docker"]
result = keyword_score(resume_text, required)
# → {'score': 75.0, 'matched': ['python', 'sql', 'docker'], 'missing': ['machine learning']}
```

#### C. Combined Scoring

The `main.py` script blends both signals:

```
combined_score = (similarity_score × 0.6) + (keyword_score × 0.4)
```

You can adjust the weights in `main.py` to match your hiring priorities.

#### D. ML Classifier (advanced — requires labelled data)

```python
from src.screener import ResumeClassifier

clf = ResumeClassifier()
report = clf.train(texts=resume_texts, labels=[1, 0, 1, 1, 0])  # 1=suitable
print(report)  # classification report

predictions = clf.predict(["...new resume text..."])
# → [{'label': 'Suitable', 'confidence': 87.3}]

clf.save()   # saves model + vectorizer to models/
clf.load()   # reload later without retraining
```

---

### Step 5 — Running the System

**Basic run (TF-IDF ranking only):**
```bash
python main.py \
  --jd data/job_description.txt \
  --resumes data/sample_resumes/ \
  --top 5
```

**With keyword matching:**
```bash
python main.py \
  --jd data/job_description.txt \
  --resumes data/sample_resumes/ \
  --skills config/skills.json \
  --top 5 \
  --output results/screening_run1.csv
```

**Run unit tests:**
```bash
python -m pytest tests/ -v
# or
python -m unittest tests/test_screener.py -v
```

---

### Step 6 — Upload to GitHub

**6.1 Initialize a Git repository in VS Code terminal:**
```bash
cd resume_screener
git init
```

**6.2 Review and verify `.gitignore`** (already included in this project):
```bash
cat .gitignore
# Confirm venv/, models/*.pkl, results/ are excluded
```

**6.3 Stage all project files:**
```bash
git add .
git status   # review what will be committed
```

**6.4 Make your first commit:**
```bash
git commit -m "feat: initial commit — NLP resume screening system"
```

**6.5 Create a new repository on GitHub:**
1. Go to https://github.com/new
2. Repository name: `resume-screening-nlp`
3. Choose Public or Private
4. **Do NOT** initialize with README (you already have one)
5. Click "Create repository"

**6.6 Connect local repo to GitHub remote:**
```bash
git remote add origin https://github.com/YOUR_USERNAME/resume-screening-nlp.git
git branch -M main
```

**6.7 Push to GitHub:**
```bash
git push -u origin main
```

**6.8 Verify on GitHub** — visit `https://github.com/YOUR_USERNAME/resume-screening-nlp`

**6.9 Subsequent pushes (after making changes):**
```bash
git add .
git commit -m "feat: add Word2Vec embedding support"
git push
```

---

## ⚙️ Customization Tips

| Goal | What to change |
|------|----------------|
| Change scoring weights | Edit `combined_score` formula in `main.py` |
| Add more required skills | Edit `config/skills.json` |
| Support new file types | Add extractor in `src/preprocessor.py` |
| Use a pre-trained model | Load GloVe vectors in `Word2VecExtractor` |
| Add a web UI | Wrap `ResumeRanker` in a Flask/FastAPI app |

---

## 📋 Requirements

| Library | Purpose |
|---------|---------|
| `nltk` | Tokenization, stopwords, lemmatization |
| `spacy` | Advanced NLP (NER, dependency parsing) |
| `scikit-learn` | TF-IDF, BoW, ML classifiers |
| `pandas` / `numpy` | Data manipulation |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `gensim` | Word2Vec embeddings |
| `joblib` | Model serialization |

---

## 📄 License

MIT License — free to use, modify, and distribute.
