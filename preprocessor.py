import re
import string
import nltk
import spacy
from pathlib import Path

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Optional: PDF / DOCX extraction
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

def extract_text_from_pdf(filepath: str) -> str:
    """Extract raw text from a PDF resume using pdfplumber."""
    if not PDF_SUPPORT:
        raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")
    text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(filepath: str) -> str:
    """Extract raw text from a DOCX resume."""
    if not DOCX_SUPPORT:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    doc = Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_txt(filepath: str) -> str:
    """Read plain text resume."""
    return Path(filepath).read_text(encoding="utf-8", errors="ignore")


def extract_text(filepath: str) -> str:
    """
    Auto-detect file type and extract text.

    Supported formats: .pdf, .docx, .txt
    """
    ext = Path(filepath).suffix.lower()
    extractors = {
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
        ".txt": extract_text_from_txt,
    }
    if ext not in extractors:
        raise ValueError(f"Unsupported file type: {ext}")
    return extractors[ext](filepath)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Basic cleaning:
      1. Lowercase
      2. Remove URLs
      3. Remove punctuation & special characters
      4. Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()             # collapse spaces
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize a cleaned string into word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter out common English stop words."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize tokens to their base form."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess(text: str) -> list[str]:
    """
    Full preprocessing pipeline:
      raw text → clean → tokenize → remove stopwords → lemmatize

    Returns a list of processed tokens.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens


def preprocess_to_string(text: str) -> str:
    """Convenience wrapper — returns tokens joined as a single string."""
    return " ".join(preprocess(text))
