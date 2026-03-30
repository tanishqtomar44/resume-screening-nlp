"""
main.py
-------
Entry point for the Resume Screening System.

Usage
-----
  # Screen resumes from a folder against a job description file:
  python main.py --jd data/job_description.txt --resumes data/sample_resumes/ --top 5

  # Also run keyword matching with a skills config:
  python main.py --jd data/job_description.txt --resumes data/sample_resumes/ --skills config/skills.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessor import extract_text
from src.screener import ResumeRanker, keyword_score, extract_keywords, SKILL_CATEGORIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_resumes(folder: str) -> tuple[list[str], list[str]]:
    """Load all supported resume files from a directory."""
    folder_path = Path(folder)
    files = [
        f for f in folder_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        print(f"[!] No supported resume files found in: {folder}")
        sys.exit(1)

    texts, ids = [], []
    for f in files:
        try:
            text = extract_text(str(f))
            if text.strip():
                texts.append(text)
                ids.append(f.name)
                print(f"  ✓ Loaded: {f.name}")
            else:
                print(f"  ✗ Empty file skipped: {f.name}")
        except Exception as e:
            print(f"  ✗ Error loading {f.name}: {e}")

    return texts, ids


def print_banner():
    print("=" * 60)
    print("        RESUME SCREENING SYSTEM  —  NLP-Powered")
    print("=" * 60)


def save_results(df: pd.DataFrame, output_path: str = "results/ranking.csv"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"\n[✓] Results saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NLP Resume Screening System")
    parser.add_argument("--jd", required=True, help="Path to job description (.txt)")
    parser.add_argument("--resumes", required=True, help="Folder containing resume files")
    parser.add_argument("--top", type=int, default=10, help="Number of top resumes to display")
    parser.add_argument("--skills", default=None, help="JSON file with required skills list")
    parser.add_argument("--output", default="results/ranking.csv", help="Output CSV path")
    args = parser.parse_args()

    print_banner()

    # ── Load Job Description ─────────────────────────────────────────────
    print(f"\n[1] Loading job description: {args.jd}")
    jd_text = Path(args.jd).read_text(encoding="utf-8", errors="ignore")
    print(f"    → {len(jd_text.split())} words detected")

    # ── Load Resumes ──────────────────────────────────────────────────────
    print(f"\n[2] Loading resumes from: {args.resumes}")
    resume_texts, resume_ids = load_resumes(args.resumes)
    print(f"    → {len(resume_texts)} resumes loaded")

    # ── TF-IDF Ranking ────────────────────────────────────────────────────
    print("\n[3] Running TF-IDF similarity ranking …")
    ranker = ResumeRanker()
    ranking_df = ranker.rank(jd_text, resume_texts, resume_ids)

    top_n = min(args.top, len(ranking_df))
    print(f"\n{'─'*45}")
    print(f"  TOP {top_n} CANDIDATES BY TF-IDF SIMILARITY")
    print(f"{'─'*45}")
    print(ranking_df.head(top_n).to_string())

    # ── Keyword Matching ──────────────────────────────────────────────────
    required_skills = []
    if args.skills:
        with open(args.skills) as f:
            required_skills = json.load(f).get("required_skills", [])
    
    if required_skills:
        print(f"\n[4] Keyword matching — required skills: {required_skills}\n")
        kw_results = []
        for rid, rtext in zip(resume_ids, resume_texts):
            result = keyword_score(rtext, required_skills)
            kw_results.append({
                "resume_id": rid,
                "keyword_score": result["score"],
                "matched": ", ".join(result["matched"]),
                "missing": ", ".join(result["missing"]),
            })
        kw_df = pd.DataFrame(kw_results).sort_values("keyword_score", ascending=False)
        print(kw_df.to_string(index=False))

        # Merge with TF-IDF ranking
        merged = ranking_df.reset_index().merge(kw_df[["resume_id", "keyword_score"]], on="resume_id")
        merged["combined_score"] = (
            merged["similarity_score"] * 0.6 + merged["keyword_score"] * 0.4
        ).round(2)
        merged = merged.sort_values("combined_score", ascending=False).reset_index(drop=True)
        merged.index += 1
        merged.index.name = "rank"

        print(f"\n{'─'*45}")
        print("  FINAL COMBINED RANKING (60% TF-IDF + 40% Keywords)")
        print(f"{'─'*45}")
        print(merged[["resume_id", "similarity_score", "keyword_score", "combined_score"]].head(top_n).to_string())
        save_results(merged, args.output)
    else:
        save_results(ranking_df, args.output)

    # ── Skill Category Report for Top Resume ─────────────────────────────
    best_idx = ranking_df.iloc[0].name - 1
    best_resume_text = resume_texts[best_idx]
    best_resume_id = resume_ids[best_idx]

    print(f"\n[5] Skill category breakdown — Best candidate: {best_resume_id}")
    skills_found = extract_keywords(best_resume_text)
    for cat, skills in skills_found.items():
        print(f"    {cat.upper()}: {', '.join(skills)}")

    print("\n[✓] Screening complete.")


if __name__ == "__main__":
    main()
