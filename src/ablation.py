"""
ablation.py

Ablation Study CLI and Summary Reporter.
Compares:
  A0: Semantic baseline (all-MiniLM-L6-v2 + FAISS IndexFlatIP)
  A1: + Entity-aware retrieval (+player/team/opponent bonuses)
  A2: + Temporal context (+sequence continuity & build-up)
  A3: + Match-state reasoning (+game phase & score margin)
  A4: + Event importance (+criticality weighting)
  A5: + Factual verification (+post-generation consistency & auto-correction)
"""

from pathlib import Path
import pandas as pd
import json

from evaluation import run_comprehensive_evaluation, EVAL_OUTPUT_DIR


def print_ablation_summary():
    csv_file = EVAL_OUTPUT_DIR / "ablation_results.csv"
    if not csv_file.exists():
        print("Ablation results not found. Running evaluation...")
        run_comprehensive_evaluation()

    df = pd.read_csv(csv_file)
    print("\n" + "=" * 80)
    print("ABLATION STUDY: RETRIEVAL & FACTUAL ACCURACY PROGRESSION")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print_ablation_summary()
