from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask DevAgent-RAG a technical question")
    parser.add_argument("question")
    args = parser.parse_args()
    result = RAGPipeline().ask(args.question)
    print(result["answer"])
    print(f"\nQuality: {result['quality']['score']:.2f} ({'passed' if result['quality']['passed'] else 'needs review'})")
