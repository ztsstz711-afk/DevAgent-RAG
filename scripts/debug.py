from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose an AI development error")
    parser.add_argument("error")
    args = parser.parse_args()
    result = RAGPipeline().ask(args.error)
    print(json.dumps(result.get("error_info", {}), ensure_ascii=False, indent=2))
    print("\n" + result["answer"])
    print("\nTool trace:")
    print(json.dumps(result["tool_trace"], ensure_ascii=False, indent=2))
