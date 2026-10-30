"""Golden-question check for retrieval (and optionally answers).

Run from the backend folder:
    python -m scripts.eval_retrieval --ingest ../docs/GEP-Jun-2026.pdf ../docs/GPO-Anual-report.pdf
    python -m scripts.eval_retrieval            # retrieval only: no chat-model calls
    python -m scripts.eval_retrieval --answer   # also ask the chat model

A case passes when an expected string appears in the retrieved context (or answer).
"""

import argparse
import sys
from pathlib import Path

from app.pipeline import pipeline

CASES = [
    # (source, question, any of these strings must appear)
    ("GEP-Jun-2026.pdf", "What is the global growth forecast for 2026?", ["2.5 percent", "2026f: 2.5"]),
    ("GEP-Jun-2026.pdf", "What is India's growth forecast for FY2026/27?", ["India: 2023: 7.2; 2024: 7.1; 2025e: 7.7; 2026f: 6.6"]),
    ("GEP-Jun-2026.pdf", "What is Viet Nam's GDP growth forecast for 2028?", ["2028f: 7.4"]),
    ("GEP-Jun-2026.pdf", "What Brent oil price is assumed for 2026?", ["$94"]),
    ("GEP-Jun-2026.pdf", "How low could global growth fall if the conflict comes with financial stress?", ["1.3 percent"]),
    ("GEP-Jun-2026.pdf", "How much previously hidden debt did Mozambique reveal in 2016?", ["US$1.15 billion"]),
    ("GEP-Jun-2026.pdf", "How much does a 1 percentage point rise in debt add to sovereign spreads when debt is 80 percent of GDP?", ["26 basis points"]),
    ("GEP-Jun-2026.pdf", "Which East Asia and Pacific economy has the highest growth forecast for 2026?", ["Viet Nam: 2023: 5.0; 2024: 7.0; 2025e: 8.0; 2026f: 6.8"]),
    ("GEP-Jun-2026.pdf", "How much will AI affect global growth?", ["Box 1.1"]),
    ("GPO-Anual-report.pdf", "How many libraries did GPO visit in FY2018?", ["166"]),
    ("GPO-Anual-report.pdf", "Which system replaced FDsys?", ["govinfo"]),
    ("GPO-Anual-report.pdf", "How much printing did GPO procure in FY2018?", ["375"]),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", nargs="*", default=[], help="PDFs to (re)ingest first")
    parser.add_argument("--answer", action="store_true", help="also generate answers with the chat model")
    args = parser.parse_args()

    for path in args.ingest:
        count = pipeline.ingest_file(Path(path))
        print(f"ingested {Path(path).name}: {count} chunks")

    loaded = {m.get("source") for m in pipeline.collection.get(include=["metadatas"])["metadatas"]}
    passed = total = 0
    for source, question, expected in CASES:
        if source not in loaded:
            continue
        total += 1
        hits = pipeline.retrieve(question)
        context = "\n".join(h["text"] for h in hits)
        ok = any(e in context for e in expected)
        line = f"{'PASS' if ok else 'FAIL'}  {question}"
        if args.answer:
            answer = pipeline.generate_answer(question, hits)
            ok = ok and any(e.split(":")[-1].strip() in answer for e in expected)
            line += f"\n      -> {answer.strip()[:300]}"
        passed += ok
        print(line)
        if not ok:
            for h in hits:
                print(f"      p.{h['page_start']} {h['kind']}: {h['text'][:110]!r}")
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
