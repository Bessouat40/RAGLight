"""Run one documented unittest tier with stable discovery semantics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "tests"
UNIT_DIRECTORIES = (
    "architecture",
    "contracts",
    "tests_document_processing",
    "tests_embeddings",
    "tests_llm",
    "tests_rag",
    "tests_vector_store",
)
TIERS = {
    "unit": UNIT_DIRECTORIES,
    "integration": ("integration",),
    "e2e": ("e2e",),
}


def suite_for(tier: str) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for directory in TIERS[tier]:
        loader = unittest.TestLoader()
        discovered = loader.discover(
            start_dir=str(TEST_ROOT / directory),
            pattern="test_*.py",
            top_level_dir=str(PROJECT_ROOT),
        )
        suite.addTests(discovered)
    return suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", choices=TIERS)
    arguments = parser.parse_args()

    result = unittest.TextTestRunner(verbosity=2).run(suite_for(arguments.tier))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
