import py_compile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceSyntaxTest(unittest.TestCase):
    """Every source file must compile; a single syntax error breaks training at import time."""

    def test_all_sources_compile(self) -> None:
        failures = []
        for directory in ("src", "tests"):
            for path in sorted((ROOT / directory).rglob("*.py")):
                try:
                    py_compile.compile(str(path), doraise=True)
                except py_compile.PyCompileError as exc:
                    failures.append(str(exc))
        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
