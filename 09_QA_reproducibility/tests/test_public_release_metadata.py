import json
import re
import subprocess
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _tracked_utf8_text() -> dict[str, str]:
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    ).decode("utf-8").split("\0")
    text: dict[str, str] = {}
    for relative_path in filter(None, paths):
        try:
            text[relative_path] = (ROOT / relative_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return text


class PublicReleaseMetadataTests(unittest.TestCase):
    def test_v102_author_and_version_metadata_are_consistent(self):
        public_text_files = _tracked_utf8_text()
        old_comma_name = re.compile(r"Zhao,\s+Jin(?!g)")
        old_natural_name = re.compile(r"(?<![A-Za-z])Jin\s+Zhao(?![A-Za-z])")
        corrected_comma_name = "Zhao" + ", Jing"
        corrected_natural_name = "Jing" + " Zhao"

        self.assertEqual(
            {path for path, text in public_text_files.items() if old_comma_name.search(text)},
            set(),
        )
        self.assertEqual(
            {path for path, text in public_text_files.items() if old_natural_name.search(text)},
            set(),
        )
        self.assertEqual(
            {path for path, text in public_text_files.items() if corrected_comma_name in text},
            {".zenodo.json", "qa/public_release_receipt_v1.0.2.json"},
        )
        self.assertIn(corrected_natural_name, public_text_files["LICENSE"])
        self.assertIn(corrected_natural_name, public_text_files["pyproject.toml"])
        self.assertIn(corrected_natural_name, public_text_files["qa/public_release_decision.md"])
        self.assertRegex(
            public_text_files["CITATION.cff"],
            r"(?ms)^  - family-names: Zhao\n    given-names: Jing$",
        )

        citation_text = public_text_files["CITATION.cff"]
        citation_version = re.search(r"(?m)^version:\s*(\S+)\s*$", citation_text)
        citation_date = re.search(r"(?m)^date-released:\s*(\S+)\s*$", citation_text)
        self.assertIsNotNone(citation_version)
        self.assertIsNotNone(citation_date)

        zenodo = json.loads(public_text_files[".zenodo.json"])
        pyproject = tomllib.loads(public_text_files["pyproject.toml"])

        self.assertEqual(citation_version.group(1), "1.0.2")
        self.assertEqual(zenodo["version"], "1.0.2")
        self.assertEqual(pyproject["project"]["version"], "1.0.2")
        self.assertEqual(citation_date.group(1), "2026-08-11")
        self.assertEqual(zenodo["publication_date"], "2026-08-11")
        self.assertIn("Version 1.0.2", public_text_files["README.md"])
        self.assertIn("**Candidate:** version 1.0.2", public_text_files["qa/public_release_decision.md"])

        self.assertEqual(zenodo["license"], "mit")
        self.assertEqual(pyproject["project"]["license"], {"text": "MIT"})
        self.assertRegex(citation_text, r"(?m)^license:\s*MIT$")
        self.assertIn(
            "Creative Commons Attribution 4.0 International License (CC BY 4.0)",
            public_text_files["LICENSE-DOCUMENTATION-AND-AGGREGATES.md"],
        )


if __name__ == "__main__":
    unittest.main()
