"""Turn report/report.md into report/report.pdf.

    python scripts/build_report.py

The report is written in Markdown so it is easy to edit. This script converts it in two steps:
  1. pandoc turns the Markdown into one HTML page, styled by report/report.css;
  2. Google Chrome (without opening a window) prints that page to a PDF.
Both programs must be installed: pandoc from https://pandoc.org, and Chrome or Chromium.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
)


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit("Could not find Google Chrome or Chromium, which is needed to print the PDF.")


def build(source: Path = REPORT_DIR / "report.md", output: Path = REPORT_DIR / "report.pdf") -> Path:
    if not shutil.which("pandoc"):
        raise SystemExit("pandoc is not installed. Get it from https://pandoc.org/installing.html")
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "report.html"
        subprocess.run(
            ["pandoc", str(source), "--standalone", "--embed-resources", "--from", "markdown+smart",
             "--css", str(REPORT_DIR / "report.css"), "--resource-path", f"{REPORT_DIR}:{ROOT}",
             "--metadata", "lang=en", "--output", str(html)],
            check=True)
        subprocess.run(
            [find_chrome(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={output}", html.as_uri()],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output


if __name__ == "__main__":
    print("Wrote", build().relative_to(ROOT))
    sys.exit(0)
