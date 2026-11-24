"""Smoke test: every dashboard page runs without exceptions.  Run: python tests/test_app_pages.py"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

pages = [ROOT / "app" / "Home.py"] + sorted((ROOT / "app" / "pages").glob("*.py"))
skip = set(sys.argv[1:])
failed = 0
for p in pages:
    if any(s in p.name for s in skip):
        continue
    at = AppTest.from_file(str(p), default_timeout=180).run()
    errs = [e.value for e in at.exception]
    print(f"{'FAIL' if errs else 'ok  '} {p.name}" + (f"  -> {errs[0][:400]}" if errs else ""))
    failed += bool(errs)
sys.exit(failed)
