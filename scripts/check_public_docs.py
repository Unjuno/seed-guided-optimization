"""Check local link targets in publication entry pages; no network requests.

This validates file/directory destinations, not remote URLs or section fragments.
Historical experiment reports are outside this entry-page check.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    "README.md", "README.en.md", "docs/README.md", "docs/QUICKSTART.md",
    "docs/SHARE_ON_X.md", "experiments/README.md", "results/README.md",
)


def main() -> int:
    errors = []
    targets = set()
    for name in PAGES:
        page = ROOT / name
        if not page.is_file():
            errors.append(f"missing entry page: {name}")
            continue
        text = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.S)
        for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            link = urlsplit(raw.strip())
            if link.scheme in ("https", "http", "mailto"):
                continue
            if link.scheme or link.netloc or link.path.startswith("/"):
                errors.append(f"unsupported local link in {name}: {raw}")
                continue
            if not link.path:
                continue
            destination = (page.parent / unquote(link.path)).resolve()
            if not destination.is_relative_to(ROOT):
                errors.append(f"link leaves repository in {name}: {raw}")
            elif not destination.exists():
                errors.append(f"missing link target in {name}: {raw}")
            else:
                targets.add(destination.relative_to(ROOT).as_posix())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PUBLIC DOC LINKS VERIFIED: {len(PAGES)} entry pages; {len(targets)} local targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
