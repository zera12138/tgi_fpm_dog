#!/usr/bin/env python3
"""去掉 pdf2zh/babeldoc 输出里被重建的链接边框（默认 /Border [0 0 1] + 绿色描边）。

用法: strip_link_borders.py FILE_OR_DIR [...]
"""
import sys
from pathlib import Path

import fitz


def strip(path: Path) -> int:
    doc = fitz.open(path)
    n = 0
    for page in doc:
        for link in page.get_links():
            xref = link.get("xref")
            if not xref:
                continue
            doc.xref_set_key(xref, "Border", "[0 0 0]")
            doc.xref_set_key(xref, "C", "[0 0 0]")
            n += 1
    if n:
        tmp = path.with_suffix(".tmp.pdf")
        doc.save(tmp, garbage=3, deflate=True)
        doc.close()
        tmp.replace(path)
    return n


def main(argv: list[str]) -> int:
    targets: list[Path] = []
    for arg in argv:
        p = Path(arg)
        targets.extend(sorted(p.glob("*.pdf")) if p.is_dir() else [p])
    for t in targets:
        print(f"{t.name}: {strip(t)} links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
