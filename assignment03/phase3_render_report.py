"""
phase3_render_report.py — render melanie_report.md to a styled, self-contained HTML.

Usage:  python3 phase3_render_report.py
Output: ../A3_MelanieAndStephen_<StudentID1>_<StudentID2>/melanie_report.html
(the image results_progression.png is referenced relatively; keep them in the same folder)
"""
from pathlib import Path
import markdown

BASE = Path(__file__).resolve().parent.parent / "A3_MelanieAndStephen_<StudentID1>_<StudentID2>"
SRC = BASE / "melanie_report.md"
OUT = BASE / "melanie_report.html"

CSS = """
:root{--ink:#1a1d21;--muted:#5b6470;--line:#e6e8eb;--accent:#4a3aa7;--accent2:#1baf7a;
--bg:#ffffff;--panel:#f7f8fa;--code:#0b1020;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:48px 28px 96px}
h1{font-size:30px;line-height:1.25;margin:0 0 18px;letter-spacing:-.4px;border-bottom:3px solid var(--accent);padding-bottom:12px}
h2{font-size:22px;margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line);letter-spacing:-.2px}
h3{font-size:17px;margin:26px 0 8px;color:#33373d}
p{margin:12px 0}
a{color:var(--accent)}
code{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:1px 6px;
 font:13.5px/1.5 "SF Mono",Menlo,Consolas,monospace;color:#7a3ea7}
pre{background:var(--code);color:#e6e9f0;border-radius:10px;padding:16px 18px;overflow:auto;
 font:13px/1.6 "SF Mono",Menlo,Consolas,monospace}
pre code{background:none;border:none;color:inherit;padding:0}
table{width:100%;border-collapse:collapse;margin:16px 0;font-size:14.5px;
 border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}
th{background:var(--panel);color:var(--muted);font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.3px}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:#fafbfc}
td:not(:first-child){font-variant-numeric:tabular-nums}
img{max-width:100%;height:auto;display:block;margin:20px auto;border:1px solid var(--line);border-radius:10px}
em{color:var(--muted)}
blockquote{border-left:3px solid var(--accent2);margin:16px 0;padding:4px 16px;color:var(--muted);background:var(--panel)}
ul,ol{padding-left:24px}
li{margin:5px 0}
hr{border:0;border-top:1px solid var(--line);margin:32px 0}
.foot{margin-top:48px;color:var(--muted);font-size:13px;text-align:center;border-top:1px solid var(--line);padding-top:16px}
@media (prefers-color-scheme: dark){
 :root{--ink:#e6e8ee;--muted:#9aa3b2;--line:#2a2f3d;--bg:#0f1117;--panel:#171a23;--accent:#9085e9;--accent2:#6ee7b7}
 body{background:var(--bg)} code{color:#c7b3f5}
 th{background:#12151d} tbody tr:hover{background:#141821}
 pre{background:#0a0c12}
}
"""

def main():
    md_text = SRC.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"noclasses": True, "pygments_style": "friendly"}},
    )
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase 3 Report — EvoAgent (Kaggle 0.747)</title>
<style>{CSS}</style></head>
<body><div class="wrap">
{html_body}
<p class="foot">Rendered from melanie_report.md · EvoAgent Advanced-NLP06 Assignment 03</p>
</div></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"rendered {OUT} ({len(page)//1024} KB)")


if __name__ == "__main__":
    main()
