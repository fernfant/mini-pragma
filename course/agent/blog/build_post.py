#!/usr/bin/env python3
"""Build a CxAI build-story post: .md -> Substack-styled .html (inline SVGs) -> .pdf.
Usage: python3 build_post.py <slug>   # operates on build-story_<slug>_<date>.md
Reuses the <head><style> from the most recent existing post as the template.
"""
import sys, re, subprocess, pathlib, glob, html

HERE = pathlib.Path(__file__).parent
KICKER = "Building with AI"
DATE = "June 1, 2026 · Consumer Experience × AI"

def main(stem):
    md_path = HERE / f"{stem}.md"
    md = md_path.read_text()
    lines = md.splitlines()
    # pull title (first # ) and subtitle (first *...* after it)
    title = next(l[2:].strip() for l in lines if l.startswith("# "))
    sub = ""
    for i, l in enumerate(lines):
        if l.startswith("# "):
            for l2 in lines[i+1:]:
                if l2.strip().startswith("*") and l2.strip().endswith("*"):
                    sub = l2.strip().strip("*").strip(); break
            break
    # body markdown = everything after the subtitle line
    body_md = re.sub(r'^#\s.*?\n', '', md, count=1, flags=re.S)
    body_md = re.sub(r'^\s*\*.*?\*\s*\n', '', body_md, count=1, flags=re.M)
    # render body
    frag = subprocess.run(["pandoc", "-f", "markdown", "-t", "html5"],
                          input=body_md, capture_output=True, text=True, check=True).stdout
    # inline SVGs: replace <img ... src="assets/x.svg" ...> with the file content
    def inline(m):
        name = m.group(1)
        svg = (HERE / "assets" / name).read_text()
        return svg.strip()
    frag = re.sub(r'<img[^>]*src="assets/([^"]+\.svg)"[^>]*/?>', inline, frag)
    # template head from most recent prior .html
    priors = sorted(glob.glob(str(HERE / "build-story_*.html")))
    head = pathlib.Path(priors[-1]).read_text().split("</head>")[0] + "</head>\n"
    # swap <title>
    head = re.sub(r'<title>.*?</title>', f'<title>{html.escape(title)} — CxAI</title>', head, flags=re.S)
    page = f"""{head}<body>
  <header class="pub-header"><div class="pub-inner"><div class="pub-name">CxAI<span class="dot">.</span></div><button class="subscribe">Subscribe</button></div></header>
  <article class="article">
    <div class="kicker">{KICKER}</div>
    <h1 class="title">{html.escape(title)}</h1>
    <p class="subtitle">{html.escape(sub)}</p>
    <div class="byline"><div class="avatar">Cx</div><div><div class="who">CxAI Team</div><div class="when">{DATE}</div></div></div>
    <div class="body">
{frag}
      <div class="signoff">Stay tuned! — CxAI Team</div>
    </div>
  </article>
  <div class="footer-note">CxAI is a consumer-experience × AI newsletter. Every figure in this piece traces to a project artifact: the scored rubric, the <a href="https://fernfant.github.io/mini-pragma/course/html/scorecard.html">live scorecard</a>, and the dated audit reports. The course itself is public at <a href="https://fernfant.github.io/mini-pragma/course/html/index.html">fernfant.github.io/mini-pragma</a>.</div>
</body>
</html>"""
    out_html = HERE / f"{stem}.html"
    out_html.write_text(page)
    print("wrote", out_html.name, f"({len(page)} bytes)")
    # PDF via headless Chrome
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    out_pdf = HERE / f"{stem}.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", "--virtual-time-budget=3000",
                    out_html.as_uri()], check=True, capture_output=True)
    print("wrote", out_pdf.name)

if __name__ == "__main__":
    main(sys.argv[1])
