"""小さな SVG 図ライブラリ(グループ枠・ノード・番号バッジ・矢印)。参考: 番号付きフロー図 / グループ化した構成図。"""
from html import escape

FONT = "'Segoe UI','Hiragino Sans','Noto Sans JP','Yu Gothic UI',sans-serif"
MONO = "'Cascadia Mono',Consolas,'Courier New',monospace"
C = dict(bg="#f6f8fc", text="#1b2333", muted="#5b667a", line="#6b7589",
         blue="#2f6fed", orange="#d9730d", green="#1b9a6c", purple="#7a4fd6", red="#d64545", teal="#0f9aa8", gray="#7d8798")
TINT = dict(blue="#eef3ff", orange="#fff4e8", green="#e9f8f1", purple="#f1ecff", red="#ffeeee", teal="#e6f7f9", gray="#f0f2f6")


class Svg:
    def __init__(self, w, h, title, desc=""):
        self.w, self.h = w, h
        self.p = []
        self.title, self.desc = title, desc

    def add(self, s): self.p.append(s)

    def rect(self, x, y, w, h, fill="#fff", stroke="none", rx=10, dash=None, sw=1.5):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, fill=None, weight=400, anchor="start", mono=False):
        fam = MONO if mono else FONT
        self.add(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill or C["text"]}" text-anchor="{anchor}">{escape(s)}</text>')

    def group(self, x, y, w, h, label, color, sub="", dash=None):
        self.rect(x, y, w, h, TINT[color], C[color], 14, dash)
        self.text(x + 18, y + 27, label, 15, C[color], 700)
        if sub: self.text(x + 18, y + 45, sub, 11.5, C["muted"])

    def node(self, x, y, w, h, title, sub="", color="blue", mono=False):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="#fff" stroke="#d3d9e6" stroke-width="1.2"/>')
        self.add(f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="{C[color]}"/>')
        ty = y + (h / 2 + 5 if not sub else h / 2 - 3)
        self.text(x + 18, ty, title, 13.5, C["text"], 600, mono=mono)
        if sub: self.text(x + 18, y + h / 2 + 15, sub, 11, C["muted"])

    def badge(self, x, y, n, color="blue"):
        self.add(f'<circle cx="{x}" cy="{y}" r="12" fill="{C[color]}"/>')
        self.text(x, y + 4.5, str(n), 12.5, "#fff", 700, "middle")

    def arrow(self, pts, color="line", dash=None, label=None, lpos=None, width=1.8):
        col = C.get(color, color)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        path = "M" + " L".join(f"{x},{y}" for x, y in pts)
        self.add(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="{width}"{d} marker-end="url(#ah-{color})"/>')
        if label:
            lx, ly = lpos or ((pts[0][0] + pts[-1][0]) / 2, (pts[0][1] + pts[-1][1]) / 2 - 6)
            self.add(f'<text x="{lx}" y="{ly}" font-family="{FONT}" font-size="11.5" fill="{col}" text-anchor="middle" '
                     f'paint-order="stroke" stroke="{C["bg"]}" stroke-width="4">{escape(label)}</text>')

    def render(self):
        marks = "".join(
            f'<marker id="ah-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{v}"/></marker>' for k, v in C.items() if k not in ("bg", "text", "muted"))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" role="img" '
                f'aria-labelledby="t d"><title id="t">{escape(self.title)}</title><desc id="d">{escape(self.desc)}</desc>'
                f'<defs>{marks}</defs><rect width="{self.w}" height="{self.h}" rx="16" fill="{C["bg"]}"/>' + "".join(self.p) + "</svg>")

    def save(self, path):
        open(path, "w", encoding="utf-8").write(self.render())
