# -*- coding: utf-8 -*-
"""رسم Hit@1 مقابل حجم المقطع من results.md (فاتح وداكن لـ README).

يقرأ: results.md (مخرج المرحلة 4). ينتج: docs/size-vs-hit1-{light,dark}.png.
يتطلب matplotlib (للرسم فقط، ليس من متطلبات خط الأنابيب).
"""

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_MD, _HERE

# ألوان من الخانات الثلاث الأولى المعتمدة (تجاوزت المدقّق في الوضعين)
_THEMES = {
    "light": dict(surface="#ffffff", ink="#191919", muted="#666666", grid="#e6e6e6",
                  fixed="#257de0", paragraph="#f47230", semantic="#00b894"),
    "dark": dict(surface="#191919", ink="#efefef", muted="#9c9c9c", grid="#2e2e2e",
                 fixed="#338ced", paragraph="#e76624", semantic="#00a888"),
}


def read_rows() -> dict[str, dict]:
    if not RESULTS_MD.exists():
        sys.exit(f"لا يوجد {RESULTS_MD}: شغّل 04_eval.py أولاً")
    rows = {}
    for line in RESULTS_MD.read_text(encoding="utf-8").splitlines():
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        if len(cells) < 10 or cells[0] in ("Method", "---"):
            continue
        lo, hi = (float(x) for x in cells[7].split("–"))
        rows[cells[0].split(" ")[0]] = {
            "x": float(cells[2]), "y": float(cells[4]), "lo": lo, "hi": hi,
        }
    return rows


def draw(theme: str, rows: dict[str, dict]) -> None:
    t = _THEMES[theme]
    fixed = sorted((r for k, r in rows.items() if k.startswith("fixed")), key=lambda r: r["x"])
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    def err(r):  # فترة الثقة 95% كأعمدة خطأ نحيلة
        return [[r["y"] - r["lo"]], [r["hi"] - r["y"]]]

    # الثابت: خط 2px مع علامات دائرية، حلقة 2px بلون السطح بين العلامات المتداخلة
    ax.errorbar([r["x"] for r in fixed], [r["y"] for r in fixed],
                yerr=[[r["y"] - r["lo"] for r in fixed], [r["hi"] - r["y"] for r in fixed]],
                color=t["fixed"], lw=2, marker="o", ms=9, mec=t["surface"], mew=2,
                capsize=3, elinewidth=1.4, zorder=3)
    for key, color, marker in (("paragraph", t["paragraph"], "s"), ("semantic", t["semantic"], "D")):
        r = rows[key]
        ax.errorbar([r["x"]], [r["y"]], yerr=err(r), color=color, marker=marker, ms=10,
                    mec=t["surface"], mew=2, capsize=3, elinewidth=1.4, zorder=4, ls="none")

    # تسميات مباشرة بلون النص لا لون السلسلة (قاعدة التباين)
    kw = dict(color=t["ink"], fontsize=10.5, va="center")
    # كل تسمية في مساحة فارغة: الثابت تحت شريط خطئه، والدلالي فوق قمة شريطه
    ax.annotate("fixed windows (100 / 150 / 200 words)", (fixed[1]["x"], fixed[1]["lo"]),
                xytext=(0, -14), textcoords="offset points", ha="center", **{**kw, "va": "top"})
    ax.annotate("semantic", (rows["semantic"]["x"], rows["semantic"]["hi"]),
                xytext=(0, 12), textcoords="offset points", ha="center", **kw)
    ax.annotate("paragraph", (rows["paragraph"]["x"], rows["paragraph"]["y"]),
                xytext=(12, 14), textcoords="offset points", ha="left", **kw)

    ax.set_xlim(85, 235)
    ax.set_ylim(0.62, 0.86)
    ax.set_xlabel("median chunk size (words)", color=t["muted"], fontsize=10)
    ax.set_ylabel("Hit@1 (95% CI)", color=t["muted"], fontsize=10)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Semantic chunks sit on the fixed-window trend: size drives retrieval, not strategy",
                 loc="left", color=t["ink"], fontsize=11.5, pad=14)

    out = _HERE / "docs" / f"size-vs-hit1-{theme}.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print(f"✓ {out}")


def main() -> None:
    rows = read_rows()
    for theme in _THEMES:
        draw(theme, rows)


if __name__ == "__main__":
    main()
