"""HTML and PDF reports."""
from __future__ import annotations

import base64
import html
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .model import fourpl

COL = {"ok": "#1b7f3b", "warn": "#b36b00", "undefined": "#a1262b"}


def plot_curve(A, ax, title=""):
    x = A.x
    ax.scatter(x, A.y, s=18, color="#33415c", zorder=3, label="data")
    if A.fit is not None and np.isfinite(A.fit.sse):
        xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 300)
        if A.boot is not None and A.boot["n_ok"] >= 20:
            curves = np.array([fourpl(xs, *p) for p in A.boot["samples"]])
            ax.fill_between(xs, *np.quantile(curves, [0.025, 0.975], axis=0), color="#9db4d8", alpha=0.45,
                            label="95% bootstrap band")
        ax.plot(xs, fourpl(xs, *A.fit.params), color="#c1121f", lw=1.6, label="4PL fit")
    if A.logic50 is not None:
        ax.axvline(A.logic50, color="#555", ls="--", lw=1)
        if A.logic50_ci:
            ax.axvspan(*A.logic50_ci, color="#f4a261", alpha=0.25, label="IC50 95% CI")
    ax.set_xlabel(f"log10 concentration ({A.unit})")
    ax.set_ylabel("response")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="best")


def _fmt_ic50(A):
    if A.ic50 is None:
        return "undefined (not reported)"
    s = f"{A.ic50:.3g} {A.unit}"
    if A.ic50_ci:
        s += f" (95% CI {A.ic50_ci[0]:.3g}–{A.ic50_ci[1]:.3g})"
    return s


def _png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def html_report(items, title="Dose–response report", intro="", extra_html=""):
    """items: list of (name, Analysis)."""
    css = ("body{font-family:system-ui,sans-serif;max-width:980px;margin:2em auto;padding:0 1em;color:#222}"
           "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:4px 8px;font-size:.9em;text-align:left}"
           ".card{border:1px solid #ddd;border-radius:6px;padding:12px;margin:18px 0}.badge{color:#fff;padding:1px 8px;border-radius:10px;font-size:.8em}"
           "li.fatal{color:#a1262b}li.warn{color:#b36b00}li.info{color:#555}")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>",
             f"<h1>{html.escape(title)}</h1>", intro]
    parts.append("<table><tr><th>Sample</th><th>Status</th><th>IC50</th><th>Hill</th><th>R²</th><th>Flags</th></tr>")
    for name, A in items:
        flags = ", ".join(c for c, _, _ in A.flags) or "–"
        hill = f"{A.fit.hill:.2f}" if A.fit is not None and np.isfinite(A.fit.hill) else "–"
        r2 = f"{A.r2:.3f}" if A.r2 is not None else "–"
        parts.append(f"<tr><td>{html.escape(str(name))}</td><td><span class='badge' style='background:{COL[A.status]}'>{A.status}</span></td>"
                     f"<td>{html.escape(_fmt_ic50(A))}</td><td>{hill}</td><td>{r2}</td><td>{html.escape(flags)}</td></tr>")
    parts.append("</table>")
    for name, A in items:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        plot_curve(A, ax, str(name))
        img = _png_b64(fig)
        parts.append(f"<div class='card'><h3>{html.escape(str(name))} — <span style='color:{COL[A.status]}'>{A.status}</span></h3>"
                     f"<img alt='curve' src='data:image/png;base64,{img}'><p><b>IC50:</b> {html.escape(_fmt_ic50(A))}</p><ul>")
        for c, s, m in A.flags:
            parts.append(f"<li class='{s}'><b>{c}</b>: {html.escape(m)}</li>")
        parts.append("</ul></div>")
    parts.append(extra_html + "</body></html>")
    return "".join(parts)


def pdf_report(items, path, title="Dose–response report"):
    with PdfPages(path) as pdf:
        for i in range(0, len(items), 4):
            fig, axes = plt.subplots(2, 2, figsize=(8.27, 8))
            fig.suptitle(title, fontsize=11)
            for ax, (name, A) in zip(axes.ravel(), items[i:i + 4]):
                plot_curve(A, ax, f"{name} | {A.status} | IC50 {_fmt_ic50(A)}")
            for ax in axes.ravel()[len(items[i:i + 4]):]:
                ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)
