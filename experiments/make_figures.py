"""Generate the paper figures from the result JSON.

Publication style: no chart junk, no titles inside the axes (captions live in the
text), direct labels where they fit, minimal spines. Run after the experiments:

    python experiments/make_figures.py
"""
import json, os, statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = f"{ROOT}/results", f"{ROOT}/docs"
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREY, DGREY = "#2a78d6", "#eb6834", "#9a988f", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "text.color": "#111111", "axes.labelcolor": "#111111",
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def load(n):
    return json.load(open(f"{RES}/{n}"))


def fig1_collapse():
    e1 = load("e1_acceptance_collapse.json")["rows"]
    kind = {int(k): v["kind"] for k, v in load("task_adapters.json").items()}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, field, lab in zip(axes, ("beta_analytic", "beta_greedy"),
                              ("stochastic acceptance  $\\beta$",
                               "greedy acceptance (top-1 agreement)")):
        for t in sorted({r["tenant"] for r in e1}):
            rs = sorted([r for r in e1 if r["tenant"] == t], key=lambda r: r["strength"])
            c = BLUE if kind[t] == "in-dist" else ORANGE
            ax.plot([r["strength"] for r in rs], [r[field] for r in rs],
                    color=c, lw=1.3, marker="o", ms=2.6, alpha=0.9)
        ax.set_xlabel("tenant adapter strength $s$")
        ax.set_title(lab, pad=6, color=DGREY)
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("acceptance rate")
    axes[0].plot([], [], color=BLUE, lw=1.3, label="in-distribution tenants")
    axes[0].plot([], [], color=ORANGE, lw=1.3, label="held-out tenants")
    axes[0].legend(loc="lower left", handlelength=1.4)
    fig.savefig(f"{OUT}/fig1_acceptance_collapse.png")
    plt.close(fig)


def fig2_theory():
    e1 = load("e1_acceptance_collapse.json")["rows"]
    kind = {int(k): v["kind"] for k, v in load("task_adapters.json").items()}
    b0 = load("stage0.json")["base_acceptance"]["beta_analytic"]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    mx = max(r["shift_tvd"] for r in e1)
    ax.plot([0, mx], [b0, b0 - mx], color=GREY, lw=1.1, ls="--", zorder=1)
    ax.annotate("$\\beta_0 - \\Delta$", xy=(mx * 0.55, b0 - mx * 0.55), xytext=(6, 8),
                textcoords="offset points", color=DGREY, fontsize=8)
    for k, c in (("in-dist", BLUE), ("held-out", ORANGE)):
        pts = [r for r in e1 if kind[r["tenant"]] == k]
        ax.scatter([r["shift_tvd"] for r in pts], [r["beta_analytic"] for r in pts],
                   s=14, color=c, edgecolor="white", linewidth=0.5, zorder=3,
                   label=f"{k} tenants")
    ax.set_xlabel("$\\Delta$ = TVD(base target, adapted target)")
    ax.set_ylabel("acceptance rate $\\beta$")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(loc="upper right", handletextpad=0.3)
    fig.savefig(f"{OUT}/fig2_beta_vs_shift.png")
    plt.close(fig)


def fig3_repair():
    rows = load("e2_companion_repair.json")["rows"]
    b0 = load("stage0.json")["base_acceptance"]["beta_analytic"]
    arms, order = {}, []
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)
        if r["arm"] not in order:
            order.append(r["arm"])
    label = {"shared-drafter": "no companion adapter",
             "companion-tvd-r4": "companion r=4, TVD loss",
             "companion-fkl-r4": "companion r=4, forward KL",
             "companion-rkl-r4": "companion r=4, reverse KL",
             "companion-ce-r4": "companion r=4, hard-label CE",
             "companion-tvd-r4-offpolicy": "companion r=4, off-policy data",
             "companion-tvd-r1": "companion r=1", "companion-tvd-r2": "companion r=2",
             "companion-tvd-r8": "companion r=8",
             "full-ft-tvd": "full drafter fine-tune (26$\\times$ params)"}
    order = order[::-1]
    vals = [st.mean(x["beta_analytic"] for x in arms[a]) for a in order]
    cols = [ORANGE if a == "shared-drafter" else (GREY if a == "full-ft-tvd" else BLUE)
            for a in order]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    y = range(len(order))
    ax.barh(list(y), vals, color=cols, height=0.62)
    ax.axvline(b0, color=DGREY, lw=1.0, ls="--")
    ax.annotate("$\\beta_0$ = %.3f" % b0, xy=(b0, len(order) - 0.30), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center", color=DGREY, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8, color="#111111")
    ax.set_yticks(list(y)); ax.set_yticklabels([label.get(a, a) for a in order])
    ax.set_xlabel("mean acceptance rate $\\beta$ over 6 tenants")
    ax.set_xlim(0, 1.16)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.savefig(f"{OUT}/fig3_companion_repair.png")
    plt.close(fig)


def fig4_sensitivity():
    from tandemspec.metrics import expected_tokens_per_step
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    betas = [i / 200 for i in range(1, 200)]
    for g, c in ((2, "#b7d3f6"), (4, "#6da7ec"), (8, "#2a78d6"), (16, "#184f95")):
        ax.plot(betas, [expected_tokens_per_step(b, g) / (g + 1) for b in betas],
                color=c, lw=1.4)
        ax.annotate(f"$\\gamma$={g}", xy=(0.985, expected_tokens_per_step(0.985, g) / (g + 1)),
                    xytext=(3, -2), textcoords="offset points", color=c, fontsize=8)
    ax.set_xlabel("acceptance rate $\\beta$")
    ax.set_ylabel("tokens/step, as fraction of max")
    ax.set_xlim(0, 1.12); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.savefig(f"{OUT}/fig4_depth_sensitivity.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    fig1_collapse(); fig2_theory(); fig3_repair(); fig4_sensitivity()
    for f in sorted(os.listdir(OUT)):
        if f.startswith("fig"):
            print(f"{OUT}/{f}  ({os.path.getsize(f'{OUT}/{f}')//1024} KB)")
