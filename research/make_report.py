"""Turn research/results/*.json into markdown tables and figures.

  python research/make_report.py

Writes research/results/tables.md and cfo_rmse.png, cfo_sweeps.png, amc_accuracy.png, amc_confusion.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "research" / "results"

NAMES = {
    "repo_dsp_lag1_x4": "Repo DSP (lag-1 on x^4)", "kay_x1": "Kay (x)", "kay_genie_M": "Kay (x^M, M known)",
    "luise_reggiannini_genie_M": "Luise-Reggiannini (M known)", "periodogram_genie_M": "Periodogram (M known)", "blind_multi_M": "Periodogram (M blind)",
    "iq_resnet": "IQ-ResNet (Chen 2023)", "oshea_cfo": "CNN (O'Shea 2017)", "speccfo_coarse": "SpecCFO network only", "speccfo": "SpecCFO v1 (paper conditions)", "speccfo_v2": "SpecCFO v2 (augmented + position channels, proposed)", "speccfo_v2_nocoords": "Ablation: v2 data, no position channels",
    "cumulants_derotated": "Cumulants + SpecCFO derotation", "centroid_shipped_repo4": "Shipped centroid (4 classes)", "centroid_retrained": "Centroid retrained", "cumulants_linear5": "Cumulants (Swami 2000)",
    "vtcnn2": "VT-CNN2 (O'Shea 2016)", "lstm_ap": "LSTM (Rajendran 2018)", "demod_amc_nocomp": "DemodAMC, no CFO removal", "demod_amc": "DemodAMC v1 (paper conditions)", "demod_amc_v2": "DemodAMC v2 (augmented, proposed)",
}
CFO_ORDER = ["repo_dsp_lag1_x4", "kay_x1", "kay_genie_M", "luise_reggiannini_genie_M", "periodogram_genie_M", "blind_multi_M", "iq_resnet", "oshea_cfo", "speccfo_coarse", "speccfo", "speccfo_v2"]
AMC_ORDER = ["centroid_shipped_repo4", "centroid_retrained", "cumulants_linear5", "cumulants_derotated", "vtcnn2", "lstm_ap", "demod_amc_nocomp", "demod_amc", "demod_amc_v2"]


def g(v, d=3):
    return "n/a" if v is None else (f"{v:.{d}g}" if abs(v) < 1e4 else f"{v:.3e}")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] + [" ---: "] * (len(headers) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def bold_best(rows, cols, lower=True):
    """Bold the best numeric cell in each given column (ties kept)."""
    for c in cols:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[c]))
            except ValueError:
                vals.append(None)
        ok = [v for v in vals if v is not None]
        if not ok:
            continue
        best = min(ok) if lower else max(ok)
        for r, v in zip(rows, vals):
            if v is not None and v == best:
                r[c] = f"**{r[c]}**"
    return rows


def cfo_tables(res):
    md = []
    for key, title in (("awgn", "AWGN, all modulations"), ("awgn_narrow", "AWGN, narrow offset range (easier reading of Chen 2023)"), ("rayleigh", "Rayleigh multipath, all modulations")):
        s = res["sets"][key]
        md.append(f"#### {title} ({s['count']} signals)\n\n{s['description']}\n")
        rows = []
        for m in CFO_ORDER:
            r = s["methods"].get(m)
            if r is None:
                continue
            rows.append([NAMES[m], g(r["mse"]), g(r["rmse"]), g(r["median_abs"]), f"{100 * r['gross_rate']:.1f}%"] + [g(r["by_snr"][str(x)]["rmse"]) for x in (0, 10, 20, 30)])
        md.append(table(["Method", "MSE", "RMSE", "Median abs err", "Gross errors", "RMSE 0 dB", "10 dB", "20 dB", "30 dB"], bold_best(rows, [1, 2, 3, 5, 6, 7, 8])))
        md.append("\nUnits are cycles per sample. Multiply by the sample rate for Hz (0.001 is 250 Hz at 250 kS/s). Gross error means more than 0.005 off.\n")
    s = res["sets"]["awgn"]
    mods = sorted({k for r in s["methods"].values() if r for k in r["by_mod"]})
    md.append("#### RMSE by modulation, SNR 10 dB and above, AWGN\n")
    rows = []
    for m in CFO_ORDER:
        r = s["methods"].get(m)
        if r:
            rows.append([NAMES[m]] + [g(r["by_mod"].get(x, {}).get("rmse")) for x in mods])
    md.append(table(["Method"] + mods, rows) + "\n")
    for key, title in (("bpsk_oversampling", "Oversampling (BPSK, Chen 2023 Fig. 5)"), ("bpsk_length", "Signal length (BPSK, Chen 2023 Fig. 6)")):
        runs = res["sets"][key]["runs"]
        md.append(f"#### {title}\n\nRMSE averaged over SNR 0 to 30 dB.\n")
        heads = list(runs)
        rows = []
        for m in CFO_ORDER:
            cells = []
            for h in heads:
                r = runs[h]["methods"].get(m)
                cells.append("n/a" if r is None else g(float(np.sqrt(np.mean([r["by_snr"][str(x)]["mse"] for x in (0, 5, 10, 15, 20, 25, 30)])))))
            if any(c != "n/a" for c in cells):
                rows.append([NAMES[m]] + cells)
        md.append(table(["Method"] + [("sps " if key == "bpsk_oversampling" else "L ") + h for h in heads], bold_best(rows, range(1, len(heads) + 1))) + "\n")
    runs = res["sets"]["oshea_channel"]["runs"]
    md.append("#### O'Shea 2017 setup: error standard deviation in Hz at 400 kS/s, SNR 5 dB\n\nQPSK, RRC 0.25, 4 samples/symbol, CFO within +-50 kHz. Lower is better. CNNs trained at 1024 samples are scored only there.\n")
    chans = ["awgn", "sigma0.5", "sigma1", "sigma2"]
    for chan in chans:
        rows = []
        for m in ("periodogram_genie_M", "blind_multi_M", "kay_genie_M", "iq_resnet", "oshea_cfo", "speccfo_coarse", "speccfo", "speccfo_v2"):
            cells = []
            for L in (64, 128, 256, 512, 1024):
                r = runs[f"{chan}/L{L}"]["methods"].get(m)
                cells.append("n/a" if r is None else g(r["by_snr"]["5"]["std"] * 400000.0))
            if any(c != "n/a" for c in cells):
                rows.append([NAMES[m]] + cells)
        md.append(f"**{chan}**\n\n" + table(["Method", "L=64", "128", "256", "512", "1024"], bold_best(rows, range(1, 6))) + "\n")
    rn = res["sets"]["repo_native"]
    md.append(f"#### The repo's own recipes and metrics ({rn['n']} signals, 1 MS/s, 4096 samples)\n\nThe README's CFO agreement tolerance is 250 Hz and its DC tolerance is 0.05. Signals come from the repo's recipe generator, which SpecCFO never saw in training.\n")
    names = {"legacy_dsp": "Repo DSP (lag-1 on x^4)", "legacy_tinymlp": "Shipped TinyMLP", "speccfo": "SpecCFO v1 (paper conditions)", "speccfo_v2_nocoords": "Ablation: v2 data, no position channels", "speccfo_v2": "SpecCFO v2 (augmented + position channels, proposed)"}
    rows = [[names[k], g(v["rmse_hz"]), g(v["median_abs_hz"]), f"{100 * v['within_250hz']:.0f}%", f"{100 * v['within_25hz']:.0f}%"] for k, v in rn["carrier_offset"].items()]
    md.append(table(["Carrier offset", "RMSE (Hz)", "Median abs err (Hz)", "Within 250 Hz", "Within 25 Hz"], rows) + "\n")
    rows = [[{"dsp": "Sample mean (DSP)", "tinymlp": "Shipped TinyMLP"}[k], g(v["rmse"]), f"{100 * v['within_0.05']:.0f}%", f"{100 * v['within_0.005']:.0f}%"] for k, v in rn["dc"].items()]
    md.append(table(["DC offset", "RMSE", "Within 0.05", "Within 0.005"], rows) + "\n")
    return "\n".join(md)


def amc_tables(res):
    md = []
    for key, title in (("in_distribution", "In distribution (CFO within +-0.05)"), ("cfo_shift", "Carrier offset shift (CFO within +-0.2, 4x wider than training)")):
        s = res["sets"][key]
        md.append(f"#### {title}\n")
        rows = []
        for m in AMC_ORDER:
            r = s.get(m)
            if not r:
                continue
            sel = r.get("selective") or {}
            rows.append([NAMES[m], g(r["accuracy"], 3), g(r["accuracy_snr_ge_0"], 3), g(r["macro_f1"], 3), g(r.get("ece_calibrated", r.get("ece")), 3) if r.get("ece_calibrated", r.get("ece")) is not None else "n/a", f"{100 * sel['coverage']:.0f}% / {100 * sel['accuracy_on_accepted']:.1f}%" if sel.get("accuracy_on_accepted") is not None else "n/a"])
        md.append(table(["Model", "Accuracy", "Accuracy SNR>=0", "Macro F1", "ECE", "Coverage / accuracy kept"], bold_best(rows, [1, 2, 3], lower=False)) + "\n")
        md.append("Classes covered differ by baseline: cumulants cover 5 linear modulations only, the shipped centroid 4. Their rows are scored on those classes only. The 12-class models are scored on all 12, and the next table scores them on the same subsets.\n")
        rows = []
        for m in ("vtcnn2", "lstm_ap", "demod_amc_nocomp", "demod_amc", "demod_amc_v2"):
            r = s.get(m)
            if r:
                rows.append([NAMES[m], g(r["linear5"]["accuracy"], 3), g(r["repo4"]["accuracy"], 3)])
        if rows:
            ref = [["Shipped centroid", "n/a", g(s["centroid_shipped_repo4"]["accuracy"], 3)], ["Cumulants (Swami 2000)", g(s["cumulants_linear5"]["accuracy"], 3), "n/a"], ["Cumulants + SpecCFO derotation", g(s["cumulants_derotated"]["accuracy"], 3), "n/a"]]
            md.append(table(["Same subset, 12-class models restricted", "5 linear classes", "BPSK/QPSK/8PSK/16QAM"], ref + rows) + "\n")
    return "\n".join(md)


def figures(cres, ares):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})
    colors = {"repo_dsp_lag1_x4": "#999", "kay_x1": "#bbb", "kay_genie_M": "#c97", "luise_reggiannini_genie_M": "#a6a", "periodogram_genie_M": "#c60", "blind_multi_M": "#960", "iq_resnet": "#2a7", "oshea_cfo": "#5a5", "speccfo_coarse": "#68f", "speccfo": "#00f", "speccfo_v2": "#e0a"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, title in zip(axes, ("awgn", "rayleigh"), ("AWGN", "Rayleigh multipath")):
        snrs = cres["protocol"]["snrs"]
        for m in CFO_ORDER:
            r = cres["sets"][key]["methods"].get(m)
            if r:
                ax.semilogy(snrs, [r["by_snr"][str(s)]["rmse"] for s in snrs], marker="o", ms=3, lw=2.2 if m.startswith("speccfo") and m != "speccfo_coarse" else 1.1, color=colors[m], label=NAMES[m])
        if key == "awgn":
            ax.semilogy(snrs, [cres["tone_crb_std_cycles"][str(s)] for s in snrs], "k--", lw=1, label="Tone CRB (unmodulated)")
        ax.set_title(f"CFO RMSE, {title}, all modulations")
        ax.set_xlabel("SNR (dB)")
    axes[0].set_ylabel("RMSE (cycles/sample)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(RES / "cfo_rmse.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, pre in zip(axes, ("bpsk_oversampling", "bpsk_length"), ("sps ", "L ")):
        runs = cres["sets"][key]["runs"]
        for m in ("kay_genie_M", "periodogram_genie_M", "iq_resnet", "oshea_cfo", "speccfo", "speccfo_v2"):
            xs, ys = [], []
            for h, r in runs.items():
                v = r["methods"].get(m)
                if v:
                    xs.append(int(h))
                    ys.append(float(np.sqrt(np.mean([v["by_snr"][str(x)]["mse"] for x in (0, 5, 10, 15, 20, 25, 30)]))))
            if xs:
                ax.semilogy(xs, ys, marker="o", color=colors[m], label=NAMES[m], lw=2.2 if m == "speccfo" else 1.2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("samples per symbol" if key == "bpsk_oversampling" else "capture length (samples)")
        ax.set_title("Oversampling" if key == "bpsk_oversampling" else "Signal length")
    axes[0].set_ylabel("RMSE, SNR 0 to 30 dB (cycles/sample)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(RES / "cfo_sweeps.png")
    plt.close(fig)
    if ares:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        ac = {"centroid_shipped_repo4": "#999", "centroid_retrained": "#bb8", "cumulants_linear5": "#c97", "cumulants_derotated": "#a6a", "vtcnn2": "#2a7", "lstm_ap": "#5a5", "demod_amc_nocomp": "#68f", "demod_amc": "#00f", "demod_amc_v2": "#e0a"}
        for ax, key, title in zip(axes, ("in_distribution", "cfo_shift"), ("In distribution", "Carrier offset 4x wider than training")):
            for m in AMC_ORDER:
                r = ares["sets"][key].get(m)
                if r:
                    xs = sorted(int(k) for k in r["by_snr"])
                    ax.plot(xs, [r["by_snr"][str(x)] for x in xs], marker="o", ms=3, color=ac[m], lw=2.2 if m.startswith("demod_amc") and m != "demod_amc_nocomp" else 1.1, label=NAMES[m])
            ax.set_title(title)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylim(0, 1.02)
        axes[0].set_ylabel("Accuracy")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(RES / "amc_accuracy.png")
        plt.close(fig)
        cm = ares["sets"]["in_distribution"].get("demod_amc", {}).get("confusion")
        if cm:
            from signal_sim import MODULATIONS
            cm = np.array(cm, dtype=float)
            cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            fig, ax = plt.subplots(figsize=(6.2, 5.4))
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(MODULATIONS)), MODULATIONS, rotation=60, ha="right")
            ax.set_yticks(range(len(MODULATIONS)), MODULATIONS)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("DemodAMC confusion, all SNRs, row-normalised")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(RES / "amc_confusion.png")
            plt.close(fig)


def main():
    cres = json.loads((RES / "cfo_results.json").read_text())
    ares = json.loads((RES / "amc_results.json").read_text()) if (RES / "amc_results.json").exists() else None
    md = ["## CFO results\n", cfo_tables(cres)]
    if ares:
        md += ["\n## Modulation classification results\n", amc_tables(ares)]
        for name, c in ares.get("calibration", {}).items():
            md.append(f"\n{NAMES[name]} calibration (validation set, disjoint from test): temperature {c['temperature']:.2f}, abstain below {c['abstain_below']:.2f} top probability, chosen so accepted predictions reach {int(100 * c['target_accepted_accuracy'])}% accuracy on validation.\n")
        rn = ares["sets"].get("repo_native")
        if rn:
            md.append(f"\n#### The repo's own recipe signals ({rn['n']} signals, 4 classes, 1 MS/s, 4096 samples)\n\nThe shipped centroid classifier was trained on these recipes, so this is its home distribution. The generator emits one independent symbol per sample (see research/README.md), a case only the v2 models were trained on.\n")
            rows = []
            for m in ["centroid_shipped_repo4", "vtcnn2", "lstm_ap", "demod_amc_nocomp", "demod_amc", "demod_amc_v2"]:
                r = rn["models"].get(m)
                if r:
                    sel = r.get("selective")
                    rows.append([NAMES[m], g(r["accuracy"], 3), g(r["accuracy_clean_stage"], 3), g(r["accuracy_impaired_stage"], 3), f"{100 * sel['coverage']:.0f}% / {100 * sel['accuracy_on_accepted']:.1f}%" if sel else "n/a"])
            md.append(table(["Model", "Accuracy", "Clean stage", "Impaired stage", "Coverage / accuracy kept"], bold_best(rows, [1, 2, 3], lower=False)) + "\n")
    (RES / "tables.md").write_text("\n".join(md))
    figures(cres, ares)
    print("wrote", RES / "tables.md")


if __name__ == "__main__":
    main()
