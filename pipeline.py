"""Babylon: Climate Tipping Point Cascade Monitor.

Have you built the tower too high? Computes halving depth D for the climate
tipping point system at different warming levels. D > 3.03 means the system
has crossed from modular to irreversible cascade.

Usage:
    python3 pipeline.py                     # all warming levels
    python3 pipeline.py --temp 2.0          # specific temperature
"""
import os
import argparse
import numpy as np
from coupling import (
    cov_from_coupling, tc_gaussian, per_component_bill,
    compile_gap, node_residuals, halving_depth, regularize
)
from tipping_elements import ELEMENTS, ELEMENT_NAMES, build_coupling_matrix

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CAPACITY_LINE = 3.03


def analyze_temperature(temperature_C):
    """Run the full coupling analysis at a given warming level."""
    C, names = build_coupling_matrix(temperature_C)
    S = cov_from_coupling(C)
    S, shrink = regularize(S)
    tc = tc_gaussian(S)
    bill = per_component_bill(S)

    levels = compile_gap(C)
    parts = [p for _, p in levels]
    Rn, R = node_residuals(C, parts)
    D, exact = halving_depth(R)

    rankings = sorted(zip(names, bill), key=lambda x: -x[1])

    modules = parts[0] if parts else [[i] for i in range(len(names))]
    module_map = {}
    for mi, mod in enumerate(modules):
        for idx in mod:
            module_map[names[idx]] = mi

    n_edges = int((C > 0).sum()) // 2

    return {
        "D": D,
        "TC": tc,
        "temperature": temperature_C,
        "n_elements": len(names),
        "n_couplings": n_edges,
        "names": names,
        "bill": bill,
        "rankings": rankings,
        "modules": modules,
        "module_map": module_map,
        "R": R,
        "C": C,
    }


def print_snapshot(result):
    """Print the tipping point analysis."""
    D = result["D"]
    T = result["temperature"]
    verdict = "MODULAR (tipping points addressable independently)" if D < CAPACITY_LINE else "GLOBAL (tipping cascade — triggering one risks all)"

    lines = []
    lines.append(f"{'='*75}")
    lines.append(f"BABYLON — CLIMATE TIPPING CASCADE ANALYSIS — {T:.1f}°C warming")
    lines.append(f"{'='*75}")
    lines.append(f"")
    d_str = f"{D:.2f}" if D != np.inf else "infinity"
    lines.append(f"  Halving depth D = {d_str}    [{verdict}]")
    lines.append(f"  Capacity line:    3.03")
    lines.append(f"  Total correlation: {result['TC']:.2f}")
    lines.append(f"  Tipping elements: {result['n_elements']}")
    lines.append(f"  Active couplings: {result['n_couplings']}")
    lines.append(f"")
    lines.append(f"  TIPPING ELEMENT BILL RANKING (cascade contribution)")
    lines.append(f"  {'─'*65}")
    lines.append(f"  {'Rank':>4}  {'Element':<6} {'Name':<30} {'Bill':>8}  {'Module':>6}  {'Threshold'}")

    for i, (name, b) in enumerate(result["rankings"]):
        info = ELEMENTS[name]
        mod = result["module_map"].get(name, "?")
        thresh = f"{info['threshold_C']:.1f}°C"
        at_risk = " ← AT RISK" if T >= info["threshold_C"] - 0.5 else ""
        tipped = " ← TIPPING" if T >= info["threshold_C"] else ""
        marker = tipped if tipped else at_risk
        lines.append(f"  {i+1:>4}  {name:<6} {info['name']:<30} {b:>8.4f}  {mod:>6}  {thresh}{marker}")

    lines.append(f"")
    lines.append(f"  MODULE STRUCTURE (coupling clusters)")
    lines.append(f"  {'─'*65}")
    for mi, mod in enumerate(result["modules"]):
        names_in = [result["names"][idx] for idx in mod]
        cats = set(ELEMENTS[n]["category"] for n in names_in)
        full_names = [ELEMENTS[n]["name"] for n in names_in]
        lines.append(f"  Module {mi:>2}: {', '.join(names_in):<40} [{', '.join(cats)}]")
        lines.append(f"            {', '.join(full_names)}")

    lines.append(f"")
    lines.append(f"  Residual at each level: {', '.join(f'{r:.3f}' for r in result['R'])}")

    return "\n".join(lines)


def print_temperature_sweep(results):
    """Print the D vs temperature table."""
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"BABYLON — CLIMATE TIPPING CASCADES — D vs WARMING LEVEL")
    lines.append(f"Capacity line: D = 3.03  |  Above → CASCADE RISK  |  Below → INDEPENDENTLY ADDRESSABLE")
    lines.append(f"{'='*80}")
    lines.append(f"")
    lines.append(f"{'Warming':>8}  {'D':>8}  {'Verdict':<12} {'TC':>6}  {'Edges':>6}  {'#1 Element':>12}  {'Bill':>6}  Note")
    lines.append(f"{'─'*85}")

    notes = {
        0.5: "",
        1.0: "← Arctic sea ice + reefs at threshold",
        1.2: "← CURRENT WARMING",
        1.5: "← Paris Agreement target",
        2.0: "← Paris upper limit",
        2.5: "",
        3.0: "← Current policy trajectory",
        4.0: "← High emissions scenario",
        5.0: "← Worst case",
    }

    for r in results:
        T = r["temperature"]
        d_str = f"{r['D']:.2f}" if r['D'] != np.inf else "     inf"
        verdict = "CASCADE" if r["D"] >= CAPACITY_LINE else "MODULAR"
        note = notes.get(T, "")
        top = r["rankings"][0]
        lines.append(
            f"  {T:>5.1f}°C  {d_str:>8}  {verdict:<12} {r['TC']:>6.2f}  {r['n_couplings']:>6}  "
            f"{top[0]:>12}  {top[1]:>6.3f}  {note}"
        )

        if r["D"] >= CAPACITY_LINE and (len(results) < 2 or
            results[max(0, results.index(r)-1)]["D"] < CAPACITY_LINE):
            lines.append(f"  {'':>8}  {'':>8}  {'*** CAPACITY LINE CROSSED ***'}")

    lines.append(f"")

    modular = [r for r in results if r["D"] < CAPACITY_LINE]
    cascade = [r for r in results if r["D"] >= CAPACITY_LINE]
    if modular and cascade:
        last_safe = max(r["temperature"] for r in modular)
        first_danger = min(r["temperature"] for r in cascade)
        lines.append(f"  The climate system crosses from modular to cascade risk")
        lines.append(f"  between {last_safe:.1f}°C and {first_danger:.1f}°C of warming.")
        lines.append(f"")
        lines.append(f"  Below {last_safe:.1f}°C: tipping points can be addressed independently.")
        lines.append(f"  Above {first_danger:.1f}°C: triggering one tipping point risks triggering all.")
    elif not modular:
        lines.append(f"  The climate system is in CASCADE territory at all analyzed temperatures.")
    else:
        lines.append(f"  The climate system remains MODULAR at all analyzed temperatures.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Babylon: Climate Tipping Point Cascade Monitor")
    parser.add_argument("--temp", type=float, help="Analyze at a specific temperature (°C above pre-industrial)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.temp is not None:
        result = analyze_temperature(args.temp)
        report = print_snapshot(result)
        print(report)
        out = os.path.join(RESULTS_DIR, f"snapshot_{args.temp:.1f}C.txt")
        with open(out, "w") as f:
            f.write(report)
        print(f"\nSaved → {out}")
    else:
        temperatures = [0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        results = []
        for T in temperatures:
            r = analyze_temperature(T)
            results.append(r)

        sweep_report = print_temperature_sweep(results)
        print(sweep_report)

        out = os.path.join(RESULTS_DIR, "temperature_sweep.txt")
        with open(out, "w") as f:
            f.write(sweep_report)
        print(f"\nSaved → {out}")

        # Also save the detailed snapshot at current warming
        current = [r for r in results if r["temperature"] == 1.2][0]
        detail = print_snapshot(current)
        out2 = os.path.join(RESULTS_DIR, "current_1.2C.txt")
        with open(out2, "w") as f:
            f.write(detail)
        print(f"Current detailed → {out2}")

        target = [r for r in results if r["temperature"] == 1.5][0]
        detail2 = print_snapshot(target)
        out3 = os.path.join(RESULTS_DIR, "paris_1.5C.txt")
        with open(out3, "w") as f:
            f.write(detail2)
        print(f"Paris target   → {out3}")


if __name__ == "__main__":
    main()
