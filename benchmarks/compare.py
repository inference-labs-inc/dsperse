import argparse
import json
import sys


REGRESSION_THRESHOLD = 0.20


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return {b["name"]: b for b in data["benchmarks"]}


def format_delta(base_val, head_val):
    if base_val == 0:
        return "N/A", "N/A"
    diff = head_val - base_val
    pct = (diff / base_val) * 100
    sign = "+" if diff > 0 else ""
    if pct > 5:
        indicator = "slower"
    elif pct < -5:
        indicator = "faster"
    else:
        indicator = "~same"
    return f"{sign}{diff:.3f}s", f"{sign}{pct:.1f}% ({indicator})"


def build_table(base, head):
    all_names = list(dict.fromkeys(list(base.keys()) + list(head.keys())))

    lines = []
    lines.append("| Benchmark | main | PR | Delta | Change |")
    lines.append("|:---|---:|---:|---:|:---|")

    regressions = []

    for name in all_names:
        b = base.get(name)
        h = head.get(name)

        if b and h:
            base_med = b["median"]
            head_med = h["median"]
            delta_str, change_str = format_delta(base_med, head_med)
            base_rss = f"{b.get('peak_rss_mb', 0):.0f}MB"
            head_rss = f"{h.get('peak_rss_mb', 0):.0f}MB"
            lines.append(
                f"| {name} | {base_med:.3f}s ({base_rss}) | {head_med:.3f}s ({head_rss}) | {delta_str} | {change_str} |"
            )
            if base_med > 0 and (head_med - base_med) / base_med > REGRESSION_THRESHOLD:
                regressions.append((name, base_med, head_med))
        elif h:
            lines.append(f"| {name} | - | {h['median']:.3f}s | new | new |")
        elif b:
            lines.append(f"| {name} | {b['median']:.3f}s | - | removed | removed |")

    return "\n".join(lines), regressions


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("base", help="Base (main) benchmark JSON")
    parser.add_argument("head", help="Head (PR) benchmark JSON")
    parser.add_argument("--output", "-o", help="Write markdown to file")
    args = parser.parse_args()

    base = load_results(args.base)
    head = load_results(args.head)

    table, regressions = build_table(base, head)

    header = "## Benchmark Results\n\n"
    footer = f"\n\n<sub>{len(base)} benchmarks compared, {REGRESSION_THRESHOLD*100:.0f}% regression threshold</sub>"

    if regressions:
        warning = "\n\n**Regressions detected:**\n"
        for name, b, h in regressions:
            pct = ((h - b) / b) * 100
            warning += f"- {name}: {b:.3f}s -> {h:.3f}s (+{pct:.1f}%)\n"
    else:
        warning = ""

    md = header + table + warning + footer

    if args.output:
        with open(args.output, "w") as f:
            f.write(md)
        print(f"Markdown written to {args.output}")
    else:
        print(md)

    if regressions:
        print(f"\nWARNING: {len(regressions)} benchmark(s) regressed >{REGRESSION_THRESHOLD*100:.0f}%", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
