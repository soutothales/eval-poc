"""Command-line entry point: ``python -m eval.cli <command>``.

Commands
--------
dataset validate | rehash | diff | release
run, score, compare, gate, report, history, judge-noise, ask
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from eval import constants as c  # noqa: E402


def _print_issues(issues) -> int:
    """Print validation issues; return the error count.

    Parameters
    ----------
    issues: List[Issue]
        Validation issues.

    Returns
    -------
    int
        Number of errors.
    """
    n_err = 0
    for i in issues:
        n_err += i.level == c.LEVEL_ERROR
        print(f"  [{i.level:7}] {i.check}: {i.message}")
    return n_err


def cmd_dataset_validate(args: argparse.Namespace) -> int:
    """Run all dataset checks against the last release tag.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from agent.retriever import corpus_index, corpus_version
    from eval.versioning.manifest import load_core, load_manifest
    from eval.versioning.release import previous_release
    from eval.versioning.validate import parse_with_issues, validate

    cases, schema_issues = parse_with_issues(c.CASES_PATH.read_text(encoding=c.TEXT_ENCODING))
    prev = previous_release()
    issues = schema_issues + validate(
        cases,
        load_core(),
        corpus_index(),
        manifest=load_manifest(),
        previous=prev.cases if prev else None,
        previous_core=prev.core if prev else None,
        allow_unreleased=args.allow_unreleased,
        corpus_version=corpus_version(),
    )
    print(f"validated {len(cases)} cases" + (f" against {prev.tag}" if prev else ""))
    n_err = _print_issues(issues)
    print("OK" if n_err == 0 else f"FAILED: {n_err} error(s)")
    return 1 if n_err else 0


def cmd_dataset_rehash(args: argparse.Namespace) -> int:
    """Recompute content_hash for every case.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from eval.versioning.manifest import load_cases, rehash_cases, save_cases

    cases = load_cases()
    fresh = rehash_cases(cases)
    changed = sum(a.content_hash != b.content_hash for a, b in zip(cases, fresh))
    save_cases(fresh)
    print(f"rehashed {len(fresh)} cases ({changed} changed)")
    return 0


def cmd_dataset_diff(args: argparse.Namespace) -> int:
    """Print the diff and inferred bump against the last release tag.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from eval.versioning.diff import diff_cases
    from eval.versioning.manifest import load_cases, load_core
    from eval.versioning.release import previous_release

    prev = previous_release()
    if prev is None:
        print("no previous release tag found; everything is new")
        return 0
    prev_by = {k.case_id: k for k in prev.cases}
    cases = load_cases()
    typo = set(args.typo_fix) | {k.case_id for k in cases if k.case_id in prev_by and k.revision > prev_by[k.case_id].revision}
    d = diff_cases(prev.cases, cases, typo, prev.core.case_ids if prev.core else None, load_core().case_ids)
    print(f"diff vs {prev.tag}:")
    print(json.dumps(d.to_dict(), indent=2))
    return 0


def cmd_dataset_release(args: argparse.Namespace) -> int:
    """Validate, bump the version, write the manifest and tag the release.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from eval.versioning.release import release

    res = release(
        bump=args.bump, what=args.what, why=args.why, author=args.author,
        typo_fixes=args.typo_fix, use_git=not args.no_git,
    )
    if res.diff:
        print(f"inferred bump: {res.diff.inferred_bump} {res.diff.reasons}")
    _print_issues([i for i in res.issues if i.level != c.LEVEL_INFO])
    for m in res.messages:
        print(m)
    if res.ok:
        print(f"released dataset {res.version}")
    return 0 if res.ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    """Run the generate phase.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from eval.generate import generate

    run_id = asyncio.run(generate(
        Path(args.config), repeats=args.repeats, limit=args.limit, core_only=args.core_only,
        concurrency=args.concurrency, resume_run_id=args.resume, retry_errors=args.retry_errors,
    ))
    print(run_id)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Run the score phase and print the summary.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from eval.score import score_run

    summary = asyncio.run(score_run(args.run_id, deterministic_only=args.deterministic_only, rescore=args.rescore))
    _print_summary(summary)
    return 0


def _print_summary(summary) -> None:
    """Print a run summary as a compact table.

    Parameters
    ----------
    summary: Dict[str, Any]
        Output of :func:`eval.summary.summarize`.
    """
    print(f"\nrun {summary[c.SUM_RUN_ID]}: {summary[c.SUM_N_CASES]} cases, "
          f"{summary[c.SUM_N_ERRORS]} agent errors, {summary.get('n_judge_errors', 0)} judge errors")
    print(f"{'metric':24} {'mean':>7} {'pass':>7} {'n':>4}")
    for m, q in summary[c.SUM_QUALITY].items():
        o = q[c.SUM_OVERALL]
        pr = "" if o[c.SUM_PASS_RATE] is None else f"{o[c.SUM_PASS_RATE]:.2f}"
        print(f"{m:24} {o[c.SUM_MEAN]:7.3f} {pr:>7} {o[c.SUM_N]:4d}")
    op = summary[c.SUM_OPERATIONAL]
    cur = op[c.SUM_CURRENCY]
    lat, cost = op[c.SUM_LATENCY], op[c.SUM_AGENT_COST]
    if lat[c.SUM_MEAN] is not None:
        print(f"latency  mean {lat[c.SUM_MEAN]:.2f}s  p95 {lat[c.SUM_P95]:.2f}s")
        print(f"agent cost/question  mean {cost[c.SUM_MEAN]:.6f} {cur}  total {op[c.SUM_AGENT_COST_TOTAL]:.4f} {cur}")
    cpc = op[c.SUM_COST_PER_CORRECT]
    print(f"cost per correct answer: {'—' if cpc is None else f'{cpc:.6f} {cur}'}")
    print(f"judge cost total: {op[c.SUM_JUDGE_COST_TOTAL]:.4f} {cur}")
    if op[c.SUM_UNPRICED_MODELS]:
        print(f"WARNING: unpriced models {op[c.SUM_UNPRICED_MODELS]}; add them to pricing/price_table.json")


def cmd_compare(args: argparse.Namespace) -> int:
    """Print a paired comparison of two runs.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from eval.compare import compare_runs

    res = compare_runs(args.run_a, args.run_b)
    for w in res[c.CMP_WARNINGS]:
        print(f"WARNING: {w}")
    print(f"{res[c.CMP_SHARED_CASES]} shared cases\n")
    print(f"{'metric':24} {'scope':14} {'A':>6} {'B':>6} {'delta':>7}  {'95% CI':>17} {'n':>3}")
    for r in res[c.CMP_QUALITY]:
        if r[c.CMP_SCOPE] != c.SCOPE_OVERALL and not args.categories:
            continue
        print(f"{r[c.CMP_METRIC]:24} {r[c.CMP_SCOPE]:14} {r[c.CMP_MEAN_A]:6.3f} {r[c.CMP_MEAN_B]:6.3f} "
              f"{r[c.CMP_DELTA]:+7.3f}  [{r[c.CMP_CI_LOW]:+.3f}, {r[c.CMP_CI_HIGH]:+.3f}] {r[c.CMP_N]:3d}")
    print()
    def num(v) -> str:
        return "—" if v is None else f"{v:.4g}"

    for r in res[c.CMP_OPERATIONAL]:
        rel = r[c.CMP_REL_CHANGE]
        print(f"{r[c.CMP_METRIC]:24} {num(r[c.CMP_VALUE_A]):>10} -> {num(r[c.CMP_VALUE_B]):<10} "
              + ("" if rel is None else f"{rel:+.1%}"))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Evaluate the regression gate; exit code 1 on breach.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from eval.gate import gate

    res = gate(args.run_a, args.run_b, Path(args.thresholds))
    for w in res[c.CMP_WARNINGS]:
        print(f"WARNING: {w}")
    if res[c.GATE_PASSED]:
        print("GATE PASSED")
        return 0
    print("GATE FAILED")
    for b in res[c.GATE_BREACHES]:
        print(f"  - {b[c.GATE_RULE]}: {b[c.GATE_DETAIL]}")
    return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Write the HTML comparison report.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from eval.report import write_report

    print(write_report(args.run_a, args.run_b))
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Print the trend table across scored runs.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    import pandas as pd

    from eval.store import history

    df = history()
    if df.empty:
        print("no scored runs yet")
        return 0
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
    return 0


def cmd_judge_noise(args: argparse.Namespace) -> int:
    """Measure judge and agent variance on a run.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from eval.noise import AGENT_NOISE, JUDGE_NOISE, MAX_RANGE, MEAN_STD, N_CASES, judge_noise

    res = asyncio.run(judge_noise(args.run_id, times=args.times, sample=args.sample))
    for label, key in (("judge (same output, re-scored)", JUDGE_NOISE), ("agent (repeated runs)", AGENT_NOISE)):
        print(f"\n{label}")
        print(f"{'metric':24} {'mean std':>9} {'max range':>10} {'n':>3}")
        for m, s in res[key].items():
            if s[N_CASES]:
                print(f"{m:24} {s[MEAN_STD]:9.3f} {s[MAX_RANGE]:10.3f} {s[N_CASES]:3d}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Ask the agent a single question.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from agent.graph import AgentConfig
    from eval.adapters.agent_adapter import AgentAdapter

    res = asyncio.run(AgentAdapter(AgentConfig.from_yaml(Path(args.config))).ainvoke(args.question))
    print(res.answer)
    print(f"\nretrieved: {res.retrieved_ids}\nusage: {res.usage}\nlatency: {res.latency_s}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all subcommands.
    """
    p = argparse.ArgumentParser(prog="eval", description="Local continuous-evaluation pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    ds = sub.add_parser(c.CMD_DATASET, help="golden dataset versioning").add_subparsers(dest="dataset_cmd", required=True)
    v = ds.add_parser(c.CMD_VALIDATE, help="run all dataset checks (CI)")
    v.add_argument("--allow-unreleased", action="store_true", help="unreleased changes are warnings, not errors")
    v.set_defaults(func=cmd_dataset_validate)
    ds.add_parser(c.CMD_REHASH, help="recompute content_hash for every case").set_defaults(func=cmd_dataset_rehash)
    d = ds.add_parser(c.CMD_DIFF, help="diff against the last release tag")
    d.add_argument("--typo-fix", nargs="*", default=[], metavar="CASE_ID")
    d.set_defaults(func=cmd_dataset_diff)
    r = ds.add_parser(c.CMD_RELEASE, help="validate, bump version, write manifest, git tag")
    r.add_argument("--bump", choices=[c.BUMP_PATCH, c.BUMP_MINOR, c.BUMP_MAJOR], required=True)
    r.add_argument("--what", required=True)
    r.add_argument("--why", required=True)
    r.add_argument("--author", default=os.environ.get("USER", "unknown"))
    r.add_argument("--typo-fix", nargs="*", default=[], metavar="CASE_ID")
    r.add_argument("--no-git", action="store_true")
    r.set_defaults(func=cmd_dataset_release)

    run = sub.add_parser(c.CMD_RUN, help="phase 1: generate agent outputs")
    run.add_argument("--config", required=True)
    run.add_argument("--repeats", type=int, default=1, help="total runs per core case")
    run.add_argument("--limit", type=int)
    run.add_argument("--core-only", action="store_true")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--resume", metavar="RUN_ID")
    run.add_argument("--retry-errors", action="store_true")
    run.set_defaults(func=cmd_run)

    sc = sub.add_parser(c.CMD_SCORE, help="phase 2: score stored outputs")
    sc.add_argument("run_id")
    sc.add_argument("--deterministic-only", action="store_true", help="skip LLM-judged metrics")
    sc.add_argument("--rescore", action="store_true", help="discard existing scores")
    sc.set_defaults(func=cmd_score)

    cp = sub.add_parser(c.CMD_COMPARE, help="paired comparison of two runs")
    cp.add_argument("run_a")
    cp.add_argument("run_b")
    cp.add_argument("--categories", action="store_true", help="also print per-category rows")
    cp.set_defaults(func=cmd_compare)

    g = sub.add_parser(c.CMD_GATE, help="exit 1 if B regresses vs A beyond thresholds")
    g.add_argument("run_a")
    g.add_argument("run_b")
    g.add_argument("--thresholds", default=str(c.THRESHOLDS_PATH))
    g.set_defaults(func=cmd_gate)

    rp = sub.add_parser(c.CMD_REPORT, help="write an HTML comparison report")
    rp.add_argument("run_a")
    rp.add_argument("run_b")
    rp.set_defaults(func=cmd_report)

    sub.add_parser(c.CMD_HISTORY, help="trend table across scored runs (DuckDB)").set_defaults(func=cmd_history)

    jn = sub.add_parser(c.CMD_JUDGE_NOISE, help="measure judge and agent variance")
    jn.add_argument("run_id")
    jn.add_argument("--times", type=int, default=3)
    jn.add_argument("--sample", type=int, default=8)
    jn.set_defaults(func=cmd_judge_noise)

    ask = sub.add_parser(c.CMD_ASK, help="ask the agent one question")
    ask.add_argument("question")
    ask.add_argument("--config", default=str(c.CONFIGS_DIR / "baseline.yaml"))
    ask.set_defaults(func=cmd_ask)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv: Optional[List[str]]
        Arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # surface a clean message for expected failures
        from agent.llm import MissingApiKeyError
        from eval.generate import FatalRunError

        if isinstance(exc, (MissingApiKeyError, FatalRunError, FileNotFoundError, ValueError)):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
