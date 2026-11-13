#!/usr/bin/env python3
"""Assemble a NatureBench Full-track submission package from OPTIMUS results backups.

    python build_pkg.py add <results_root> <pkg_dir> <source_label> [--logs <dir>]
    python build_pkg.py replace <results_root> <pkg_dir> <source_label> <archive_name> [--logs <dir>]
    python build_pkg.py notrun <pkg_dir> <case_id> <note_file>
    python build_pkg.py finalise <pkg_dir> <case_metadata.csv>

`add` finds every task directory (named like a case id) under <results_root> and copies a lean,
review-oriented record of it into <pkg_dir>/raw-results/<case_id>/:

  result.json, submissions.jsonl, judge_verdict.json, judge_verdict.error_*.json   (unchanged)
  trajectory.jsonl          Claude Code session transcript(s), concatenated in session order
                            (the same file NatureBench's judge reads for this agent)
  trajectory.stream.jsonl   the agent process's stream-json stdout, as logged by solve.py
  trajectory.harness.log    the OPTIMUS loop log (rounds, restarts, deadline handling)
  trajectory.execution.log  only when no agent ran: the pipeline's own failure message
  harness/                  OPTIMUS config, initial prompt, Claude Code settings (hook), ledger
  workspace_code/           code and notes from the final workspace (no data or model files)
  previous_runs/<archive>/  runs archived by NatureBench --force-fresh, same layout

`replace` is for a fresh re-run of cases already in the package: the case's current record moves,
unchanged, to previous_runs/<archive_name>/ (its own previous_runs/ entries move up beside it) and
the new run takes its place.

`notrun` gives a case that was never started a directory holding the explanation in <note_file>, as
README.md and as trajectory.execution.log. No pipeline record (result.json etc.) is created.

Nothing is edited: every copied file is byte-identical to the backup. `finalise` writes
results.csv for all official cases and _build/summary.json for the README. A case whose run was
stopped before NatureBench wrote result.json gets final_status "interrupted"; a case never started
gets "not_run".
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

CASE_RE = re.compile(r"^s\d{5}-\d{3}-\d{5}-[0-9a-z]$")
CODE_EXT = {".py", ".sh", ".md", ".ipynb", ".yaml", ".yml", ".toml", ".cfg", ".r"}
SMALL_EXT = {".json", ".txt", ".log"}  # only near the top of the workspace, and small
CODE_MAX = 2_000_000
SMALL_MAX = 200_000
MAX_WS_FILES = 1500
# Third-party environments and caches the agent created inside its workspace: not its own code.
SKIP_PARTS = {"site-packages", "dist-packages", "lib", "lib64", "bin", "include", "__pycache__", ".git",
              "node_modules", ".venv", "venv", ".cache", ".optimus", "wandb", "lightning_logs", "checkpoints"}


def copy(src: Path, dst: Path) -> bool:
    if not src.is_file() or src.stat().st_size == 0:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def session_files(run: Path) -> list[Path]:
    sids: list[str] = []
    sf = run / ".optimus" / "sessions.txt"
    if sf.is_file():
        for line in sf.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s and s not in sids:
                sids.append(s)
    proj = run / ".claude_state" / "projects"
    by_stem = {p.stem: p for p in proj.rglob("*.jsonl")} if proj.is_dir() else {}
    files = [by_stem[s] for s in sids if s in by_stem]
    # Sessions not listed in sessions.txt (should not happen) are appended by name.
    files += sorted(p for stem, p in by_stem.items() if stem not in sids)
    return files


def lean_copy(run: Path, out: Path, runner_log: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    info: dict = {"files": []}
    for name in ["result.json", "submissions.jsonl", "judge_verdict.json"]:
        if copy(run / name, out / name):
            info["files"].append(name)
    for p in sorted(run.glob("judge_verdict.error_*.json")):
        copy(p, out / p.name)
        info["files"].append(p.name)
    sess = session_files(run)
    if sess:
        with open(out / "trajectory.jsonl", "wb") as fh:
            for p in sess:
                fh.write(p.read_bytes().rstrip(b"\n") + b"\n")
        info["sessions"] = len(sess)
    if copy(run / "optimus.jsonl", out / "trajectory.stream.jsonl"):
        info["files"].append("trajectory.stream.jsonl")
    if copy(run / ".optimus" / "loop.log", out / "trajectory.harness.log"):
        info["files"].append("trajectory.harness.log")
    if not list(out.glob("trajectory.*")):
        lines = []
        rj = run / "result.json"
        if rj.is_file():
            try:
                r = json.loads(rj.read_text(encoding="utf-8"))
                lines.append(f"status: {r.get('status')}")
                lines.append(f"message: {r.get('message')}")
            except ValueError:
                pass
        lines += runner_log
        if lines:
            (out / "trajectory.execution.log").write_text(
                "No agent session ran for this case. NatureBench's own records of the run follow.\n\n"
                + "\n".join(lines) + "\n", encoding="utf-8")
            info["files"].append("trajectory.execution.log")
    h = out / "harness"
    for src, dst in [(".optimus/config.json", "optimus_config.json"), (".optimus/initial_prompt.md", "initial_prompt.md"),
                     (".optimus/sessions.txt", "sessions.txt"), (".optimus/steer_state.json", "steer_state.json"),
                     (".claude_state/settings.json", "claude_code_settings.json"), ("optimus.err", "agent_stderr.log"),
                     ("workspace/.optimus/ledger.jsonl", "ledger.jsonl"), ("workspace/optimus_ledger.md", "optimus_ledger.md"),
                     ("workspace/optimus_assembly.md", "optimus_assembly.md")]:
        copy(run / src, h / dst)
    ws = run / "workspace"
    if ws.is_dir():
        n = 0
        for p in sorted(ws.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ws)
            if any(part in SKIP_PARTS or part.startswith("python3") for part in rel.parts[:-1]):
                continue
            ext = p.suffix.lower()
            size = p.stat().st_size
            if size == 0 or size > CODE_MAX:
                continue
            if ext in CODE_EXT or (ext in SMALL_EXT and len(rel.parts) <= 2 and size < SMALL_MAX):
                copy(p, out / "workspace_code" / rel)
                n += 1
                if n >= MAX_WS_FILES:
                    info["workspace_code_truncated"] = True
                    break
        info["workspace_code_files"] = n
    return info


def runner_lines(logs: Path | None, case: str) -> list[str]:
    if not logs or not logs.is_dir():
        return []
    hits = []
    for lf in sorted(logs.rglob("*.log")):
        for line in lf.read_text(encoding="utf-8", errors="replace").splitlines():
            if case in line and any(k in line for k in ("ERROR", "Error", "error", "status=", "force-fresh", "Setup")):
                hits.append(f"[{lf.name}] {line.strip()[:2000]}")
    return hits[-40:]


def cmd_add(root: Path, pkg: Path, label: str, logs: Path | None) -> int:
    raw = pkg / "raw-results"
    meta_dir = pkg / "_build" / "sources"
    meta_dir.mkdir(parents=True, exist_ok=True)
    found = 0
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and CASE_RE.match(p.name)):
        if "_force_fresh_archive_" in str(d.relative_to(root)):
            continue
        case = d.name
        if (meta_dir / f"{case}.json").exists():
            print(f"!! duplicate case {case} from {label}: kept the earlier source", file=sys.stderr)
            continue
        out = raw / case
        info = lean_copy(d, out, runner_lines(logs, case))
        archives = []
        for a in sorted(d.glob("_force_fresh_archive_*")):
            if a.is_dir():
                lean_copy(a, out / "previous_runs" / a.name, [])
                archives.append(a.name)
        info.update(source=label, results_dir=str(d.relative_to(root)), previous_runs=archives)
        (meta_dir / f"{case}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        found += 1
    print(f"{label}: {found} cases added")
    return 0


def cmd_replace(root: Path, pkg: Path, label: str, archive: str, logs: Path | None) -> int:
    raw = pkg / "raw-results"
    meta_dir = pkg / "_build" / "sources"
    found = 0
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and CASE_RE.match(p.name)):
        if "_force_fresh_archive_" in str(d.relative_to(root)):
            continue
        case = d.name
        out = raw / case
        meta = meta_dir / f"{case}.json"
        if not out.is_dir() or not meta.exists():
            print(f"!! {case} is not in the package yet; use add", file=sys.stderr)
            continue
        old_info = json.loads(meta.read_text(encoding="utf-8"))
        hold = raw / f".{case}.replacing"
        if hold.exists():
            print(f"!! {hold} exists from an interrupted replace; resolve it first", file=sys.stderr)
            return 1
        out.rename(hold)
        info = lean_copy(d, out, runner_lines(logs, case))
        prev = out / "previous_runs"
        archives = []
        if (hold / "previous_runs").is_dir():
            for a in sorted((hold / "previous_runs").iterdir()):
                (prev).mkdir(parents=True, exist_ok=True)
                a.rename(prev / a.name)
                archives.append(a.name)
            (hold / "previous_runs").rmdir()
        prev.mkdir(parents=True, exist_ok=True)
        hold.rename(prev / archive)
        archives.append(archive)
        for a in sorted(d.glob("_force_fresh_archive_*")):
            if a.is_dir():
                lean_copy(a, prev / a.name, [])
                archives.append(a.name)
        info.update(source=label, results_dir=str(d.relative_to(root)), previous_runs=archives,
                    replaced={"archive": archive, "source": old_info.get("source"),
                              "results_dir": old_info.get("results_dir")})
        meta.write_text(json.dumps(info, indent=1), encoding="utf-8")
        found += 1
        print(f"{case}: replaced; earlier run kept in previous_runs/{archive}")
    print(f"{label}: {found} cases replaced")
    return 0


def cmd_notrun(pkg: Path, case: str, note: Path) -> int:
    if not CASE_RE.match(case):
        print(f"not a case id: {case}", file=sys.stderr)
        return 1
    out = pkg / "raw-results" / case
    meta = pkg / "_build" / "sources" / f"{case}.json"
    if meta.exists():
        print(f"!! {case} already has a run record in the package", file=sys.stderr)
        return 1
    text = note.read_text(encoding="utf-8").strip() + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(text, encoding="utf-8")
    (out / "trajectory.execution.log").write_text(
        "No agent session ran for this case and NatureBench wrote no result.json. Explanation follows.\n\n" + text,
        encoding="utf-8")
    print(f"{case}: not-run note written")
    return 0


def finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def cmd_finalise(pkg: Path, metadata: Path) -> int:
    with open(metadata, newline="", encoding="utf-8") as fh:
        cases = [(r["case_id"].strip(), r["domain"].strip()) for r in csv.DictReader(fh)]
    rows, summary = [], {"cases": {}, "missing": []}
    for case, domain in cases:
        d = pkg / "raw-results" / case
        src = pkg / "_build" / "sources" / f"{case}.json"
        if not src.exists():
            summary["missing"].append(case)
            rows.append(dict(case_id=case, final_status="not_run", best_score="", judge_verdict="not_applicable",
                             judge_reason="not_applicable"))
            continue
        if (d / "result.json").is_file():
            status = json.loads((d / "result.json").read_text(encoding="utf-8")).get("status")
        else:
            status = "interrupted"  # run stopped before NatureBench's solve.py wrote result.json
        best = None
        if (d / "submissions.jsonl").is_file():
            for line in (d / "submissions.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    v = finite(json.loads(line).get("aggregate_improvement"))
                except ValueError:
                    continue
                if v is not None and (best is None or v > best):
                    best = v
        verdict, reason, jmodel, jnote = "not_applicable", "not_applicable", None, None
        if (d / "judge_verdict.json").is_file():
            jv = json.loads((d / "judge_verdict.json").read_text(encoding="utf-8"))
            jmodel, jnote = jv.get("model"), jv.get("judge_note")
            if jv.get("is_valid") in (True, False):
                verdict = "valid" if jv["is_valid"] else "invalid"
                reason = str(jv.get("reason", "")).strip()
            elif best is not None:
                verdict, reason = "JUDGE_ERROR", str(jv.get("reason", "")).strip()
        rows.append(dict(case_id=case, final_status=status, best_score="" if best is None else repr(best),
                         judge_verdict=verdict, judge_reason=reason))
        src_info = json.loads(src.read_text(encoding="utf-8"))
        summary["cases"][case] = dict(domain=domain, status=status, best=best, verdict=verdict, judge_model=jmodel,
                                     judge_note=jnote, source=src_info.get("source"),
                                     previous_runs=src_info.get("previous_runs", []),
                                     replaced=src_info.get("replaced"),
                                     judge_error_files=sorted(p.name for p in d.glob("judge_verdict.error_*.json")))
    with open(pkg / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "final_status", "best_score", "judge_verdict", "judge_reason"])
        w.writeheader()
        w.writerows(rows)
    vals = summary["cases"].values()
    summary["counts"] = dict(
        official=len(cases), with_records=len(summary["cases"]), missing=len(summary["missing"]),
        interrupted=sum(1 for c in vals if c["status"] == "interrupted"),
        scored=sum(1 for c in vals if c["best"] is not None),
        valid_scored=sum(1 for c in vals if c["best"] is not None and c["verdict"] == "valid"),
        invalid=sum(1 for c in vals if c["verdict"] == "invalid"),
        judge_errors=sum(1 for c in vals if c["verdict"] == "JUDGE_ERROR"),
        match=sum(1 for c in vals if c["verdict"] == "valid" and c["best"] is not None and c["best"] >= 0),
        surpass=sum(1 for c in vals if c["verdict"] == "valid" and c["best"] is not None and c["best"] > 0.1))
    (pkg / "_build" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["counts"]))
    print("missing:", " ".join(summary["missing"]) or "none")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 4 and a[0] == "add":
        logs = Path(a[a.index("--logs") + 1]) if "--logs" in a else None
        sys.exit(cmd_add(Path(a[1]), Path(a[2]), a[3], logs))
    if len(a) >= 5 and a[0] == "replace":
        logs = Path(a[a.index("--logs") + 1]) if "--logs" in a else None
        sys.exit(cmd_replace(Path(a[1]), Path(a[2]), a[3], a[4], logs))
    if len(a) == 4 and a[0] == "notrun":
        sys.exit(cmd_notrun(Path(a[1]), a[2], Path(a[3])))
    if len(a) == 3 and a[0] == "finalise":
        sys.exit(cmd_finalise(Path(a[1]), Path(a[2])))
    print(__doc__)
    sys.exit(1)
