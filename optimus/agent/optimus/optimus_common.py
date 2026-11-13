"""Shared helpers for the OPTIMUS in-container tools.

Stdlib only: these run inside the NatureBench task container with whatever
``python3`` is on PATH. Paths default to the container layout and can be
overridden with environment variables so the tools can be tested outside
Docker.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE = Path(os.environ.get("OPTIMUS_WORKSPACE", "/workspace"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKSPACE / "output")))
STATE_DIR = WORKSPACE / ".optimus"
# "outputs" is in the judge's SOURCE_SKIP_DIRS, so snapshots never crowd out
# the agent's real source files in the judge's evidence budget.
SNAP_DIR = STATE_DIR / "outputs"
LEDGER_JSONL = STATE_DIR / "ledger.jsonl"
LEDGER_MD = WORKSPACE / "optimus_ledger.md"

SNAP_MAX_BYTES = int(float(os.environ.get("OPTIMUS_SNAP_MAX_GB", "2")) * 1e9)
SNAP_MIN_FREE_BYTES = int(float(os.environ.get("OPTIMUS_SNAP_MIN_FREE_GB", "25")) * 1e9)
SURPASS = 0.10


def eval_url() -> str:
    url = os.environ.get("EVAL_SERVICE_URL", "").rstrip("/")
    if not url:
        raise SystemExit("[optimus] EVAL_SERVICE_URL is not set")
    return url


def eval_token() -> str:
    tok = os.environ.get("OPTIMUS_EVAL_TOKEN", "")
    if not tok:
        raise SystemExit("[optimus] OPTIMUS_EVAL_TOKEN is not set")
    return tok


HTTP_TIMEOUT = float(os.environ.get("OPTIMUS_HTTP_TIMEOUT", 30))


def _get(path: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    timeout = HTTP_TIMEOUT if timeout is None else timeout
    sep = "&" if "?" in path else "?"
    url = f"{eval_url()}{path}{sep}eval_token={eval_token()}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def time_remaining() -> Optional[Dict[str, Any]]:
    return _get("/time_remaining")


def best_score() -> Optional[Dict[str, Any]]:
    return _get("/best_score")


def post_evaluate(timeout: float = 7200) -> Dict[str, Any]:
    body = json.dumps({"eval_token": eval_token()}).encode("utf-8")
    req = urllib.request.Request(
        f"{eval_url()}/evaluate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            return json.loads(err.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {"error": f"HTTP {err.code}"}
    except (urllib.error.URLError, OSError, ValueError) as err:
        return {"error": str(err)}


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def disk_free(path: Path = WORKSPACE) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return -1


def gb(n: int) -> str:
    return "?" if n < 0 else f"{n / 1e9:.1f} GB"


def fmt_g(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:+.4f}"


def read_ledger() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if LEDGER_JSONL.exists():
        for line in LEDGER_JSONL.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def append_ledger(row: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    new = not LEDGER_MD.exists()
    with open(LEDGER_MD, "a", encoding="utf-8") as fh:
        if new:
            fh.write("# OPTIMUS experiment ledger\n\n")
            fh.write("| attempt | elapsed (min) | g | family | note |\n|---|---|---|---|---|\n")
        note = str(row.get("note", "")).replace("|", "/").replace("\n", " ")
        fam = str(row.get("family", "")).replace("|", "/")
        fh.write(
            f"| {row.get('attempt')} | {row.get('elapsed_min', '?')} | "
            f"{fmt_g(row.get('g'))} | {fam} | {note} |\n"
        )


def snapshot_path(attempt: int) -> Path:
    return SNAP_DIR / f"attempt_{attempt:04d}"


def keep_set(rows: List[Dict[str, Any]]) -> set:
    """Attempts whose snapshots are worth keeping: best overall + best per instance."""
    keep = set()
    scored = [r for r in rows if r.get("g") is not None and r.get("snapshot")]
    if not scored:
        return keep
    keep.add(max(scored, key=lambda r: r["g"])["attempt"])
    per: Dict[str, tuple] = {}
    for r in scored:
        for inst, v in (r.get("per_instance") or {}).items():
            if v is None:
                continue
            if inst not in per or v > per[inst][0]:
                per[inst] = (v, r["attempt"])
    keep.update(a for _v, a in per.values())
    return keep


def prune_snapshots(rows: List[Dict[str, Any]]) -> None:
    if not SNAP_DIR.exists():
        return
    keep = keep_set(rows)
    for child in SNAP_DIR.iterdir():
        if not child.name.startswith("attempt_"):
            continue
        try:
            num = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        if num not in keep:
            shutil.rmtree(child, ignore_errors=True)


def elapsed_minutes() -> Optional[float]:
    tr = time_remaining()
    if not tr:
        return None
    try:
        return round(float(tr.get("elapsed_seconds", 0)) / 60.0, 1)
    except (TypeError, ValueError):
        return None


def status_lines(max_ledger: int = 10) -> List[str]:
    lines: List[str] = []
    tr = time_remaining() or {}
    bs = best_score() or {}
    rem = tr.get("remaining_seconds")
    tot = tr.get("timeout_seconds")
    if rem is not None:
        lines.append(f"time remaining: {float(rem) / 60:.1f} min of {float(tot or 0) / 60:.0f} min")
    best = bs.get("best_aggregate_improvement")
    lines.append(
        f"best g: {fmt_g(best)} (attempt {bs.get('best_attempt')}, "
        f"{bs.get('total_attempts', 0)} attempts total); surpass threshold +{SURPASS:.2f}"
        + ("" if best is None else f", gap {SURPASS - best:+.4f}" if best <= SURPASS else ", ABOVE threshold")
    )
    per = bs.get("best_per_instance_improvement") or {}
    if per:
        ordered = sorted(per.items(), key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else 0))
        lines.append("per-instance g at best attempt (weakest first): " + ", ".join(
            f"{k}={fmt_g(v)}" for k, v in ordered))
    rows = read_ledger()
    if rows:
        best_inst: Dict[str, tuple] = {}
        for r in rows:
            for inst, v in (r.get("per_instance") or {}).items():
                if v is not None and (inst not in best_inst or v > best_inst[inst][0]):
                    best_inst[inst] = (v, r["attempt"])
        if len(best_inst) > 1:
            mean_best = sum(v for v, _a in best_inst.values()) / len(best_inst)
            lines.append(
                f"best-per-instance across logged attempts would give g ~ {mean_best:+.4f} "
                f"(see optimus-assemble)"
            )
        lines.append(f"ledger (last {min(max_ledger, len(rows))} of {len(rows)}):")
        for r in rows[-max_ledger:]:
            lines.append(f"  #{r.get('attempt')} t={r.get('elapsed_min')}m g={fmt_g(r.get('g'))} "
                         f"[{r.get('family', '?')}] :: {r.get('note', '')}")
    lines.append(f"disk free: {gb(disk_free())}")
    return lines


def now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Steering (OPTIMUS 1.1)
# ---------------------------------------------------------------------------
# Run 1 of OPTIMUS showed agents that used the whole budget but spent hours
# hill-climbing one stalled method (e.g. +0.003 over three hours, finishing at
# +0.098 while another method had reached +0.215 in 16 minutes). Steering is
# now delivered *during* the work: in every optimus-eval result, in a timed
# PostToolUse hook, and in resume prompts.

PLATEAU_WINDOW_MIN = float(os.environ.get("OPTIMUS_PLATEAU_WINDOW_MIN", 45))
PLATEAU_MIN_GAIN = float(os.environ.get("OPTIMUS_PLATEAU_MIN_GAIN", 0.005))
MIN_FAMILIES = int(os.environ.get("OPTIMUS_MIN_FAMILIES", 3))
EXPLORE_DEADLINE_FRAC = 0.33   # by ~80 min of 240, MIN_FAMILIES should be scored
FINAL_MIN = float(os.environ.get("OPTIMUS_FINAL_MIN", 15))
STEER_STATE = Path(os.environ.get("OPTIMUS_STATE", "/opt/optimus-state")) / "steer_state.json"


def _families(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    fams: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        fam = (r.get("family") or "untagged").strip() or "untagged"
        g = r.get("g")
        info = fams.setdefault(fam, {"best": None, "n": 0, "first": r.get("elapsed_min"), "last": r.get("elapsed_min")})
        info["n"] += 1
        info["last"] = r.get("elapsed_min")
        if g is not None and (info["best"] is None or g > info["best"]):
            info["best"] = g
    return fams


def _best_by_time(rows: List[Dict[str, Any]], t_max: float) -> Optional[float]:
    vals = [r["g"] for r in rows
            if r.get("g") is not None and r.get("elapsed_min") is not None and r["elapsed_min"] <= t_max]
    return max(vals) if vals else None


def steering(rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Compute the phase and any directives from the ledger and the clock."""
    rows = read_ledger() if rows is None else rows
    tr = time_remaining() or {}
    try:
        total = float(tr.get("timeout_seconds") or 14400) / 60.0
        rem = float(tr.get("remaining_seconds")) / 60.0
        elapsed = float(tr.get("elapsed_seconds") or 0) / 60.0
    except (TypeError, ValueError):
        total, rem, elapsed = 240.0, None, None
    bs = best_score() or {}
    best = bs.get("best_aggregate_improvement")
    fams = _families(rows)
    tagged = {k: v for k, v in fams.items() if k != "untagged"}
    out: Dict[str, Any] = {"phase": "?", "directives": [], "elapsed": elapsed, "remaining": rem,
                           "best": best, "families": fams}
    if rem is None or elapsed is None:
        return out
    frac = elapsed / total if total else 0.0
    if rem <= FINAL_MIN:
        phase = "FINAL"
    elif frac < 0.45:
        phase = "EXPLORE"
    elif frac < 0.80:
        phase = "EXPLOIT"
    else:
        phase = "CONSOLIDATE"
    out["phase"] = phase
    d: List[str] = out["directives"]

    if best is None:
        d.append("NO VALID SCORE YET: produce any valid, data-fitted output and run optimus-eval now.")
        return out

    # Diversity: too few method families tried.
    if phase in ("EXPLORE", "EXPLOIT") and len(tagged) < MIN_FAMILIES and frac >= 0.15:
        late = frac >= EXPLORE_DEADLINE_FRAC
        d.append(
            f"DIVERSITY{' (overdue)' if late else ''}: only {len(tagged)} method famil"
            f"{'y' if len(tagged) == 1 else 'ies'} scored so far "
            f"({', '.join(tagged) or 'none tagged'}). At least {MIN_FAMILIES} genuinely different "
            "families should be scored early. Your next experiment must be a new family "
            "(tag it with --family)."
        )

    # Plateau: best barely moved over the recent window.
    if elapsed >= 50 and phase != "FINAL":
        past = _best_by_time(rows, elapsed - PLATEAU_WINDOW_MIN)
        recent = [r for r in rows if r.get("elapsed_min") is not None and r["elapsed_min"] > elapsed - PLATEAU_WINDOW_MIN]
        if past is not None and len(recent) >= 3 and best - past < PLATEAU_MIN_GAIN:
            recent_fams = sorted({(r.get("family") or "untagged") for r in recent})
            if phase == "CONSOLIDATE":
                d.append(
                    f"PLATEAU: best moved only {best - past:+.4f} in the last {PLATEAU_WINDOW_MIN:.0f} min. "
                    "Stop small parameter variations; spend the remaining time on combining your strongest "
                    "families (ensembles, stacking, per-instance choice)."
                )
            else:
                d.append(
                    f"PLATEAU: best moved only {best - past:+.4f} in the last {PLATEAU_WINDOW_MIN:.0f} min "
                    f"over {len(recent)} attempts (families: {', '.join(recent_fams)}). Stop tuning these. "
                    "Your next two experiments must be method families you have not tried, chosen because they "
                    "could beat the current best by a wide margin, not by a few thousandths."
                )

    if 0.0 <= best <= SURPASS and phase in ("EXPLOIT", "CONSOLIDATE"):
        d.append(
            f"THRESHOLD: you are {SURPASS - best:.4f} below the +{SURPASS:.2f} surpass line. Matching SOTA is not "
            "a win. A different family, or an ensemble across families, is more likely to cross it than "
            "further tuning."
        )
    if phase == "FINAL":
        d.append(
            "FINAL: do not start anything that cannot finish and be evaluated in time. With 2+ instances, "
            "run `optimus-assemble --dry-run` and then `optimus-assemble` if it predicts a gain. Keep run.py "
            "reproducing your best pipeline."
        )
    return out


def steering_lines(st: Optional[Dict[str, Any]] = None) -> List[str]:
    st = steering() if st is None else st
    lines: List[str] = []
    if st.get("remaining") is not None:
        lines.append(
            f"[OPTIMUS] phase {st['phase']} | {st['elapsed']:.0f} min elapsed, {st['remaining']:.0f} min left "
            f"| best g {fmt_g(st.get('best'))} (surpass line +{SURPASS:.2f})"
        )
    fams = {k: v for k, v in st.get("families", {}).items()}
    if fams:
        ranked = sorted(fams.items(), key=lambda kv: -(kv[1]['best'] if kv[1]['best'] is not None else -9))
        shown = "; ".join(f"{k} best {fmt_g(v['best'])} ({v['n']} attempts)" for k, v in ranked[:6])
        more = f"; +{len(ranked) - 6} more" if len(ranked) > 6 else ""
        lines.append(f"[OPTIMUS] families scored ({len(ranked)}): {shown}{more}")
    for x in st.get("directives", []):
        lines.append(f"[OPTIMUS] {x}")
    return lines


def mark_steered() -> None:
    try:
        STEER_STATE.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if STEER_STATE.exists():
            data = json.loads(STEER_STATE.read_text(encoding="utf-8") or "{}")
        data["last_steer"] = time.time()
        STEER_STATE.write_text(json.dumps(data), encoding="utf-8")
    except (OSError, ValueError):
        pass


def last_steered() -> float:
    try:
        return float(json.loads(STEER_STATE.read_text(encoding="utf-8")).get("last_steer", 0))
    except (OSError, ValueError, TypeError):
        return 0.0
