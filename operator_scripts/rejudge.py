#!/usr/bin/env python3
"""Re-run NatureBench's validity judge on tasks whose verdict is a judge ERROR.

    python ~/rejudge.py results/v11_full_gpu_low [--dry-run]

Only verdicts with is_valid = null (judge_error: API failure, bad output, ...) are
re-run. A valid or INVALID verdict is never touched. The failed verdict is kept
next to the new one as judge_verdict.error_<n>.json, for disclosure.
Uses the same judge model, endpoint and key as run_optimus.sh.
"""
import json
import os
import sys
from pathlib import Path

NB = Path.home() / "NatureBench"
TASKS = NB / "data" / "naturebench_data" / "tasks"


def targets(out_dir: Path):
    for d in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        jv = d / "judge_verdict.json"
        if not jv.is_file() or not (d / "submissions.jsonl").is_file():
            continue
        try:
            v = json.loads(jv.read_text(encoding="utf-8"))
        except ValueError:
            v = {"is_valid": None, "reason": "unreadable judge_verdict.json"}
        if v.get("is_valid") is None:
            yield d, v


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if not args:
        print(__doc__)
        return 1
    out_dir = (NB / args[0]).resolve() if not Path(args[0]).is_absolute() else Path(args[0])
    todo = list(targets(out_dir))
    if not todo:
        print(f"No judge errors in {out_dir}.")
        return 0
    for d, v in todo:
        print(f"{d.name}: judge error -> {str(v.get('reason'))[:300]}")
    if dry:
        return 0

    key_file = Path.home() / ".openrouter_key"
    os.environ.setdefault("JUDGE_MODEL", "openai/gpt-5.5")
    os.environ.setdefault("JUDGE_OPENAI_BASE_URL", "https://openrouter.ai/api")
    if not os.environ.get("JUDGE_OPENAI_API_KEY") and key_file.is_file():
        os.environ["JUDGE_OPENAI_API_KEY"] = key_file.read_text().strip()
    os.chdir(NB)
    sys.path.insert(0, str(NB))
    from judge import judge_task  # noqa: E402  (NatureBench's own judge)

    for d, old in todo:
        problem = TASKS / d.name / "problem"
        for f in ("README.md", "data_description.md"):
            if not (problem / f).is_file():
                print(f"{d.name}: cannot re-judge, {problem / f} is missing")
                break
        else:
            n = 1
            while (d / f"judge_verdict.error_{n}.json").exists():
                n += 1
            (d / f"judge_verdict.error_{n}.json").write_text(json.dumps(old, indent=2), encoding="utf-8")
            print(f"{d.name}: re-judging...", flush=True)
            new = judge_task(d.name, d, "optimus", task_problem_dir=problem)
            (d / "judge_verdict.json").write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
            tag = {True: "VALID", False: "INVALID"}.get(new.get("is_valid"), "judge error again")
            print(f"{d.name}: {tag} | {str(new.get('reason'))[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
