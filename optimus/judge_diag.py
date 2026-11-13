#!/usr/bin/env python3
"""Diagnose an empty judge reply and, if the cause is the output-token cap, re-judge with a larger cap.

    python ~/judge_diag.py results/v11_full_gpu_low <task>

Step 1 repeats NatureBench's stock judge call (same model, prompt, policy, 4096-token cap) and
prints the API's status, stop reason and token usage. If it returns a verdict, that verdict is kept.
Step 2 runs only if step 1 came back empty because the cap was hit (status "incomplete",
reason max_output_tokens): it re-runs the same call with a 16384-token cap and records that
change inside judge_verdict.json for disclosure. Anything else is printed, and nothing is written.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

NB = Path.home() / "NatureBench"
RAISED_CAP = 16384


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    out_dir = NB / sys.argv[1]
    task = sys.argv[2]
    tdir = out_dir / task
    problem = NB / "data" / "naturebench_data" / "tasks" / task / "problem"
    key_file = Path.home() / ".openrouter_key"
    os.environ.setdefault("JUDGE_MODEL", "openai/gpt-5.5")
    os.environ.setdefault("JUDGE_OPENAI_BASE_URL", "https://openrouter.ai/api")
    if not os.environ.get("JUDGE_OPENAI_API_KEY") and key_file.is_file():
        os.environ["JUDGE_OPENAI_API_KEY"] = key_file.read_text().strip()
    os.chdir(NB)
    sys.path.insert(0, str(NB))
    import judge_core.clients as cl
    from judge import judge_task

    cap = [cl.MAX_JUDGE_OUTPUT_TOKENS]
    info: dict = {}

    def call(user_prompt, model=None, base_url=None, api_key=None, timeout=cl.DEFAULT_TIMEOUT):
        """Same request as NatureBench's call_openai_judge, plus diagnostics."""
        _, m, b, k = cl.resolve_judge_config(model, base_url, api_key, provider="openai")
        body = {"model": m, "instructions": cl.JUDGE_SYSTEM, "input": user_prompt, "max_output_tokens": cap[0]}
        req = urllib.request.Request(cl.build_api_url(b, "/v1/responses"), data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {k}"})
        info.clear()
        info["prompt_chars"] = len(user_prompt)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            info["error"] = f"HTTP {e.code}"
            return False, f"HTTP {e.code}: request failed"
        except Exception as e:  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"
            return False, info["error"]
        out = payload.get("output", []) if isinstance(payload, dict) else []
        text = "".join(str(c.get("text", "")) for i in out if isinstance(i, dict)
                       for c in (i.get("content") or []) if isinstance(c, dict) and c.get("type") == "output_text")
        info.update(status=payload.get("status"), incomplete=payload.get("incomplete_details"),
                    usage=payload.get("usage"), error_field=payload.get("error"),
                    items=[i.get("type") for i in out if isinstance(i, dict)], text_chars=len(text))
        return True, text

    cl.call_openai_judge = call

    def show(label: str) -> None:
        print(f"--- {label}", flush=True)
        for k, v in info.items():
            print(f"    {k}: {json.dumps(v)[:400]}")

    def save(verdict: dict) -> None:
        n = 1
        while (tdir / f"judge_verdict.error_{n}.json").exists():
            n += 1
        old = tdir / "judge_verdict.json"
        if old.is_file():
            (tdir / f"judge_verdict.error_{n}.json").write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
        old.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
        tag = "VALID" if verdict["is_valid"] else "INVALID"
        print(f"\n{task}: {tag} | {verdict.get('reason', '')[:500]}")

    print(f"{task}: step 1, stock judge call (cap {cap[0]} tokens)...", flush=True)
    v = judge_task(task, tdir, "optimus", task_problem_dir=problem)
    show("API response")
    if v.get("is_valid") in (True, False):
        save(v)
        return 0
    inc = info.get("incomplete") or {}
    if not (info.get("status") == "incomplete" and inc.get("reason") == "max_output_tokens" and not info.get("text_chars")):
        print(f"\nStill a judge error, and not from the token cap: {v.get('reason')}. Nothing written; paste this output.")
        return 1

    reasoning = ((info.get("usage") or {}).get("output_tokens_details") or {}).get("reasoning_tokens")
    print(f"\nConfirmed: the judge spent the whole {cap[0]}-token cap reasoning ({reasoning} reasoning tokens) "
          f"and returned no text. Step 2: same call with a {RAISED_CAP}-token cap...", flush=True)
    cap[0] = RAISED_CAP
    v = judge_task(task, tdir, "optimus", task_problem_dir=problem)
    show("API response")
    if v.get("is_valid") in (True, False):
        v["judge_note"] = (f"Stock judge calls (max_output_tokens {cl.MAX_JUDGE_OUTPUT_TOKENS}) returned no text: "
                           f"status incomplete, reason max_output_tokens, {reasoning} reasoning tokens. "
                           f"Re-run with the same model, prompt and policy, max_output_tokens {RAISED_CAP}.")
        save(v)
        return 0
    print(f"\nStill no verdict with the larger cap: {v.get('reason')}. Nothing written; paste this output.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
