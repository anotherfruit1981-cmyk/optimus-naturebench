#!/usr/bin/env python3
"""Progress of a NatureBench results folder:  python3 ~/status.py [results-name]

Defaults to the most recently modified folder under ~/NatureBench/results.
"""
import json
import os
import subprocess
import sys

root = os.path.expanduser("~/NatureBench/results")
if len(sys.argv) > 1:
    name = sys.argv[1]
else:
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    name = max(dirs, key=lambda d: os.path.getmtime(os.path.join(root, d)))
base = os.path.join(root, name)
prefix = f"naturebench-solve-{name}-"
names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()
running = {n[len(prefix):] for n in names if n.startswith(prefix)}

print(f"results/{name}")
counts = {"surpass": 0, "match": 0, "invalid": 0}
for t in sorted(os.listdir(base)):
    d = os.path.join(base, t)
    if not os.path.isdir(d):
        continue
    st = "RUNNING" if t in running else "waiting"
    r = os.path.join(d, "result.json")
    if t not in running and os.path.exists(r):
        try:
            st = json.load(open(r)).get("status", "?")
        except Exception:
            pass
    best, n = None, 0
    p = os.path.join(d, "submissions.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                v = json.loads(line).get("aggregate_improvement")
            except Exception:
                continue
            n += 1
            if v is not None and (best is None or v > best):
                best = v
    verdict = ""
    jv = os.path.join(d, "judge_verdict.json")
    if os.path.exists(jv):
        try:
            ok = json.load(open(jv)).get("is_valid")
            verdict = " [valid]" if ok else (" [INVALID]" if ok is False else " [judge error]")
            if ok is False:
                counts["invalid"] += 1
        except Exception:
            pass
    tag = ""
    if best is not None and not verdict.startswith(" [INVALID"):
        if best > 0.1:
            tag = " SURPASS"
            counts["surpass"] += 1
            counts["match"] += 1
        elif best >= 0:
            tag = " match"
            counts["match"] += 1
    rounds = ""
    loop_log = os.path.join(d, ".optimus", "loop.log")
    if os.path.exists(loop_log):
        rounds = f" rounds={sum(1 for l in open(loop_log) if ' ended: ' in l)}"
    print(f"{t:22} {st:12} attempts={n:<3}{rounds:10} best={'-' if best is None else f'{best:+.3f}'}{tag}{verdict}")
print(f"\n{len(running)} agents running | surpass {counts['surpass']} | match {counts['match']} | invalid {counts['invalid']}")
os.system("df -h / | tail -1")
