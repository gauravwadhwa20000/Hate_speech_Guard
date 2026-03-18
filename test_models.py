"""Analyze eval cache for failures."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
data = json.loads(open(r"c:\Users\v-gawadh\hate-speech-guardian\evals\cache.json").read())
print(f"Total cached: {len(data)} cases\n")
for r in data:
    fails = []
    if r.get("verdict_correct") == False: fails.append("VERDICT")
    if r.get("category_correct") == False: fails.append("CATEGORY")
    if r.get("severity_in_range") == False: fails.append("SEVERITY")
    if r.get("json_valid") == False: fails.append("JSON")
    if r.get("rewrite_ok") == False: fails.append("REWRITE")
    if r.get("error"): fails.append("ERROR: " + r["error"][:60])
    status = ", ".join(fails) if fails else "PASS"
    cid = r["id"]
    verdict = r["expected_verdict"]
    text = r["text"][:55]
    print(f"  #{cid:2d} [{verdict:6s}] {status}  -- {text}")
