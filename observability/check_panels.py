"""Query every panel through Grafana exactly as the browser would."""

import json
import subprocess
import urllib.request

DASH = json.loads(
    subprocess.run(
        ["curl", "-s", "http://127.0.0.1:3000/api/dashboards/uid/pipeline-vitals"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
)["dashboard"]


def query(panel):
    t = panel["targets"][0]
    ds = panel["datasource"]
    if ds["type"] == "prometheus":
        q = {"refId": "A", "datasource": ds, "expr": t["expr"], "instant": True}
    else:
        q = {
            "refId": "A",
            "datasource": ds,
            "format": t.get("format", "table"),
            "rawSql": t["rawSql"],
            "rawQuery": True,
        }
    body = json.dumps({"queries": [q], "from": "now-6h", "to": "now"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:3000/api/ds/query",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            res = json.load(r)["results"]["A"]
    except Exception as exc:
        return "ERROR", str(exc)[:50]
    if res.get("error"):
        return "ERROR", res["error"][:60]
    frames = res.get("frames", [])
    rows = sum(len(f.get("data", {}).get("values", [[]])[0]) for f in frames if f.get("data"))
    return ("ok" if rows else "EMPTY"), f"{rows} rows"


bad = 0
for p in DASH["panels"]:
    if p["type"] == "row" or not p.get("targets"):
        continue
    status, detail = query(p)
    if status != "ok":
        bad += 1
    mark = "  " if status == "ok" else "->"
    print(f"{mark} {status:<6} {p['title'][:34]:<36} {detail}")
print(f"\n{bad} panel(s) not returning data")
