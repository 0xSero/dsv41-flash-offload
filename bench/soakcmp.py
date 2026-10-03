import json, statistics as st, sys
for name in sys.argv[1:]:
    rows = [json.loads(l) for l in open(name) if l.startswith("{")]
    out = [name.split("/")[-2], f"n={len(rows)}"]
    for k in ["short", "mid", "long"]:
        r = [x for x in rows if x["kind"] == k]
        if r:
            d = [x["dec"] for x in r if x["dec"]]
            out.append(f"{k}: n{len(r)} ttft {round(st.median([x['ttft'] for x in r]), 1)} dec {round(st.median(d), 2) if d else None}")
    print(" | ".join(out))
