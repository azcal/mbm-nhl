import requests, json, os, sys, time, pandas as pd
from concurrent.futures import ThreadPoolExecutor
H = {"User-Agent": "Mozilla/5.0"}
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 300
g = pd.read_csv("nhl/games.csv")
done = set()
if os.path.exists("nhl/goalies.jsonl"):
    for line in open("nhl/goalies.jsonl"):
        try: done.add(json.loads(line)["gid"])
        except Exception: pass
todo = [int(x) for x in g.game_id if int(x) not in done]
print("remaining:", len(todo), flush=True)
out = open("nhl/goalies.jsonl", "a")
t0 = time.time()
def toi_sec(t):
    try: m, s = str(t).split(":"); return int(m)*60 + int(s)
    except Exception: return 0
def pull(gid):
    try:
        r = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore", headers=H, timeout=10).json()
        pb = r.get("playerByGameStats", {})
        rec = {"gid": gid}
        for side in ("homeTeam", "awayTeam"):
            gl = []
            for p in pb.get(side, {}).get("goalies", []):
                ss = str(p.get("saveShotsAgainst", "0/0")).split("/")
                gl.append([p.get("playerId"), toi_sec(p.get("toi")), int(ss[0] or 0), int(ss[1] or 0)])
            rec[side[0]] = gl
        return rec
    except Exception:
        return None
n = 0
with ThreadPoolExecutor(max_workers=12) as ex:
    for rec in ex.map(pull, todo):
        if rec: out.write(json.dumps(rec) + "\n"); n += 1
        if n % 200 == 0: out.flush()
        if time.time() - t0 > BUDGET:
            print("budget hit after", n); break
out.close()
print("chunk:", n, "in", round(time.time()-t0, 1), "s")
