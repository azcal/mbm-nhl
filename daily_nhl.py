"""
MBM NHL daily runner (rebuild v2, frozen spec).
- Refreshes current-season schedule/results + goalie boxscores from the NHL API
- Projects starters (most-used goalie over last 10 team games) into pseudo rows
- Boards upcoming games (2-day horizon), winner head only (totals benched)
- v2: winner head is the three-way regulation/OT decomposition (see engine_nhl.fit_decomp).
  The v1 binary is still fitted and logged as p_home_binary for side-by-side grading.
- CALIBRATION GATE: value lines suppressed until EVERY team has 9+ GP
- Grades logged predictions (no pushes: OT/SO included), last-300 tripwire
Usage: python daily_nhl.py [--date YYYY-MM-DD]
"""
import argparse, json, math, os, sys, datetime as dt
import numpy as np, pandas as pd, requests
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_nhl as E

H = {"User-Agent": "Mozilla/5.0"}
API = "https://api-web.nhle.com/v1"
GAMES = os.path.join(HERE, "data", "games.csv")
GOALIES = os.path.join(HERE, "data", "goalies.jsonl")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C = SPEC["cfg"], SPEC["C"]
TRAIN_LO, TRAIN_HI = 2018, 2025
CAL_MIN_GP = 9
MIN_EDGE = 0.05
TEAMS = ["ANA","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM","FLA","LAK","MIN","MTL",
         "NSH","NJD","NYI","NYR","OTT","PHI","PIT","SJS","SEA","STL","TBL","TOR","UTA","VAN","VGK","WPG","WSH"]

def min_odds(p):
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + MIN_EDGE) / p * 100) / 100.0

def season_code(d):
    return d.year if d.month >= 8 else d.year - 1

def refresh_games(season, today):
    g = pd.read_csv(GAMES, parse_dates=["date"])
    def pull(tm):
        try:
            return requests.get(f"{API}/club-schedule-season/{tm}/{season}{season+1}",
                                headers=H, timeout=30).json().get("games", [])
        except Exception:
            return []
    rows = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for gs in ex.map(pull, TEAMS):
            for x in gs:
                if x.get("gameType") not in (2, 3):
                    continue
                rows[x["id"]] = dict(game_id=x["id"], date=x.get("gameDate"), gtype=x["gameType"],
                    home=E.REN.get(x["homeTeam"]["abbrev"], x["homeTeam"]["abbrev"]),
                    away=E.REN.get(x["awayTeam"]["abbrev"], x["awayTeam"]["abbrev"]),
                    hs=x["homeTeam"].get("score"), as_=x["awayTeam"].get("score"),
                    endtype=(x.get("gameOutcome") or {}).get("lastPeriodType"),
                    neutral=bool(x.get("neutralSite")), utc=x.get("startTimeUTC"),
                    season=season)
    cur = pd.DataFrame(rows.values())
    done, up = pd.DataFrame(), pd.DataFrame()
    if len(cur):
        cur["date"] = pd.to_datetime(cur.date)
        done = cur[cur.hs.notna() & cur.as_.notna()]
        up = cur[cur.hs.isna() & (cur.date >= today) & (cur.date <= today + pd.Timedelta(days=2))]
        if len(done):
            g = pd.concat([g[~g.game_id.isin(done.game_id)], done], ignore_index=True)
            g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
            g.to_csv(GAMES, index=False)
    print(f"schedule: {len(done)} completed, {len(up)} upcoming in horizon")
    return g, up

def load_box():
    box = {}
    for line in open(GOALIES):
        d = json.loads(line)
        box[int(d["gid"])] = d
    return box

def refresh_box(g, box, season):
    need = [int(x) for x in g.loc[(g.season == season) & g.hs.notna(), "game_id"] if int(x) not in box]
    if not need:
        return box
    def toi_sec(t):
        try: m, s = str(t).split(":"); return int(m)*60 + int(s)
        except Exception: return 0
    def pull(gid):
        try:
            r = requests.get(f"{API}/gamecenter/{gid}/boxscore", headers=H, timeout=15).json()
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
    out = open(GOALIES, "a")
    with ThreadPoolExecutor(max_workers=10) as ex:
        for rec in ex.map(pull, need):
            if rec:
                out.write(json.dumps(rec) + "\n")
                box[int(rec["gid"])] = rec
    out.close()
    print(f"goalie boxscores added: {len(need)}")
    return box

def projected_starters(g, box):
    proj = {}
    done = g[g.hs.notna()].sort_values("date")
    for tm in set(done.home) | set(done.away):
        rec = done[(done.home == tm) | (done.away == tm)].tail(10)
        starters = []
        for r in rec.itertuples():
            b = box.get(int(r.game_id))
            if not b:
                continue
            lst = b["h"] if r.home == tm else b["a"]
            if lst:
                starters.append(max(lst, key=lambda x: x[1])[0])
        if starters:
            proj[tm] = Counter(starters).most_common(1)[0][0]
    return proj

def run(today_str):
    today = pd.Timestamp(today_str)
    season = season_code(today)
    g, up = refresh_games(season, today)
    box = refresh_box(g, load_box(), season)
    proj = projected_starters(g, box)

    pseudo = []
    for r in up.itertuples():
        box[int(r.game_id)] = {"h": [[proj.get(r.home), 3600, 0, 0]] if proj.get(r.home) else [],
                               "a": [[proj.get(r.away), 3600, 0, 0]] if proj.get(r.away) else []}
        pseudo.append(dict(game_id=r.game_id, date=r.date, gtype=r.gtype, home=r.home, away=r.away,
                           hs=np.nan, as_=np.nan, endtype=None, neutral=r.neutral, utc=r.utc,
                           season=season))
    g_all = pd.concat([g, pd.DataFrame(pseudo)], ignore_index=True).sort_values(
        ["date", "game_id"]).reset_index(drop=True) if pseudo else g
    f = E.add_diffs(E.build(g_all, box, CFG))
    tr = (f.season >= TRAIN_LO) & (f.season <= TRAIN_HI) & f.hs.notna()

    # v2 shipped head: three-way regulation / OT decomposition
    fitted = E.fit_decomp(f, tr, C)
    # v1 binary, fitted alongside purely so the grade block can show both
    sc = StandardScaler().fit(f.loc[tr, E.W].values)
    mdl_bin = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(f.loc[tr, E.W].values),
                                                         f.loc[tr, "y"].values)
    base = float(f.loc[tr, "y"].mean())
    past_reg_rate = float(f.loc[tr, "past_reg"].mean())
    print(f"head v2: q(home wins past regulation)={fitted[2]:.4f}  "
          f"past-regulation rate in train={past_reg_rate:.4f}  n_train={int(tr.sum())}")

    gp = g[(g.season == season) & g.hs.notna()]
    counts = pd.concat([gp.home, gp.away]).value_counts()
    gate_open = len(counts) >= 32 and counts.min() >= CAL_MIN_GP

    alerts = []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"game_id": str}) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["date", "game_id", "season", "home", "away", "p_home",
                              "p_home_binary", "p_elo", "model"])
    if "p_home_binary" not in log.columns:
        log["p_home_binary"] = np.nan          # v1 rows predate the decomposition

    board = None
    if len(up):
        rows = f[f.game_id.isin(up.game_id)].copy()
        p = E.predict_decomp(fitted, rows[E.W].values)
        p_bin = mdl_bin.predict_proba(sc.transform(rows[E.W].values))[:, 1]
        pe = 1/(1+10**(-((rows.h_elo + CFG["HA"]) - rows.a_elo)/400))
        board = pd.DataFrame({"date": pd.to_datetime(rows.date).dt.strftime("%Y-%m-%d"),
                              "game_id": rows.game_id.astype(str), "season": season,
                              "home": rows.home, "away": rows.away, "p_home": p,
                              "p_home_binary": p_bin, "p_elo": pe.values, "model": "v2-decomp"})
        log = pd.concat([log[~log.game_id.isin(board.game_id)], board], ignore_index=True)
        gate_txt = "" if gate_open else f"  [CALIBRATION: no betting until every team has {CAL_MIN_GP} GP]"
        print(f"\n=== MBM NHL board ==={gate_txt}")
        for _, r in board.iterrows():
            vals = (f"value: {r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}"
                    if gate_open else "CALIBRATION: predictions only")
            print(f"{r.away:>3} @ {r.home:<3}  p(home)={r.p_home:.3f}  "
                  f"(binary {r.p_home_binary:.3f}, elo {r.p_elo:.3f})  | {vals}")
    else:
        print("no games in the next 2 days")

    fin = g.assign(gid=g.game_id.astype(str)).set_index("gid")
    log["result"] = log.game_id.map((fin.hs > fin.as_).astype(float).where(fin.hs.notna()))
    log = log[["date", "game_id", "season", "home", "away", "p_home", "p_home_binary",
               "p_elo", "model", "result"]]
    log.to_csv(pred_path, index=False)

    dec = log[log.result.notna()]
    if len(dec) >= 30:
        y = dec.result.astype(int).values
        print(f"\nrolling grade (n={len(y)}):")
        series = [("v2 decomp", dec.p_home), ("v1 binary", dec.p_home_binary),
                  ("elo floor", dec.p_elo), ("base rate", pd.Series(np.full(len(y), base)))]
        for lbl, pp in series:
            v = pp.values.astype(float)
            if not np.isfinite(v).all():
                print(f"{lbl}: n/a for {int((~np.isfinite(v)).sum())} rows logged before v2")
                continue
            pc = np.clip(v, 1e-6, 1-1e-6)
            print(f"{lbl}: LL={log_loss(y, pc):.4f}  Brier={brier_score_loss(y, pc):.4f}  "
                  f"hit={np.mean((pc > .5) == (y == 1)):.3f}")
        w = dec.tail(300)
        if len(w) >= 300:
            yy = w.result.astype(int).values
            edge = log_loss(yy, np.full(len(yy), base)) - \
                log_loss(yy, np.clip(w.p_home.values.astype(float), 1e-6, 1-1e-6))
            print(f"last-300 edge vs base: {edge:+.4f}  [{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("NHL winner")
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    tg, ch = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if tg and ch and (board is not None or alerts):
        if board is not None:
            lines = []
            for _, r in board.iterrows():
                if gate_open:
                    lines.append(f"{r.away} @ {r.home}: {r.p_home:.3f} | {r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}")
                else:
                    lines.append(f"{r.away} @ {r.home}: {r.p_home:.3f} | CALIBRATION, no betting")
            txt = "\n".join(lines)
        else:
            txt = "no games today"
        if alerts:
            txt += "\n\u26a0 TRIPWIRE: " + ", ".join(alerts) + " below base rate, stop betting this board"
        requests.post(f"https://api.telegram.org/bot{tg}/sendMessage",
                      data={"chat_id": ch, "text": f"MBM NHL\n{txt}"}, timeout=20)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
