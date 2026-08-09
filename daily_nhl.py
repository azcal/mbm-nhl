"""
MBM NHL daily runner (rebuild v2, frozen spec).
- Refreshes current-season schedule/results + goalie boxscores from the NHL API
- Projects starters (most-used goalie over last 10 team games) into pseudo rows
- Boards upcoming games (2-day horizon), winner head only (totals benched)
- Discord push routed through tg_delta so refreshes only report what moved
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
import tg_delta
import board_format as BF
import freshness as FR
import notify
import engine_nhl as E

# Shared files must be at least this new. A minimum rather than an exact match, so a later
# improvement to a shared file does not force an edit here.
SHARED_MIN = "2026-08-03"
for _mod in (BF, notify, tg_delta, FR):
    _have = getattr(_mod, "VERSION", "")
    if _have < SHARED_MIN:
        raise ImportError(
            f"{_mod.__name__}.py is version {_have or 'unstamped'}, this runner needs "
            f"{SHARED_MIN} or newer. Copy the current shared files into this repo.")

H = {"User-Agent": "Mozilla/5.0"}
API = "https://api-web.nhle.com/v1"
GAMES = os.path.join(HERE, "data", "games.csv")
GOALIES = os.path.join(HERE, "data", "goalies.jsonl")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C = SPEC["cfg"], SPEC["C"]
TRAIN_LO, TRAIN_HI = 2018, 2025
CAL_MIN_GP = SPEC.get("cal_min_gp", 9)
HORIZON_DAYS = SPEC.get("horizon_days", 2)
# Tracks board_format rather than restating it. This is the console-side copy, and a
# local 0.05 here while board_format ran at 0.00 meant the Actions log and Discord
# quoted different prices for the same game.
MIN_EDGE = BF.MIN_EDGE
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
        up = cur[cur.hs.isna() & (cur.date >= today) & (cur.date <= today + pd.Timedelta(days=HORIZON_DAYS))]
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

def fetch_rosters(season, timeout=25):
    """team -> set of goalie ids currently on the roster.

    api-web.nhle.com/v1/roster/{TEAM}/{SEASON} splits forwards, defensemen and goalies, so
    this is a direct answer to who could start tonight. Empty on any failure, and the
    caller falls back to usage alone rather than boarding nothing.

    There is deliberately no confirmed-starter call here: the official API has none. The
    boxscore carries goalies but its `starter` field is null and only populates after the
    game, and gameInfo has scratches and coaches, nothing pregame. Confirmed starters exist
    only on third-party sites, and a scrape of one is a worse dependency than the gap it
    closes.
    """
    out = {}
    for tm in TEAMS:
        try:
            # season_code returns a single start year; this endpoint wants the pair.
            r = requests.get(f"https://api-web.nhle.com/v1/roster/{tm}/{season}{season+1}",
                             timeout=timeout)
            if r.status_code != 200:
                continue
            ids = {int(p["id"]) for p in r.json().get("goalies", []) if p.get("id")}
            if ids:
                out[tm] = ids
        except Exception:
            continue
    if out:
        print(f"rosters: {len(out)} teams, "
              f"{sum(len(v) for v in out.values())} goalies")
    else:
        print("roster feed unavailable, projecting on usage alone")
    return out


def projected_starters(g, box, rosters=None):
    """Most-used goalie over the team's last ten games, constrained to the roster.

    Usage alone was wrong in both directions across a trade: a goalie who has just arrived
    appears in none of his new team's last ten and can never be projected, while he is
    still in his old team's ten and can be projected for a team he has left. Intersecting
    with the roster fixes both at once. Usage still picks between the goalies who are
    actually there, which is the part it is good at.
    """
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
        keep = (rosters or {}).get(tm)
        if keep:
            elig = [x for x in starters if x in keep]
            if elig:
                starters = elig
            else:
                # Nobody who has been starting for this team is still on its roster, which
                # happens in an offseason when both goalies move. Fall back to whichever
                # roster goalie has played most recently ANYWHERE, which is the same
                # league-wide lookup the basketball boards use: a goalie traded in has no
                # minutes for this team but plenty for his last one.
                last_seen = {}
                for r2 in done.itertuples():
                    b2 = box.get(int(r2.game_id))
                    if not b2:
                        continue
                    for ent in list(b2["h"]) + list(b2["a"]):
                        if ent and ent[0] in keep:
                            last_seen[ent[0]] = r2.date
                if last_seen:
                    pick = max(last_seen, key=lambda k: last_seen[k])
                else:
                    # Nothing on record for any of them. Sorted, so the same run twice
                    # gives the same answer; set ordering would not.
                    pick = sorted(keep)[0]
                print(f"  {tm}: no recent starter still on the roster, "
                      f"projecting {pick}")
                starters = [pick]
        if starters:
            proj[tm] = Counter(starters).most_common(1)[0][0]
    return proj

def run(today_str):
    today = pd.Timestamp(today_str)
    season = season_code(today)
    g, up = refresh_games(season, today)
    box = refresh_box(g, load_box(), season)

    # Refuse to price tonight on a history the source knows is incomplete. This is not an
    # "is it today's data" test: a league that did not play yesterday is current with old
    # data. It asks whether any completed game exists that this history has not seen.
    _newest = pd.to_datetime(g[g.hs.notna()].date).max().date()
    _miss, _ = FR.missed_games("hockey/nhl", _newest)
    FR.guard("NHL", _newest, "hockey/nhl")
    FRESH_LINE = FR.board_line(_newest, checked=(_newest, _miss))
    SEASON_LINE = None

    proj = projected_starters(g, box, fetch_rosters(season))

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
    min_gp = int(counts.min()) if len(counts) >= 32 else 0

    alerts = []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"game_id": str}) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["date", "game_id", "season", "home", "away", "p_home",
                              "p_home_binary", "p_elo", "model"])
    if "p_home_binary" not in log.columns:
        log["p_home_binary"] = np.nan          # v1 rows predate the decomposition
    # On a v1 row the binary WAS the published forecast, so p_home already holds it. Filling
    # the column makes that explicit rather than leaving it blank for every consumer of this
    # file, including the dashboard. A definitional fill, not a revision: no number changes.
    _v1 = log.model.astype(str).str.startswith("v1")
    log.loc[_v1 & log.p_home_binary.isna(), "p_home_binary"] = log.loc[
        _v1 & log.p_home_binary.isna(), "p_home"]

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
        print("\n--- NHL detail: model margin, total ---")
        for _, r in board.iterrows():
            vals = (f"value: {r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}"
                    if gate_open else "CALIBRATION: predictions only")
            print(f"{r.away:>3} @ {r.home:<3}  p(home)={r.p_home:.3f}  "
                  f"(binary {r.p_home_binary:.3f}, elo {r.p_elo:.3f})  | {vals}")
    else:
        print(f"no games in the next {HORIZON_DAYS} days")

    fin = g.assign(gid=g.game_id.astype(str)).set_index("gid")
    log["result"] = log.game_id.map((fin.hs > fin.as_).astype(float).where(fin.hs.notna()))
    log = log[["date", "game_id", "season", "home", "away", "p_home", "p_home_binary",
               "p_elo", "model", "result"]]
    log.to_csv(pred_path, index=False)

    dec = log[log.result.notna()]
    if len(dec) >= 30:
        y = dec.result.astype(int).values
        print(f"\nrolling grade (n={len(y)}):")
        # `p_home` is whatever this board published at the time: v1's binary across the
        # backfill, v2's decomposition from 2026-07-26 on. Grading it as one series and
        # labelling that series "v2 decomp" would credit v2 with thousands of games it
        # never forecast, so it is labelled for what it is. The v2-only comparison is
        # printed separately below, once v2 has a window of its own.
        is_v1 = dec.model.astype(str).str.startswith("v1").values
        series = [("published", dec.p_home), ("v1 binary", dec.p_home_binary),
                  ("elo floor", dec.p_elo), ("base rate", pd.Series(np.full(len(y), base)))]
        for lbl, pp in series:
            v = pp.values.astype(float)
            if not np.isfinite(v).all():
                print(f"{lbl}: n/a for {int((~np.isfinite(v)).sum())} rows logged before v2")
                continue
            pc = np.clip(v, 1e-6, 1-1e-6)
            print(f"{lbl}: LL={log_loss(y, pc):.4f}  Brier={brier_score_loss(y, pc):.4f}  "
                  f"hit={np.mean((pc > .5) == (y == 1)):.3f}")
        # v2 against v1 on the rows v2 actually forecast. Both columns are populated on
        # those rows, so this is a paired comparison rather than two different windows.
        v2 = dec[~is_v1]
        if len(v2) >= 100:
            yv = v2.result.astype(int).values
            pv = np.clip(v2.p_home.values.astype(float), 1e-6, 1-1e-6)
            pb = np.clip(v2.p_home_binary.values.astype(float), 1e-6, 1-1e-6)
            if np.isfinite(pb).all():
                print(f"v2 vs v1, live only (n={len(v2)}): v2 LL={log_loss(yv, pv):.4f}  "
                      f"v1 LL={log_loss(yv, pb):.4f}  gain {log_loss(yv, pb)-log_loss(yv, pv):+.4f}")
        else:
            print(f"v2 has forecast {int((~is_v1).sum())} graded games; "
                  f"the side-by-side needs 100")

        w = dec.tail(300)
        if len(w) >= 300:
            yy = w.result.astype(int).values
            edge = log_loss(yy, np.full(len(yy), base)) - \
                log_loss(yy, np.clip(w.p_home.values.astype(float), 1e-6, 1-1e-6))
            n_v1 = int(pd.Series(w.model.astype(str).str.startswith("v1")).sum())
            mix = f", {n_v1} of them v1 backfill" if n_v1 else ""
            print(f"last-300 published edge vs base: {edge:+.4f}{mix}  "
                  f"[{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("NHL winner")
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    push, skipped, cal_n = {}, 0, 0
    if board is not None:
        bd = [(today_str, f"{r.away} @ {r.home}",
               r.home if r.p_home >= .5 else r.away,
               max(r.p_home, 1 - r.p_home), r.game_id) for _, r in board.iterrows()]
        wg = max([len(x[1]) for x in bd] + [10]); wp = max([len(x[2]) for x in bd] + [8])
        for d_, gm, pk, q, gid in bd:
            if gate_open and BF.min_odds(q) < BF.PRICE_FLOOR:
                skipped += 1
                continue
            if gate_open:
                ln = BF.line(d_, gm, pk, q, w_game=wg, w_pick=wp)
            else:
                # Built to match notify's HEADER rule rather than BOARD, because there is
                # no price on this row and there must not appear to be one. Padding it
                # into BF.line's column layout without the "find X or better" tail matched
                # no regex at all and rendered as a run-on line.
                ln = (f"{d_}  {gm}  ·  calibration, {pk} {q:.1%}, gate opens at "
                      f"{CAL_MIN_GP} GP (lowest now {min_gp})")
                cal_n += 1
            push[gid] = {"line": ln, "min_odds": BF.min_odds(q),
                         # Carries the pick, so a goalie change flipping the side is
                         # reported even when the price barely moves.
                         "match": pk}
    print("\n=== MBM NHL board ===")
    for _v in push.values():
        print(_v["line"])
    if skipped:
        print(f"({skipped} game{'s' if skipped != 1 else ''} skipped, "
              f"price floor {BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)})")

    mode, lines = tg_delta.gate(LOGS, today_str.replace("-", ""), push)
    if lines or alerts:
        body = list(lines) if lines else ["no price changes since the last push"]
        foot = (f"{skipped} game{'s' if skipped != 1 else ''} skipped, "
                f"price floor {BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)}")\
               if skipped else None
        title = "MBM NHL" + (" · changes since the last push"
                             if mode == "delta" and lines else "")
        # GREY is the calibration colour. The gate here is all-or-nothing on games played,
        # so a board is either entirely inside the window or entirely out of it.
        colour = notify.GREY if push and cal_n == len(push) else notify.GREEN
        if alerts:
            body = body + ["", "TRIPWIRE: " + ", ".join(alerts)
                           + " below base rate, stop betting this board"]
            colour = notify.RED
        # The board carries its own provenance. A reader should not have to trust that
        # the data was current; the row says so.
        foot = " · ".join([x for x in (FRESH_LINE, SEASON_LINE, foot) if x])
        notify.send(title, body, foot, colour)
    elif not push:
        print("no board to send")
    else:
        print("board unchanged since the last push, nothing sent")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
