"""
MBM NHL rebuild v1 -- standalone, no market inputs.
Winner head: P(home win incl OT/SO), clean binary, matches the 2-way moneyline.
Total head: reference goals total (MAE floor). Data: official NHL API (schedules +
per-game goalie boxscores). Goalie layer = rolling save% above league per starter.
Protocol: burn-in 2017-18; tune on 2023-24 only; 2024-25 and 2025-26 frozen holdouts.
"""
import numpy as np, pandas as pd, json, math
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

REN = {"ARI": "UTA"}
CITY = {"ANA":(33.81,-117.88),"BOS":(42.37,-71.06),"BUF":(42.88,-78.88),"CGY":(51.04,-114.05),
 "CAR":(35.80,-78.72),"CHI":(41.88,-87.67),"COL":(39.75,-105.01),"CBJ":(39.97,-83.01),
 "DAL":(32.79,-96.81),"DET":(42.34,-83.06),"EDM":(53.55,-113.50),"FLA":(26.16,-80.33),
 "LAK":(34.04,-118.27),"MIN":(44.94,-93.10),"MTL":(45.50,-73.57),"NSH":(36.16,-86.78),
 "NJD":(40.73,-74.17),"NYI":(40.72,-73.59),"NYR":(40.75,-73.99),"OTT":(45.30,-75.93),
 "PHI":(39.90,-75.17),"PIT":(40.44,-79.99),"SJS":(37.33,-121.90),"SEA":(47.62,-122.35),
 "STL":(38.63,-90.20),"TBL":(27.94,-82.45),"TOR":(43.64,-79.38),"UTA":(40.77,-111.90),
 "VAN":(49.28,-123.11),"VGK":(36.10,-115.18),"WPG":(49.89,-97.14),"WSH":(38.90,-77.02)}

def hav(a, b):
    if a is None or b is None: return 0.0
    la1, lo1 = a; la2, lo2 = b; p = math.pi/180
    x = 0.5 - math.cos((la2-la1)*p)/2 + math.cos(la1*p)*math.cos(la2*p)*(1-math.cos((lo2-lo1)*p))/2
    return 12742*math.asin(math.sqrt(x))

class Decay:
    def __init__(self, hl): self.d = 0.5**(1.0/hl); self.sw = 0.0; self.sx = 0.0
    def add(self, x, w=1.0): self.sw = self.sw*self.d + w; self.sx = self.sx*self.d + w*x
    def brk(self, keep): self.sw *= keep; self.sx *= keep
    def shrunk(self, tau=4.0): return self.sx/(self.sw + tau)

def load():
    import json as J
    g = pd.read_csv("nhl/games.csv", parse_dates=["date"])
    for c in ("home", "away"): g[c] = g[c].replace(REN)
    box = {}
    for line in open("nhl/goalies.jsonl"):
        d = J.loads(line); box[d["gid"]] = d
    return g.sort_values(["date", "game_id"]).reset_index(drop=True), box

def build(g, box, cfg):
    K, HA, CARRY, OTW, HLF, HLS, GTAU, GCARRY = (cfg["K"], cfg["HA"], cfg["carry"], cfg["otw"],
                                                 cfg["hl_f"], cfg["hl_s"], cfg["g_tau"], cfg["g_carry"])
    elo = defaultdict(lambda: 1500.0)
    gf_f = defaultdict(lambda: Decay(HLF)); gf_s = defaultdict(lambda: Decay(HLS))
    ga_s = defaultdict(lambda: Decay(HLS)); sog = defaultdict(lambda: Decay(HLS))
    gk = {}                                      # goalie -> {"sv","sh","season"}
    recent_gk = defaultdict(lambda: deque(maxlen=10))
    season_of = {}; gp = defaultdict(int); prev = {}
    lg_sv = Decay(4000); lg_g = Decay(2000)
    feats = []
    for r in g.itertuples():
        for tm in (r.home, r.away):
            if season_of.get(tm) != r.season:
                elo[tm] = 1500 + CARRY*(elo[tm]-1500)
                for D in (gf_f[tm], gf_s[tm], ga_s[tm], sog[tm]): D.brk(CARRY)
                gp[tm] = 0; prev.pop(tm, None)
                season_of[tm] = r.season
        b = box.get(int(r.game_id), {})
        hg_list, ag_list = b.get("h", []), b.get("a", [])
        def starter(lst):
            return max(lst, key=lambda x: x[1])[0] if lst else None
        h_st, a_st = starter(hg_list), starter(ag_list)
        lsv = lg_sv.sx/max(lg_sv.sw, 1) if lg_sv.sw > 100 else 0.905
        def grate(pid, tm):
            if pid is None or pid not in gk:
                return 0.0, 0.0, 1
        # rating = shots-shrunk save% above rolling league
            s = gk[pid]
            rate = (s["sv"] - lsv*s["sh"])/(s["sh"] + GTAU)
            usual = max(set(recent_gk[tm]) or {None},
                        key=lambda x: list(recent_gk[tm]).count(x)) if recent_gk[tm] else None
            return rate, min(s["sh"], 1500.0)/1500.0, (1 if usual is not None and pid != usual else 0)
        row = {"game_id": r.game_id, "season": r.season, "date": r.date, "home": r.home,
               "away": r.away, "hs": r.hs, "as_": r.as_, "endtype": r.endtype, "gtype": r.gtype}
        city = CITY.get(r.home)
        for side, tm, st in (("h", r.home, h_st), ("a", r.away, a_st)):
            pv = prev.get(tm)
            rest = min((r.date - pv["d"]).days, 7) if pv else 4
            b2b = 1 if (pv and (r.date - pv["d"]).days == 1) else 0
            g4 = sum(1 for d0 in (pv["ds"] if pv else []) if (r.date - d0).days <= 3)
            trav = hav(pv["c"], city)/1000.0 if pv else 0.0
            grt, gexp, gbkp = grate(st, tm)
            row.update({f"{side}_elo": elo[tm], f"{side}_gf_f": gf_f[tm].shrunk(),
                        f"{side}_gf_s": gf_s[tm].shrunk(), f"{side}_ga_s": ga_s[tm].shrunk(),
                        f"{side}_sog": sog[tm].shrunk(), f"{side}_rest": rest, f"{side}_b2b": b2b,
                        f"{side}_3in4": int(g4 >= 2), f"{side}_trav": trav,
                        f"{side}_gk": grt, f"{side}_gexp": gexp, f"{side}_gbkp": gbkp,
                        f"{side}_gp": min(gp[tm], 12)})
        feats.append(row)
        # ---- post-game updates ----
        lgm = lg_g.sx/max(lg_g.sw, 1) if lg_g.sw > 100 else 3.0
        h_sog = sum(x[3] for x in ag_list)       # shots faced by away goalies = home SOG
        a_sog = sum(x[3] for x in hg_list)
        for tm, gfor, gag, sfor, sag, opp in ((r.home, r.hs, r.as_, h_sog, a_sog, r.away),
                                              (r.away, r.as_, r.hs, a_sog, h_sog, r.home)):
            adj_f = gfor - (lgm - (-ga_s[opp].shrunk()))
            adj_a = gag - (lgm + gf_s[opp].shrunk())
            gf_f[tm].add(adj_f); gf_s[tm].add(adj_f); ga_s[tm].add(adj_a)
            if sfor and sag: sog[tm].add((sfor - sag))
            gp[tm] += 1
            pv = prev.get(tm, {"ds": deque(maxlen=6)})
            pv["ds"] = pv.get("ds", deque(maxlen=6)); pv["ds"].append(r.date)
            pv["d"] = r.date; pv["c"] = city
            prev[tm] = pv
        lg_g.add(r.hs); lg_g.add(r.as_)
        for lst, tm in ((hg_list, r.home), (ag_list, r.away)):
            st = starter(lst)
            if st is not None: recent_gk[tm].append(st)
            for pid, toi, sv, sh in lst:
                s = gk.setdefault(pid, {"sv": 0.0, "sh": 0.0, "season": r.season})
                if s["season"] != r.season:
                    s["sv"] *= GCARRY; s["sh"] *= GCARRY; s["season"] = r.season
                d = 0.5 ** (sh/400.0)            # ~400-shot half-life
                s["sv"] = s["sv"]*d + sv; s["sh"] = s["sh"]*d + sh
                lg_sv.add(sv/max(sh, 1), w=sh)
        res = 1.0 if r.hs > r.as_ else 0.0
        w = 1.0 if r.endtype == "REG" else OTW
        e = 1/(1+10**(-((elo[r.home]+HA)-elo[r.away])/400))
        k = K * (1.2 if r.gtype == 3 else 1.0)
        elo[r.home] += k*w*(res-e); elo[r.away] -= k*w*(res-e)
    return pd.DataFrame(feats)

W = ["elo_d","gf_f_d","gf_s_d","ga_d","sog_d","gk_d","h_gbkp","a_gbkp","h_gexp","a_gexp",
     "rest_d","h_b2b","a_b2b","h_3in4","a_3in4","h_trav","a_trav","h_gp","a_gp"]
TT = ["h_gf_s","a_gf_s","h_ga_s","a_ga_s","h_gk","a_gk","h_b2b","a_b2b"]

def add_diffs(f):
    f = f.copy()
    f["elo_d"] = f.h_elo - f.a_elo
    f["gf_f_d"] = f.h_gf_f - f.a_gf_f; f["gf_s_d"] = f.h_gf_s - f.a_gf_s
    f["ga_d"] = f.a_ga_s - f.h_ga_s; f["sog_d"] = f.h_sog - f.a_sog
    f["gk_d"] = f.h_gk - f.a_gk; f["rest_d"] = f.h_rest - f.a_rest
    f["y"] = (f.hs > f.as_).astype(int)
    f["total"] = f.hs + f.as_
    return f

def fit_winner(f, tr, te, C):
    sc = StandardScaler().fit(f.loc[tr, W].values)
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(f.loc[tr, W].values), f.loc[tr, "y"].values)
    return m.predict_proba(sc.transform(f.loc[te, W].values))[:, 1], f.loc[te, "y"].values, f.loc[te]

def M(y, p, lbl):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return dict(model=lbl, n=len(y), log_loss=round(float(log_loss(y, p)), 4),
                brier=round(float(brier_score_loss(y, p)), 4), hit=round(float(np.mean((p > .5) == (y == 1))), 4))

def run():
    g, box = load()
    best = None
    for K in (8, 12):
        for otw in (1.0, 0.7):
            for carry in (0.6, 0.75):
                cfg = dict(K=K, HA=35, carry=carry, otw=otw, hl_f=8, hl_s=20, g_tau=300, g_carry=0.85)
                f = add_diffs(build(g, box, cfg))
                tr = (f.season >= 2018) & (f.season <= 2022); va = f.season == 2023
                for C in (0.05, 0.2, 1.0):
                    p, y, _ = fit_winner(f, tr, va, C)
                    ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
                    if best is None or ll < best[0]: best = (ll, cfg, C, f)
    ll23, cfg, C, f = best
    print(f"tuned on 2023-24: LL={ll23:.4f}  K={cfg['K']} otw={cfg['otw']} carry={cfg['carry']} C={C}")
    json.dump({"cfg": cfg, "C": C}, open("nhl/spec.json", "w"))
    f.to_parquet("nhl/features_best.parquet")
    out, preds = [], []
    for test, hi in ((2024, 2023), (2025, 2024)):
        tr = (f.season >= 2018) & (f.season <= hi); te = f.season == test
        p, y, rows = fit_winner(f, tr, te, C)
        preds.append(rows.assign(p_home=p))
        base = f.loc[tr, "y"].mean()
        e = 1/(1+10**(-((rows.h_elo + cfg["HA"]) - rows.a_elo)/400)).values
        res = [M(y, np.full_like(y, 0.5, float), "coin flip"),
               M(y, np.full_like(y, base, float), f"base rate ({base:.3f})"),
               M(y, e, "Elo (floor)"),
               {"model": "always Elo favorite", "n": len(y), "log_loss": None, "brier": None,
                "hit": round(float(np.mean((e > .5).astype(int) == y)), 4)},
               M(y, p, "REBUILD v1")]
        R = pd.DataFrame(res); R.insert(0, "holdout", f"{test}-{str(test+1)[2:]}"); out.append(R)
        trm = f.loc[tr]
        sct = StandardScaler().fit(trm[TT].values)
        mt = Ridge(alpha=5.0).fit(sct.transform(trm[TT].values), trm.total.values)
        pt = mt.predict(sct.transform(rows[TT].values))
        print(f"{test}-{str(test+1)[2:]} total MAE: model {np.mean(np.abs(pt-rows.total)):.3f} | "
              f"constant {np.mean(np.abs(trm.total.mean()-rows.total)):.3f}")
    allr = pd.concat(out, ignore_index=True)
    allr.to_csv("nhl/comparison.csv", index=False)
    pd.concat(preds).to_csv("nhl/holdout_predictions.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n" + allr.to_string(index=False))

if __name__ == "__main__":
    run()
