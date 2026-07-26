# NHL rebuild v1

Data: official NHL API (11,870 games 2017-18..2025-26, per-game goalie boxscores).
Winner head = P(home win incl OT/SO), matching the 2-way moneyline, no pushes.
Tuned on 2023-24 only (K=8, OT/SO Elo weight 0.7, carry 0.6). Frozen holdouts:

  2024-25 n=1,398: LL 0.6620 / Brier 0.2351 / hit 57.5%  (Elo 0.6673, base 0.6868)
  2025-26 n=1,394: LL 0.6835 / Brier 0.2452 / hit 55.9%  (Elo 0.6889, base 0.6930)

Model edge over Elo held at +0.0053/+0.0054 in both years, including 2025-26 where
league-wide team-strength signal collapsed (Elo edge over base fell to +0.004).

TOTALS HEAD: BENCHED AT BIRTH. MAE 1.881/1.831 vs constant 1.883/1.833. Without
shot-quality data the model adds two-thousandths of a goal over "predict 6.1".
Board is winner-only until an xG layer earns totals back.

Design: Elo with discounted OT/SO results, opponent-adjusted decayed goals (fast/slow),
shot-share, rolling goalie save% above league per starter (shot-shrunk, backup flag),
b2b / 3-in-4 / travel / rest. Runner ships with the 9-games-per-team calibration gate.
