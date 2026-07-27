# MBM NHL daily engine (rebuild v2)

Standalone predictive model, no market inputs. Winner head only: P(home win incl
OT/SO), matching the 2-way moneyline, no pushes. The totals head failed validation
(two-thousandths of a goal over a constant) and is permanently benched.
Frozen through 2025-26; 2026-27 is the forward test.
Holdouts: 2024-25 LL 0.6613, 2025-26 LL 0.6821, clearing Elo and base both years.

## v2: the overtime coin flip is modelled, not absorbed

About 22% of NHL games go past sixty minutes. Home teams win those at almost exactly
50% (0.4884, 0.4974 and 0.5040 across three training windows), so the regulation home
advantage does not survive into overtime. And whether a game gets there is not
predictable from team strength: fitting P(past regulation) on the full feature set
scores AUC 0.4877 on the dev season, worse than a coin, with predictions spanning only
0.154 to 0.316.

The v1 binary head had no way to represent that. It applied its team-strength edge to
every game and lost to a constant on the subset that went to overtime, by 0.0007 in
2024-25 and 0.0138 in 2025-26.

v2 fits three outcomes instead, home regulation win, away regulation win, and past
regulation, then recombines them as

    P(home) = P(home regulation win) + P(past regulation) * q

with q the training-set rate at which home teams win past regulation. Worth +0.0007 to
+0.0014 log loss, positive in all three windows tested (dev 2023-24 and both frozen
holdouts). Fitting q from features rather than using the constant was tested and added
nothing.

This is a small gain and the ceiling behind it is not small. Roughly a quarter of the
slate carries an irreducible log loss near 0.693 no matter what the model knows. Even
if regulation prediction reached 0.550, which would be excellent, the blended total
would only reach about 0.580.

## Expected goals: tested, declined

MoneyPuck xG was evaluated on 2026-07-26 against all 11,870 games in the training set,
joined on game id at 100% coverage. Adding opponent-adjusted decayed xG for and against
came in worse than baseline on two of three windows. Replacing shot share with xG was
worse on two of three. Replacing the goals states with xG entirely was worse everywhere.
Only a four-feature version (xG for/against, a fast-decay xG state, and high-danger xG)
helped consistently, and it was worth about +0.0007 on top of the decomposition.

Declined. MoneyPuck blocks automated requests and asks scrapers to arrange a data
licence, so a twice-daily job against them is not something to build on a +0.0007 gain
without written permission. The pipeline stays on free unlicensed sources.

## Operating notes

Goalie layer is live-projected: expected starter = most-used goalie over the team's
last 10 games, repriced as usage shifts. Confirmed-starter feeds are a later upgrade.
CALIBRATION GATE: predictions print from opening night; value lines unlock only when
EVERY team has played 9+ games (typically ~3-4 weeks in). Runs twice daily.
Tripwire: last-300 vs base rate, pings Telegram if it fires.

Every forecast logs both heads. `p_home` is the shipped v2 decomposition, `p_home_binary`
is the v1 binary fitted alongside it, so the two can be graded side by side as v2
accumulates a window. Rows logged before 2026-07-26 have `p_home_binary` blank because
the binary was the shipped forecast at the time.
