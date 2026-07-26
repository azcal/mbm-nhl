# MBM NHL daily engine (rebuild v1)

Standalone predictive model, no market inputs. Winner head only: P(home win incl
OT/SO), matching the 2-way moneyline, no pushes. The totals head failed validation
(two-thousandths of a goal over a constant) and is permanently benched.
Frozen through 2025-26; 2026-27 is the forward test.
Holdouts: 2024-25 LL 0.6620, 2025-26 LL 0.6835, clearing Elo and base both years.

Goalie layer is live-projected: expected starter = most-used goalie over the team's
last 10 games, repriced as usage shifts. Confirmed-starter feeds are a later upgrade.
CALIBRATION GATE: predictions print from opening night; value lines unlock only when
EVERY team has played 9+ games (typically ~3-4 weeks in). Runs twice daily.
Tripwire: last-300 vs base rate, pings Telegram if it fires.
