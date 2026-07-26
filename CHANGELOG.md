# Changelog

## v0.2 — Full roster

### v0.1 follow-ups

Spec corrections to `PROJECT.md`, from the v0.1 review:

- **§7 v0.1 criterion 2** made the closed form `(0.1 + 0.1/D) × 535` normative. The printed table
  is rounded to 3 dp and cannot be met at `1e-6`: only nights 3 and 4 are exact, night 1 is off by
  8.3e-5 and nights 2/5/6 by 3.3e-4. The table is now labelled display-only.
- **§7 v0.1 criterion 4** reworded. "Hour boundary crossings fire escalation exactly once each"
  over-counts: there are five interior boundaries but §3.3 assigns escalation to three. Now states
  three events at ticks 1790/2680/3570, with §3.3 authoritative.
- **§3.2** records that exactly one action resolves per decision step, so closing both doors takes
  1.0 s and simultaneous threats at both doors cannot both be answered. Intended, not an artifact.
- **§3.10** records that the clamp floor makes the *first* active control free.
- **§1.1 and §8** corrected to the test layout actually built: no `test_fidelity.py`, exit criteria
  in `scripts/validate.py`, per-mechanic assertions beside their mechanics.
- **New §8.0 "Non-vacuity"**: every statistical assertion needs a check that it can fail; threshold
  assertions record the measured value; structurally-unfailable assertions are marked provisional.
- **§7 v0.2 scope** corrected — door jam and office invasion shipped in v0.1. WARDEN has no jam
  mechanic, so v0.2 adds WARDEN, SPRINTER, the trace format, and office *entry* for WARDEN.
- **§4** lost `blackout.enabled`; **§1.2** raised the Python floor to 3.12.

Code:

- **`blackout.enabled` deleted.** With the flag off, power went negative and nothing happened — an
  undefined corner in the one subsystem everything else is validated against. Blackout is now
  unconditional and a config that still sets the key raises `ConfigError`.
- **Python floor 3.12.** numpy's stubs use `type` statements that only parse under 3.12, so
  `mypy --strict` could not be run at 3.11. An unverifiable support claim is worse than a floor.
- **`Action` split.** `env/actions.py` now owns the `Discrete(17)` space and the encode/decode
  boundary. `core.Action` keeps the integer values: §5's trace records `"action": 5` and §1 permits
  only `trace → core`, so moving the indices out would leave the trace writer unable to emit its
  own format. The split is indices (core) versus space (env).
- **Dead artifacts removed**: `configs/presets/v0_smoke.yaml`, `Clock.is_hour_boundary`.
- **Timing grid changed to 1/300 s**, expressed as `timing.time_units_per_second: 300` so the
  config value stays an exact integer — 1/300 has no finite decimal form, and dividing by a
  fractional resolution would carry representation error into every constant. `300 = lcm(60, 100)`:
  the countdown table divides by 60, the opportunity intervals are two-decimal. At 1/1000 or 1/100,
  six of WARDEN's ten countdowns are non-terminating; at 1/60, all four intervals and both immunity
  bounds fail. `test_config.py` now asserts every timing constant in every shipped night config
  lands on the grid — that assertion is the durable part, not the value.

### Entities, trace and policies

`WARDEN` (§3.4) and `SPRINTER` (§3.7) are implemented, along with §5's trace format, §3.9's audio
channel, and the reference policies §8.2 requires. `env/actions.py` is the only `env/` code, per
§7's v0.2 scope.

Two defects surfaced during the policy work, both of which had made a documented mechanic
unreachable while every targeted test still passed:

- **Monitor edges were resolved before the clock advanced,** so a SPRINTER attack triggered by
  raising the monitor was stamped with the previous tick. Its grace period then expired exactly on
  the next decision boundary, meaning the first step at which the agent could observe the `running`
  cue was the step the attack resolved. §3.7's "0.5 s grace period during which closing `door_left`
  still saves the agent" was unreachable on the monitor-raise path — only the 25 s forced-attack
  path worked. Edges are now resolved after the clock advance, giving exactly one usable decision.
- **Audio flags were per-tick, but §3.9 emits them per decision step.** A tick is a fifth of a
  step, so single-tick cues (`footstep`, `bang`) were invisible to a policy unless they happened to
  land on a step's last tick. State now carries both: per-tick flags for the trace, and a
  step-level accumulation for observation.

The first defect alone moved `rhythm` from 0.665/0.113/0.077/0/0/0 to 0.780/0.300/0.370/0.068/
0.003/0 across nights 1–6.

New config keys, all defaulted: `entities.*.enabled` (§8.3 requires running with "all other
entities disabled" and there was no mechanism), `warden.office_kill_interval_s`, `warden.door`, an
`audio.footstep_nodes` section, and a `trace` section carrying `cam_duty_window_steps` and
`stride`.

**§5's `event` field extended.** Several events genuinely co-occur on one tick — an invasion always
jams its door on the same tick — but the field holds a single value, and taking the first silently
dropped the invasion in favour of the jam. Co-occurring events now resolve by a fixed priority, and
§5's list gained `warden_retreat`, `sprinter_armed`, `sprinter_attack` and `escalation_hour_<n>`,
which the viewer wants as scrubber markers. `door_jam_*` consequently never appears alone.

`rhythm` was tuned once and frozen: `peek_every_steps` swept over {10, 12, 14, 16, 20, 28} at 400
seeds per night, with 12 dominating. The load-bearing detail is that it closes **both** doors before
each scheduled peek — raising the monitor is exactly what lets a waiting entity in — and that the
peek is not optional, because freezing SPRINTER is the only alternative to an escalating bang drain.

### §3.4 resolved: WARDEN's `E_CORNER` "attack" is a move

§3.4 said WARDEN "attacks" when an opportunity succeeds at `E_CORNER` with the monitor up and a
camera other than `E_CORNER` selected. Read as an immediate kill, `OFFICE` was unreachable and
§3.4's own 25%/s office mechanic was dead code.

Resolved as a **move into `OFFICE` gated on `monitor_up`**, with the kill being the 25%/s roll
afterwards while the monitor is down. A closed `door_right` still retreats to `E_HALL`. CeriW is
explicit: WARDEN "cannot enter your office when your camera is down. He can only enter while you
are looking at a camera that isn't [`E_CORNER`] while the doors are open", the kill is "25% every 1
second" while the cameras are down, and a permanently-down monitor means he never attacks.

This makes all three path entities enter only while the monitor is up, so SPRINTER is the sole
entity punishing a monitor-down policy — the asymmetry §3.5's v0.1 rationale already claimed.

### §8.2 restructured, with the derivation

**Night-1 `do_nothing` survival, derived from §3.1, §3.3 and §3.7 before implementing SPRINTER.**

`do_nothing` never raises the monitor, so no camera freeze and no immunity window ever apply, and
never closes a door, so there are no bangs and no stage resets. SPRINTER's opportunities fire at
`ceil(n × 5.01 / 0.1)` ticks. Night-1 levels are 0 until 3AM (t = 268 s), 1 until 4AM (t = 357 s),
2 to dawn, giving **53 opportunities at level 0, 18 at level 1, 35 at level 2** and an expected
4.400 successes against an arming threshold of 3.

Arming is not sufficient: the forced attack fires 25 s later, and §3.1 makes reaching t = 535 s
uncaught a win, so the 3rd success must land by **t = 510 s**. That removes the last five level-2
opportunities, leaving **18 at level 1 and 30 at level 2** and an expected 3.900.

Survival is then `P(X + Y < 3)` for `X ~ Bin(18, 0.05)`, `Y ~ Bin(30, 0.10)`:

| Method | Survival |
|---|---|
| Poisson approximation on the full-night mean 4.4 | 0.1851 |
| Exact binomial, 25 s deadline ignored | 0.1719 |
| **Exact binomial with the 25 s deadline** | **0.2397** |

**0.2397 is normative.** At 10,000 episodes the binomial SD is 0.0043, so the earlier estimate of
0.19 sits 11.6 SD away.

§8.2's original table required night-1 survival ≥ 0.8 and a strict monotonic chain across nights.
Neither is reachable against a faithful SPRINTER: expected success counts are 8.95 on night 2 and
14.0 on night 3, so `do_nothing` survival is indistinguishable from zero from night 2 onward and
the chain cannot be satisfied at any sample size. Restructured to an agreement test on night 1
(which can fail in both directions, unlike a threshold) and non-increasing-with-ties on nights 2–6.

**Note on the v0.1 rationale below.** v0.1 justified the §3.5 monitor gate partly by §8.2's ≥ 0.8
threshold, which this entry supersedes. The decision stands on its other two legs — §3.5's kill
trigger presupposes the monitor was up at entry, and CeriW states the rule directly — and is now
further corroborated by the identical rule applying to WARDEN.

## v0.1 — Skeleton

Initial implementation: simulation core with clock, power model, the five office controls, and the
`DRIFTER` and `PROWLER` entities. Blackout is a placeholder that terminates immediately with
`KILLED_BLACKOUT`.

### Spec changes

**`PROJECT.md` §7, v0.1 exit criterion 3: "One door" corrected to "Two doors".**

§3.10 defines `active` as the count of true office controls clamped to `[1, 4]`, so exactly one door
closed drains at the idle rate (`0.1104167 pp/s` on night 1) and the night never blacks out. The
criterion as written was therefore unreachable. Two doors closed gives `0.2104167 pp/s` and blackout at
t = 470.495 s — the figure the criterion already quoted, indicating the rate was computed for two units
and then described as one door.

Sources disagree on the underlying rule. CeriW's `research/how-the-game-works.md` states
count-with-floor-1, max 4, matching §3.10 verbatim. The Steam "Power Drainage Rates" guide measured
12/20/29/38 percentage points per hour across its four usage levels, which implies `1 + count`. Resolved
in favour of CeriW, consistent with the two existing precedents in §10 and with `CLAUDE.md` designating
§3 normative. Recorded as a new §10 row.

**`PROJECT.md` §3.5: office entry now requires `monitor_up`.**

As written, §3.5 let a door entity enter through an open door at any time, and §7's exit criteria do
not test it. That is both internally inconsistent and unfaithful:

- Being killed "the next time `monitor_up` becomes false" presupposes the monitor was up at entry.
  With a policy that never raises it, the phrase has no referent.
- §8.2 requires `do_nothing` to survive night 1 at least 80% of the time and calls it "the single
  most diagnostic assertion in the suite". `do_nothing` leaves both doors open all night, so under
  the unguarded rule it was invaded on roughly two seeds in three — measured at 0.30 survival before
  the fix, 1.00 after.
- CeriW's kill trigger ("jumpscared the next time you bring the cameras down") only makes sense if
  entry happens behind a raised monitor, and the technical wiki states outright that an entity at the
  door cannot enter while the monitor is down.

A closed door still turns the entity away regardless of the monitor; an open door with the monitor
down is a stalemate in which the entity camps at the corner, which §3.5 already anticipates.
Recorded as a §10 row.

**Office invasion implemented in v0.1** rather than v0.2 as §7 schedules. §3.5's rule — door permanently
jammed, kill on the next monitor-down or after 30 s — is confirmed faithful by CeriW, and deferring it
would leave an open door with no consequence. No v0.1 exit criterion is affected. v0.2 is now
`WARDEN` and `SPRINTER` only.

### New config keys

Added to §4's schema, all with defaults, so no logic carries a literal (§1.3):

- `power.night_constant_numerator` (`0.1`) — the numerator of `night_constant = 0.1 / D`.
- `entities.drifter.corner` (`W_CORNER`) and `entities.prowler.corner` (`E_CORNER`) — the node at
  which door resolution replaces movement, previously implicit in the pool and chain definitions.
- `timing.time_resolution_s` (`0.001`) and `timing.conversion_tolerance` (`1e-9`) — numerical
  hygiene rather than game rules. The opportunity intervals (3.02, 4.97, 4.98, 5.01 s) are not
  multiples of the 0.1 s tick, so the schedule is held in exact integer units of this resolution and
  each firing tick is recomputed from the firing count. Accumulating floats drifts (the 5th WARDEN
  firing lands on tick 152 instead of 151), and rounding the intervals to the tick grid would put
  DRIFTER and PROWLER on the same 50-tick period forever, which §3.3 explicitly does not want.
