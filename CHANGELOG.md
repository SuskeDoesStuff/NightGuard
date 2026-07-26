# Changelog

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
