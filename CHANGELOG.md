# Changelog

## v1.0 — Environment

A Gymnasium interface over the validated simulator. All seven exit criteria met; `validate.py` now
runs **81 checks**, covering v0.1 through v1.0.

### The environment

`env/obs.py`, `env/reward.py`, `env/nightguard_env.py` and `env/wrappers.py`. Everything about *what
happens* stays in `core/`; this layer only decides what the agent is told and what it is paid.
Ground truth reaches the caller through `info`, never through `obs`.

**§6.2's observation is now 100 dims, not 98.** Two jam bits were added, inserted into the office
block rather than appended, so every later block shifts by two. The decisive argument is not a
source appeal: a door toggle with invisible semantics is unlearnable, because the agent presses the
button, nothing happens, and it cannot distinguish "jammed" from "I toggled twice in a row". Jam
state is office state and the block should read as one coherent unit. Nothing depended on the old
indices — v1.0 is the first version with an observation at all — so this was the last moment the
change was free.

**Belief is seeded at reset** from the start state §3 fixes, with staleness 0, and
`ticks_since_observed` reads 1.0 when never observed. The start never varies and is publicly known,
so a competent player begins with correct belief; modelling ignorance nobody has would contradict
§6.2's own design rule. Seeding reads **config, not live state**, which is what keeps the no-leak
criterion exact. Zero would have collided with "seen this instant" and made staleness jump upward on
the first sighting, inverting its meaning.

**Two observation channels, not one.** §6.2 joins them with `or`: the camera, and the door light.
Both corners carry a video feed, so watching a corner camera *is* an observation, and the proximity
block is the light-specific channel rather than the only corner channel. An earlier draft of the
plan had this backwards, which would have made both corners camera-invisible, taught the policy that
peeking at a corner is pointless, and silently broken the wing asymmetry §2.3 depends on — while
looking entirely plausible. It was caught in review before any code was written.

**`Oracle` carries SPRINTER's stage**, since it has no node. Recorded in §6.4: the measured gap is a
**lower bound**, because SPRINTER's immunity window and pending attack resolution tick stay hidden.
An `armed` flag is excluded as exactly `stage == stages_to_arm`.

### Measured

| Criterion | Result |
|---|---|
| 1. `check_env` | passes, **0 warnings** |
| 2. Random episodes | **9,996** across six nights, no exceptions |
| 3. No position leak | all four entities, hidden / in-office / jammed / lights-on, plus a non-vacuity check |
| 4. Throughput | **100,000 steps in 2.1 s** with encoding active — 47,026 steps/s on x86_64 Linux, Python 3.14.4 |
| 5. `reset(seed=k)` | identical trajectories |
| 6. Bounds | **0** observations outside the `Box`; lowest `power_pct` **0.000000** |

### v0.3 follow-ups

- **Power can no longer go negative.** The drain triggered blackout at `power_pct <= 0` but never
  clamped, and then kept draining throughout the blackout. SPRINTER's bang applies a discrete
  `1 + 5n` cost outside the drain step, so a late one overshot by a wide margin: night 6 with both
  doors held reached **−14.6 pp at onset and −19.5 pp at death** across 300 seeds. §6.2 encodes
  `power / 99.0` into a `Box(low=0.0)`, so this would have been either a `check_env` failure or,
  worse, a silent out-of-range input the policy trained against. Power is now clamped in the same
  branch that sets the flag, the drain is skipped entirely while in blackout, and the bang clamps at
  zero. The three exposed `max(0.0, …)` guards are gone; the invariant is asserted per tick instead,
  so a future discrete cost fails loudly rather than being papered over. §3.13 step 3 updated.
- **Forcing a blackout now zeroes power, and onset is in-band.** `--force-blackout-at` forced the
  blackout *state* without the *power*, so the one trace a human actually looks at depicted a
  blackout with a quarter of the battery left. Forcing now zeroes power one tick early and lets
  §3.13 step 3 detect the crossing, which also fixes the event stamp: calling `apply_onset` from
  outside the tick loop stamped it with an already-written tick and the trace showed it one record
  late. Onset now lands on exactly the intended tick with power at 0.0.
- **Fast suite 23.4 s → 11.5 s.** `test_survival_agrees_with_the_derivation` alone was 11.4 s — a
  slow test wearing a fast marker, duplicating its own `slow` twin at lower n.
- **§8.0 extended to human verification steps**, and §9.3.1 names two confirmed viewer fixtures.
  v0.3's suggested fixture could not exercise most of criterion 5: with the roster disabled every
  entity token sits on `STAGE` for all 5,290 records, no entity event fires, and `jams` is
  `(False, False)` throughout, so the `JAMMED` versus `open` distinction was unreachable rather than
  merely unverified. That is the same failure mode as a silently shadowed check, one level down.
- **§7's v1.0 throughput criterion restated by its purpose.** See below.

### §7 criterion 4: why a throughput number was replaced

Identical code, identical setup (night 4, single environment, all-`NOOP`, 152 steps per episode):

| Machine | Decision steps/s | Against the 50,000 gate |
|---|---|---|
| This host | **73,726** | passes |
| Reference host (throttled sandbox) | **13,031** | fails, by 3.8× |

A 5.7× spread on the same code means the gate was measuring the host. It would pass here, fail on
CI or a colleague's laptop, and the failure would read as a regression. The 50,000 figure was never
derived from anything; §6.5's own feasibility argument states the requirement as a PPO run finishing
in tens of minutes, and the environment is a single-digit percentage of v1.1's wall clock at either
measurement — 27 s of environment time for a 2,000,000-step run here, ~154 s there.

Criterion 4 is now: *a 100,000-step rollout using the v1.1 policy architecture completes in under
60 s of environment time on one core, with observation encoding active; record the figure and the
hardware.* That passes on both hosts (1.4 s and ~7.7 s) and is revisited in v1.1 with the training
loop as the benchmark. Recorded as a §10 row.

## v0.3 — Fidelity lock

### v0.2 follow-ups

- **§3.13 gained step 2b.** The code resolves monitor edges between steps 2 and 3, and had done
  since v0.2, but §3.13 still listed the original nine steps under a header reading "Fix this order
  and never change it." A fresh session reconciling code against spec could have restored the
  documented order, silently made §3.7's 0.5 s window unreachable again, and no test named for
  §3.13 would have caught it. Spec now matches code.
- **§8.2's provisional marker cleared.** SPRINTER exists; the assertion is live.
- **§8.3 records the countdown's ceiling quantisation.** Countdowns are exact on the 1/300 s grid,
  but a tick is 30 units and `units_to_ticks` ceilings, so AI 2's 13.333 s countdown expires at
  13.4 s and per-level deltas alternate 16/17 ticks rather than a uniform 16.67.
- **§8.4 and §8.7 rewritten** — see the derivations below.
- **Survival measurements moved behind a `slow` marker**, deselected by default. `pytest` is 6.4 s
  where it was 66 s; `scripts/validate.py` still runs everything.
- **`test_different_seeds_diverge` strengthened** from `len(outcomes) > 1` on tick counts to ten
  distinct full state signatures over forty seeds, matching `test_trace.py`'s threshold. A check
  satisfied by two outcomes in thirty is nearly as vacuous as what §8.0 exists to prevent.

### Derivations (written before measuring)

Every statistical target in §8 is derived analytically here first. If a measurement disagrees, one
of the two is wrong and finding out which is the point; neither gets adjusted to match the other.

#### §8.2, nights 2–6 — SPRINTER successes against the arming threshold

Extending the night-1 derivation. Opportunities fire at `ceil(n × 1503 / 30)` ticks; SPRINTER's
level changes at 3AM (t = 268 s) and 4AM (t = 357 s); and the 3rd success must land by t = 510 s
for the 25 s forced attack to resolve before dawn. That gives **53 / 18 / 30** opportunities at the
starting level, +1 and +2 respectively.

| Night | SPRINTER levels | E[successes] | P(survive) |
|---|---|---|---|
| 1 | 0 / 1 / 2 | 3.900 | **0.2397** |
| 2 | 1 / 2 / 3 | 8.950 | **0.0046** |
| 3 | 2 / 3 / 4 | 14.000 | < 0.0001 |
| 4 | 6 / 7 / 8 | 34.200 | < 0.0001 |
| 5 | 5 / 6 / 7 | 29.150 | < 0.0001 |
| 6 | 16 / 17 / 18 | 84.700 | < 0.0001 |

Night 5 sits below night 4 because SPRINTER starts at 5 rather than 6; both are far past the
threshold, so survival is indistinguishable from zero either way. This is why §8.2 permits ties.

#### §8.3 — STAGE→E_CORNER latency

Per hop: `E[hop] = C + Δ(C) + (20/L − 1) × 3.02`, where `C` is the countdown quantised up to a
tick and `Δ(C) = 3.02 × (⌊C/3.02⌋ + 1) − C` is the wait from the countdown expiring to the next
opportunity firing. Five hops.

**Δ is a credit, not a penalty**, and an earlier draft had the sign backwards. The countdown
consumes part of an inter-firing interval, so the next opportunity arrives *sooner* than a full
3.02 s after the move — adding a phase penalty moves the model away from the truth. At AI 10 there
is no countdown and therefore no phase effect at all, and `Δ(0) = 3.02` recovers the naive value.

| AI | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| countdown (ticks) | 150 | 134 | 117 | 100 | 84 | 67 | 50 | 34 | 17 | 0 |
| Δ (s) | 0.10 | 1.70 | 0.38 | 2.08 | 0.66 | 2.36 | 1.04 | 2.64 | 1.32 | 3.02 |
| **predicted (s)** | **362.4** | **211.4** | **146.0** | **120.8** | **90.6** | **80.5** | **58.2** | **52.9** | **33.6** | **30.2** |

Monotonically decreasing, with a sawtooth from Δ riding on the trend. AI 1's 362 s is most of a
535 s night, so this is measured on a lengthened night to avoid a censored mean (§8.3).

#### §8.4 — unfrozen fraction

SPRINTER is frozen while the monitor is up and for a `Uniform[0.83, 16.67] s` window sampled on
each monitor-down edge. For a policy peeking 0.5 s every `k` seconds the unfrozen time per cycle is
`max(0, k − 0.5 − U)`, so the unfrozen fraction is `E[max(0, k − 0.5 − U)] / k`.

| k (s) | 0.5 | 1.0 | 1.5 | 2 | 4 | 6 | 8 | 10 | 15 | 20 | ∞ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unfrozen | 0.0000 | 0.0000 | 0.0006 | 0.0071 | 0.0563 | 0.1147 | 0.1755 | 0.2373 | 0.3932 | 0.5375 | 1.0000 |

The hard-zero bound is **1.4 s**, not the continuous 1.33 s: immunity is sampled in grid units and
249 units ceilings to 9 ticks, so the realised minimum window is 0.9 s. And `k` must be a whole
number of 0.5 s decision steps, so only `{0.5, 1.0}` are both reachable and inside the zero region.

#### §8.7 — blackout survival

Under §3.11's three settled conventions — first roll one interval in, the 20 s guarantee replacing
the roll at 20 s, and a kill roll on the boundary counting — each of Approach and Song completes
with mass `{5: 0.2, 10: 0.16, 15: 0.128, 20: 0.512}`, and Kill rolls at +2, +4, … Convolving:

| Budget | 20 s | 25 s | 30 s | **35 s** | 40 s | 45 s |
|---|---|---|---|---|---|---|
| P(survive) | 0.9501 | 0.8977 | 0.7736 | **0.6148** | 0.4707 | 0.2833 |

**0.6148** is §8.7's target at power forced to zero at t = 500 s. The strict-boundary reading gives
0.6367, which is 4.5σ apart at n = 10,000.

### Measurements against those derivations

Every one agrees. Derivations were written above before any of this was run.

| Assertion | Derived | Measured | Agreement |
|---|---|---|---|
| §8.2 night 1 | 0.2397 | 0.2373 (n=10,000) | 0.55σ |
| §8.2 night 2 | 0.0046 | 0.0060 (n=2,000) | 0.90σ |
| §8.2 nights 3–6 | < 0.0001 | 0.0000 | within sampling |
| §8.7 blackout, 35 s | 0.6148 | 0.6141 (n=10,000) | 0.15σ |
| §8.4 hard zero, k ∈ {0.5, 1.0} | 0 attacks | 0 attacks over 40 seeds | exact |
| §8.4 curve, k = 2 … ∞ | see table | max absolute error 0.016 | within band |
| §8.3 latency, AI 1–10 | see table | mean absolute residual **2.51%** | within band |

**§8.3's residual is systematic, not statistical.** It spans −2.2% to +5.5% with z-scores of 1 to 5
at n=600, so it is a modelling gap rather than noise: the derivation is continuous while the
simulation quantises both the firing schedule and the countdown upward to whole ticks. The residual
has no consistent sign across levels and largely cancels, which is why the aggregate assertion
(mean absolute residual ≤ 4%) is much tighter than the per-level band (≤ 8%). **§8.4's residual is
consistently negative** for the same reason and in the expected direction: ceiling the immunity
window makes it slightly longer, so slightly less of the night is unfrozen than the continuous
model predicts.

### Reference policy series, post-tuning (A6)

400 seeds per night, `rhythm` frozen at `peek_every_steps=12`:

| Night | `rhythm` | `monitor_down` |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 1.000 | 0.980 |
| 3 | 0.998 | 0.715 |
| 4 | 0.917 | 0.000 |

**The curriculum start stays at night 3.** The concern that prompted this measurement — that the
monitor-down probe beat `rhythm` at the locked start, so an agent would learn "never raise the
monitor" first — used the pre-tuning figure of 0.370. Post-tuning `rhythm` wins at every night, so
no §10 row moving the start is warranted on those grounds.

**A second finding, which is the better argument.** `rhythm` scores 0.998 on night 3, leaving
essentially no headroom for a learned policy to improve against. A curriculum stage where the
reference strategy already survives 99.8% of episodes gives a very weak gradient regardless of what
the probe does. If v1.1 moves the curriculum start, this — not the probe ordering — should be the
reason, and the reasoning should be inherited rather than the number.

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
