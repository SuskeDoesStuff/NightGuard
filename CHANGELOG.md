# Changelog

## v1.1 — Baseline policy

The first learned policy: `RecurrentPPO` with an LSTM, a three-stage curriculum, and the
measurements that turn partial observability into a number.

This milestone changed character. Up to v1.0 every question had a verifiable right answer; from here
progress comes from experiments, runs take real time, and "the curve is flat" has a dozen causes.
§7's v1.1 criteria were rewritten to match: they ask for **measurements being made and reported**,
not for particular numbers. A criterion that demands a number invites tuning until it appears.

### Exit criteria: 5 of 8 met

No threshold was moved and no run was tuned until a number appeared. The three misses are reported
as misses.

| # | Criterion | Outcome |
|---|---|---|
| 1 | `gym.make` works for every ID in §2.4 | **Met.** Two IDs registered, three v2.0 IDs correctly absent, tested |
| 2 | A curve that rises on stage 1, single configuration, before sweeping | **Met.** `mean_steps` 268.7 → 830.9, duty cycle 1.000 → 0.002 |
| 3 | Beats `do_nothing` on nights 4, 5, 6 outside sampling error at n=500 | **Met on night 4 only** — 0.140 against 0.000, **9.02σ**, transferred from a policy that never saw night 4. Night 5 is 0.0060, 1.74σ, under the 2σ bar. Night 6 is 0.000 |
| 4 | Beats `rhythm` on night 6 | **Not met, and not attempted.** Stage 1 never graduated, so training night 6 from a policy at 0.006 was not a defensible hour. `rhythm` measures 0.330 on the evaluation seeds |
| 5 | Oracle gap, non-zero, reported as a lower bound | **Not met.** −0.0040 ± 0.0040, −1.00σ. A floor effect, not a leak — see below |
| 6 | Camera duty cycle logged, trend reported either way | **Met.** Logged every evaluation across four arms; falls 1.000 → 0.002 |
| 7 | Every run reproducible from SHA, config hash and seed | **Met, with a caveat.** Every run records all three, and the hash covers CLI overrides. But only **1 of 3** ran from a clean tree; the other two are `-dirty` |
| 8 | Four §11 gates green; §6, §7, §10 describe the code | **Met** |

`validate.py` now runs **94 checks**, of which the four above fail, each printing its measured value
against its target.

**Criterion 5's check was itself wrong at first, and that is worth recording.** It passed on any
non-zero gap, which meant it passed on the negative, statistically-null result actually measured. A
check that cannot fail for the right reason is worth nothing (§8.0), so it now requires a *positive*
gap of at least 2σ and fails at −1.00σ. Same trap as v0.3's shadowed validation function, one level
up: the check ran, printed a number, and said PASS.

**Criterion 7's caveat is real.** `-dirty` says a run is not reproducible from its SHA; it does not
say how far from it. Two of three runs were dirty in documentation only, and there was no way to
demonstrate that from the artefact. The manifest now records `dirty_paths`, so the claim is
checkable rather than asserted. The v1.1 runs predate that and keep the weaker guarantee.

The honest summary of the milestone: **the harness, the measurements and the discipline are sound;
the policy is not.** Nothing here learned to survive night 5. What the milestone produced instead is
a characterised account of *why*, plus four findings nobody had on their list — the `sparse_mode`
discount gradient, the `monitor_down` attractor, the night-4 transfer, and a criterion that cannot
be answered from the floor.

### Expectations, recorded before the runs

Written down first, per the evidence discipline in `CLAUDE.md`. There is no analytic derivation to
compare against here — an RL curve is not derivable from §3 — so what can honestly be fixed in
advance is the *expectation*, and then reported against whether or not it held.

1. **Stage 1 will learn from the dense term, not from reaching dawn.** On night 5 `do_nothing`
   scores 0.000 and `monitor_down` 0.007, so a random policy essentially never sees the `+10`. The
   gradient comes from `+0.01` per step survived, and `mean_steps` should move before `survival`
   does.
2. **`sparse_mode` on stage 1 will produce no learning signal at all**, for the same reason with the
   dense term removed. Run it as a final ablation and report the flat curve; it is not a bug.
3. **The Oracle gap is a lower bound, and the config it is measured on matters.** §6.4 already
   records why it is a lower bound. Measuring it on a config where both arms saturate — night 5,
   where `rhythm` alone reaches 0.973 — or where both arms floor would manufacture a spurious zero,
   and criterion 5 reads a zero as evidence of a leak. The matched pair therefore runs on the
   hardest config where the base agent demonstrably learns, chosen from the stage results.
4. **Camera duty cycle may go either way,** and §7's criterion was rewritten to say so. The
   interesting result is beating `rhythm` while peeking *less*; beating it by peeking more is a
   different and duller strategy, and both are reportable.

### The headline result: the agent rediscovers `monitor_down` and stops there

**Four arms, 2,000,000 steps each on night 5, all evaluated on the same 500 held-out seeds.** None
of them learned to survive the night.

| Policy | Survival | `mean_steps` | Duty | Dominant cause |
|---|---|---|---|---|
| `do_nothing` | 0.000 | 172.2 | 0.000 | SPRINTER, 200/200 |
| `monitor_down` | 0.000 | 820.4 | 0.000 | **blackout, 200/200** |
| learned, **baseline** | **0.0060** | 824.9 | 0.162 | blackout, 438/500 |
| learned, **oracle** | 0.0020 | 773.6 | 0.042 | blackout 363, **PROWLER 125** |
| learned, **sparse** | 0.0000 | 723.6 | 0.029 | blackout, 500/500 |
| learned, **normalised** | 0.0000 | 171.1 | 0.000 | **SPRINTER, 500/500** |
| `rhythm` | **0.980** | 1069.3 | 0.083 | survived, 196/200 |

The baseline lands on `monitor_down` almost exactly — 824.9 steps against 820.4, and the same
power-bound failure in 88% of episodes. Along the way the curve does what §7's criterion 2 asks:
`mean_steps` rises 268.7 → 830.9 and the duty cycle falls **1.000 → 0.002**. Both are real. Neither
is survival.

**The policy is not useless — it is power-limited, and night 4 shows it.** Evaluated on nights it
never trained on, at 500 held-out seeds each:

| Config | `night_divisor` | Learned | `do_nothing` | `rhythm` | Margin over `do_nothing` |
|---|---|---|---|---|---|
| Night 4 | 4.0 | **0.140** | 0.000 | 0.994 | **9.02σ** |
| Night 5 | 3.0 | 0.006 | 0.000 | 0.980 | 1.74σ |
| Night 6 | 3.0 | 0.000 | 0.000 | 0.330 | — |

That 0.140 is the milestone's one clean pass, and it arrived by transfer: the policy trained on
night 5 and was never shown night 4. It is also exactly what the diagnosis predicts. §3.10's
`night_constant = 0.1 / night_divisor` makes night 4 drain less than night 5, and a policy whose
only failure mode is running out of power converts that headroom straight into survival. The agent
did not learn to *manage* power; it learned a fixed posture that happens to fit inside night 4's
budget and not night 5's.

**The one thing that worked, briefly.** The `normalised` arm — `VecNormalize` on returns,
`target_kl = 0.02`, linear learning-rate annealing, the three orthodox things the baseline lacked —
reached **survival 0.1200 with a positive mean return (+0.54) and a duty cycle of 0.0702 at 700,000
steps**. That is the qualitative result the milestone was after: off the plateau, and beating it
while peeking *less* than `rhythm`'s 0.0793. It did not hold. By 900,000 it was back at 168.6 steps,
and its 2,000,000-step weights score 0.0000 with 500/500 SPRINTER deaths.

**And that transient was lost, which is a harness defect rather than a result.** The stage saved the
*final* policy, not the best one seen. On a run this unstable the final weights are an arbitrary
sample of an oscillation. Fixed: `EvaluationCallback` now checkpoints the best evaluation to
`best.zip`, ranked on survival with `mean_steps` breaking ties, and `StageResult` records both paths.
Every figure above predates the fix and none is restated by it.

**`target_kl` did not prevent the collapse.** That was the hypothesis the `normalised` arm existed
to test, and it is the cleanest negative result here: the arm still fell from 823.1 steps to 168.6
inside 200,000 steps. So the instability is not simply an unguarded update, and the "the baseline was
under-configured" explanation is at best partial — normalisation bought the only real survival
anybody saw, and did not buy stability.

**The plateau has a structure, and it is the environment's, not the optimiser's.** Escaping
`monitor_down` requires a *sequence* — close the left door, close the right door, raise the monitor
to freeze SPRINTER, drop it, reopen — and **every prefix of that sequence is strictly worse than not
starting it**. Raising the monitor with a door open is precisely what lets an entity into the office
(§3.4, §3.5), and closing doors without peeking just spends power. §6.3's power term charges for the
first step immediately, while the payoff — SPRINTER frozen, no escalating `1 + 5n` bang, power still
in hand at 5AM — arrives hundreds of steps later. The gradient points away from the peek loop from
every direction.

That is §7's v0.2 criterion 7 arriving from the other side. `monitor_down` was built as a probe for
a degenerate strategy, and the answer recorded then was that `rhythm` beats it comfortably. It does.
But it is still the strategy that gradient ascent on §6.3 finds first, and the one it stays in.

**Every run is unstable, in the same way.** `mean_steps` swings by ~400 between evaluations 100,000
steps apart, and three of the four arms visited 168.6 — below `do_nothing`'s 172.2 — at some point.
All four oscillate between the same two attractors: hold the monitor up (~270 steps, duty ≈ 1.0) and
`monitor_down` (~820 steps, duty ≈ 0.0).

A detail that makes the collapse legible: two independently trained arms with different configs,
different observation widths and separate processes produced **bit-identical** evaluation statistics
(543.60 mean steps, duty 0.9625). That is not a bug. *Which* camera is selected has no effect on the
simulation at all — only `monitor_up` gates office entry (§3.4, §3.5) and drains power (§3.10) — so
any two policies that collapse to "hold the monitor up" trace identical trajectories on identical
seeds. It is a clean signature of the attractor.

The `tuned` arm — `gae_lambda` 0.95 → 0.995 to match the advantage window to the mechanic's
timescale, `ent_coef` 0.01 → 0.02, `lr` 3e-4 → 2e-4, changed together and therefore not separably
attributable — damped none of it, and was stopped at 1,000,000 steps once it was oscillating 309 ↔
670 with survival 0.0000 throughout. Its curve is kept as a recorded failed attempt.

### Criterion 5 is not answerable at floor performance

The matched pair, night 5, 2,000,000 steps, same seed, same architecture, same budget, differing
only in the `Oracle` wrapper:

| Arm | Survival | `mean_steps` | Duty | Causes |
|---|---|---|---|---|
| base | 0.0060 (3/500) | 824.9 | 0.162 | 438 blackout, 40 SPRINTER, 18 PROWLER |
| **oracle** | **0.0020 (1/500)** | **773.6** | 0.042 | 363 blackout, **125 PROWLER**, 11 WARDEN |
| gap | **−0.0040 ± 0.0040** | −51.3 | | |

The gap is **negative and 1.00σ from zero** — three survivors against one, a difference of two
episodes. §7's criterion 5 says a zero gap means stop, because either the observation is leaking or
the task does not require memory. **Neither is the explanation, and the criterion's phrasing must not
be allowed to force one.**

It is not a leak. v1.0's probe covered 33 hidden-state mutations, and `validate.py`'s four no-leak
checks plus the non-vacuity converse still pass on this commit. It is not "memory is unnecessary"
either. **Both arms are pinned at the floor**, and when neither has learned the task the comparison
measures optimisation noise rather than information. This is the floor effect the planning round
predicted when choosing which config to run the pair on, and it is why the check also reports
`mean_steps`; that went the wrong way too, so the fallback does not rescue it.

The arms are nonetheless *not the same policy*. Ground truth changed the behaviour substantially —
125 PROWLER deaths against 18, at a quarter the duty cycle — it just did not change survival,
because both die to power.

**The finding is that criterion 5 presupposes a base arm that has learned something.** A meaningful
Oracle gap can only be measured against a policy that is off the floor. Recorded rather than
answered, and the "stop" condition is deliberately not triggered: stopping would assert a leak that
four independent checks say is not there.

### Expectation 2 was wrong: `sparse_mode` is not gradient-free

Reported rather than corrected, per the rule. The expectation above — and the prompt's — was that
stage 1 under `sparse_mode` would produce **no learning signal at all**, on the reasoning that a
random policy never reaches dawn and the only remaining term is the death penalty. It measures
otherwise: the sparse arm went from `mean_steps` 268.74 to **684.44 by 200,000 steps**, ahead of the
dense arm at the same point, and finished at **723.6** — the same `monitor_down` plateau the dense
arm found, reached with every dense term switched off.

The reasoning missed the discount. PPO optimises the *discounted* return, and a terminal penalty
arriving at step `t` is worth `-10 · γ^t` at the start of the episode — which is monotonically
**increasing** in `t`. At `γ = 0.999`:

| Dies at step | Discounted return |
|---|---|
| 268.74 | −7.642 |
| 485.50 | −6.152 |
| 684.44 | −5.042 |
| 1070 (dawn, `+10`) | +3.428 |

So the observed improvement is worth **+2.6 of discounted return**, on a run whose *undiscounted*
mean return stayed pinned at −10.00 to −10.05 — which is exactly what the summary reports, and
exactly why it looks flat. Discounting converts *when* the penalty lands into *how much* it costs,
and delaying death is therefore rewarded even with every dense term switched off.

The claim that survives is narrower and still worth making: `sparse_mode` removes the *shaped*
signal, so what is left is a much weaker and more delayed one that depends on the discount factor
rather than on §6.3. A `γ` close enough to 1 would genuinely flatten it. §7's v1.1 preamble is
worded to say that, rather than repeating the stronger claim.

### The curriculum, and what the measurements say

Measured at 400 seeds per night, `rhythm` frozen at `peek_every_steps=12`:

| Night | `rhythm` | `do_nothing` | `monitor_down` | Headroom above `rhythm` |
|---|---|---|---|---|
| 1 | 1.000 | 0.287 | 1.000 | 0.000 |
| 2 | 1.000 | 0.010 | 0.990 | 0.000 |
| 3 | **1.000** | 0.000 | 0.840 | **0.000** |
| 4 | 0.993 | 0.000 | 0.003 | 0.007 |
| 5 | 0.973 | 0.000 | 0.007 | 0.027 |
| 6 | 0.325 | 0.000 | 0.000 | 0.675 |

§10's curriculum row said *"start at night 3; night 1 is near-degenerate"*. Night 3 measures
**1.000** — the locked curriculum start was a night where a fixed script never dies. The row's
stated reason was also wrong on its own terms: it claimed `do_nothing` survives night 1 "most of the
time", which predates SPRINTER and measures **0.287** at 400 seeds, against §8.2's analytic 0.2397.
Night 1 is degenerate because `rhythm` and `monitor_down` both score 1.000, not for the reason
recorded. Both §10 and §7's v1.1 preamble carried it; both are replaced.

Nights 1-4 are not curriculum stages. A random policy has room to improve on them, but `rhythm`
already scores 1.000, 1.000, 1.000 and 0.993, so an agent that graduates has learned nothing a fixed
script did not already do. They stay as sanity checks. **Night 6 is the only shipped night with
meaningful room above the reference.**

### Difficulty saturates, and not for the reason you would expect

Measured at 200 seeds, night-6 escalation, varying only the starting AI levels:

| Starting levels (W/D/P/S) | `rhythm` | Causes |
|---|---|---|
| 4/10/12/16 (night 6) | 0.345 | 131 blackout, 69 survived |
| 4/13/14/18 | 0.235 | 153 blackout, 47 survived |
| 6/14/15/19 | 0.215 | 157 blackout, 43 survived |
| 8/15/16/20 | **0.185** | 163 blackout, 37 survived |
| 10/16/17/20 | **0.185** | 163 blackout, 37 survived |
| 14/18/18/20 | **0.185** | 163 blackout, 37 survived |
| **20/20/20/20** | **0.185** | 163 blackout, 37 survived |

A hard floor at 0.185, identical across four very different configurations, with **zero entity
deaths at every level**. `rhythm` closes both doors before every peek, so nothing ever gets in; its
only failure mode is running out of power under SPRINTER's escalating bang drain. Once SPRINTER
saturates at `ai.level_max`, raising the other three changes nothing at all.

Three consequences, and the first is the trap:

1. **`rhythm` is a power-economy probe above night 6, not a difficulty probe.** Calibrating a custom
   night against its score and concluding the night is easy would be exactly backwards.
2. **This is §3.10's designed tension working**, not a defect. A door-discipline policy is
   power-bound by construction. Recorded as a §10 row rather than filed as a bug.
3. **It is the most interesting regime for a learned policy.** `rhythm`'s fixed cadence provably
   wastes power — it peeks on a timer regardless of what it knows — so adaptive peek timing has
   somewhere to go.

The three stages replace §10's row: night 5 (reference 0.973, graduate at 0.90), night 6 (0.325,
graduate at 0.40), and `custom_max` at 20/20/20/20 (0.185, report only).

### Where the training wall clock goes

`scripts/profile_training.py` times `learn()` end to end and then the same number of environment
steps alone. Measured on a real `RecurrentPPO` rollout, night 5, x86_64 Linux, Python 3.14.4:

| Device | Training | Environment only | Environment share |
|---|---|---|---|
| CPU | 251 steps/s | 28,843 steps/s | **0.87%** |
| CUDA | 169 steps/s | 30,771 steps/s | 0.55% |

Two findings.

**The environment is 0.7% of a training run**, so `env/vector.py` stays unbuilt — now on a
measurement rather than an argument. §5 of `docs/V1.0-DISCREPANCY-RESOLUTION.md` deferred it on
exactly this claim and asked for it to be checked; it holds with room to spare. Taking the
environment to zero would buy under one percent. §6.5 updated to say so, with the equivalence-test
gate retained for whenever it is built.

**The GPU is slower than the CPU**, by a third. At `n_envs=16` and 100 input dims the LSTM is
launch-bound: the kernels are too small to amortise the dispatch. Recorded rather than assumed —
the runs use CPU.

What actually dominates is the PPO update, at **40× the rollout collection**, and it is driven by
SB3's fixed per-minibatch cost to pad and unroll LSTM sequences rather than by anything here:

| Shape | Minibatches/epoch | Steps/s | 2M steps |
|---|---|---|---|
| `n_envs=8, n_steps=256, batch=256` | 8 | 191 | 2 h 54 m |
| `n_envs=8, n_steps=256, batch=2048` | 1 | 709 | 47 m |
| `n_envs=16, n_steps=256, batch=2048` | 2 | **750** | **44 m** |

`configs/train/baseline.yaml` takes the last row. **This was chosen from a throughput measurement
taken before any training run, not by a sweep against results** — the prompt's ban is on tuning to
chase a number, and a configuration that cannot finish is not a configuration.

### v1.0 follow-ups

- **The environment was never registered with Gymnasium.** §2.4 declared `NightGuard-v0` from v0.1
  onwards, but nothing anywhere called `register()`, so `gym.make("NightGuard-v0")` raised
  `NameNotFound`. v1.0's criterion 1 passed because it constructed `NightGuardEnv` directly — a
  legitimate path that never touches the declaration, which is exactly how the gap survived a
  review. Two IDs now ship: `NightGuard-v0` with a `night` kwarg, and `NightGuard-CustomMax-v0`.
  The three IDs §2.4 schedules for v2.0 stay unregistered and a test asserts it, because an ID that
  silently resolves to something else is worse than a missing one. No `max_episode_steps`: the night
  is a fixed horizon the simulator ends itself, and a `TimeLimit` would report termination as
  truncation.
- **`custom_template.yaml` was a schema document, not a loadable preset.** `load_night_config` took
  an `int` in `[1, 6]` and had no by-name path. Added `load_preset(name)`, and
  `configs/nights/custom_max.yaml` for stage 3.

### Found during execution: the first unseeded reset replayed the seeded episode

Not in the prompt, and it lands squarely on training. `NightGuardEnv.reset` built its own
`np.random.default_rng(seed)` on the seeded path but used `self.np_random` on the unseeded one.
Gymnasium seeds `self.np_random` from `SeedSequence(seed)`, which is **bit-identical** to
`default_rng(seed)` — same bit-generator state, same spawned children, verified directly — so the
first unseeded reset spawned substreams 0..5 for a second time and replayed the seeded episode
verbatim. Measured on night 6 under all-`NOOP`, episodes 1 and 2 were both
`(102 ticks, KILLED_SPRINTER, 92.253)`; episodes 3 and 4 diverged.

SB3 seeds once and then resets without a seed for the remainder of a run, so **every training run
and every naive 500-episode evaluation silently duplicated its first episode.** The fix is to always
spawn from `self.np_random`, which is exactly equivalent on the seeded path — `reset(seed=k)` still
matches `NightSim.from_seed(k)` seed-for-seed and no existing fixture moves. `validate.py` still
passes 81 of 81 with it in, and now carries the regression guard.

### `rhythm`'s camera duty cycle is 0.0793, not 1/12

Its design figure is one peek every 12 decision steps, 0.0833. The peek cycle spends extra steps
closing both doors before the peek, which stretches the period past 12, and the measured
monitor-up fraction is **0.0793** on night 6. That is the bar criterion 6 compares against.

The definition also had to be pinned. For `rhythm` the monitor-up fraction and the
camera-action fraction coincide exactly, so either would have looked right; they come apart for any
policy that holds the monitor up across steps. §10 fixes it as the **monitor-up fraction**, because
that is the quantity that both drains power (§3.10) and opens the office (§3.4, §3.5).

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
