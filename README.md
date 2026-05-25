# Alalu DJ Audio Engine

An automated DJ mixing engine built with FastAPI and Python. Analyses tracks, plans musically-aware transitions, and renders continuous DJ sets — including beat sync, harmonic mixing, strategy selection, and full audio output.

---

## Architecture Overview

```
Library Scan → Queue Planning → Transition Planning → Rendering → Output WAV
    ↓               ↓                  ↓                  ↓
library.py    queue_manager.py     planner.py         renderer.py
              smart_set_builder.py  analysis.py        transitions.py
                                   strategy_router.py
```

The pipeline is strictly one-way. The library scan runs once per track and stores metadata to disk. Planning and rendering happen at request time.

---

## Core Modules

### `core/analysis.py`
Low-level audio analysis. Everything the planner needs before it can make decisions.

**Beat and downbeat detection** uses madmom's RNNBeatProcessor and DBNBeatTrackingProcessor in combination. Madmom is used over librosa because it handles tempo changes and rubato more reliably, and its probabilistic downbeat model is significantly more accurate for electronic music with sidechain compression.

**Key estimation** uses chromagram analysis with spectral rolloff gating. High-frequency content is filtered before chroma extraction so synthesiser noise doesn't corrupt key readings. Returns both the detected key and a confidence score — low-confidence key estimates are penalised in harmonic scoring.

**Energy curve** computes per-frame RMS normalised to the track's peak, producing a 0–1 energy timeline. Used downstream for A exit selection, B entry gating, and silence detection.

**Phrase boundary detection** groups downbeats into musical phrases. The phrase length adapts to BPM: 16 bars below 100 BPM, 8 bars from 100–145 BPM, 4 bars above 145 BPM. Returns timestamps of phrase-start positions in both the A exit zone (60–85% of track) and B intro zone (0–20% of track).

**Sync functions:**

- `fine_sync_onset` — sub-beat refinement. Given a phrase boundary, searches ±4 beats for the exact sample where onset patterns best correlate. Returns the adjusted sample position and a confidence score. Falls back to the original phrase boundary when confidence < 0.55.

- `find_best_downbeat_sync` — downbeat-level alignment. Searches B downbeats within ±4 bars of the phrase sync result. Scores each candidate by onset correlation × alignment bonus (how close to zero-lag the bar-1s land). Keeps the result local — never searches outside the phrase region.

- `align_beats_to_grid` — sample-accurate beat locking. Finds A's next beat after the transition point and B's next beat after the entry point, then shifts B's start sample so both beats land at the same moment. Clamped to ±1 beat to guard against grid detection errors.

- `verify_beat_alignment` — post-sync drift measurement. Runs onset cross-correlation on the full mix duration window and returns lag in milliseconds. Drift > half the segment length is capped to zero (measurement artefact from spectrally mismatched content).

---

### `core/planner.py`
The main transition planning logic. Given two track paths, returns a complete transition plan including A exit time, B entry time, mix duration, beat-aligned sample positions, sync score, and recommended strategy.

**Pair scoring** (`score_transition_pair`) evaluates every `(A_time, B_time)` candidate combination on nine dimensions:

| Component | Weight | What it measures |
|---|---|---|
| `phrase_score` | 0.20 | Is A exiting on a strong phrase boundary (bar-1 of 8-bar phrase)? |
| `harmonic_score` | 0.18 | Are the tracks harmonically compatible (Camelot wheel)? |
| `b_intro_score` | 0.22 | Does B's entry have energy? Is it vocal-free? Is it early in the track? |
| `vocal_free_a` | 0.14 | Is Track A free of vocals at the exit point? |
| `runway_score` | 0.12 | Does A have enough track remaining for the full mix duration? |
| `a_energy_score` | 0.07 | Is A still energetic at the exit (not in a breakdown)? |
| `low_stability` | 0.04 | Is A's low-frequency content stable (not a breakdown-level drop)? |
| `loudness_match` | 0.02 | Are the tracks at similar absolute loudness levels? |
| `outro_zone_score` | 0.01 | Is A in a musically appropriate exit region of the track? |

Weights adjust by mix profile (BPM and energy level of the track pair).

**Exhaustive sync search** runs `fine_sync_onset` on all candidate pairs, sorted by pair score descending. Selection criterion: highest pair score that passes the sync gate (sync ≥ 0.35) and post-verification. Ties in pair score are broken by higher sync accuracy.

**Post-verification** checks the winning pair by running `verify_beat_alignment` on the full mix duration window. Pairs where drift exceeds 1 bar (≈1875ms at 128 BPM) are rejected as false positives. Sparse-intro tracks (sync < 0.62) use a 2-bar tolerance since piano and ambient content produces unreliable drift readings.

**Three-stage sync pipeline** runs in sequence after the best pair is selected:
1. `fine_sync_onset` — phrase-level region and sub-beat refinement
2. `find_best_downbeat_sync` — bar-level alignment within ±4 bars of stage 1 result
3. `align_beats_to_grid` — sample-accurate beat locking

**Strategy selection** runs after all three sync stages complete, using the actual energy values at the final A exit and B entry positions — not global track averages. This means the strategy reflects what the audio actually sounds like at the transition point, not what the track typically sounds like.

**Silence penalties** prevent the planner from choosing bad transition points:
- Track A energy < 40% of its own peak → 85% score penalty (breakdown/outro territory)
- Track A after-window energy < 25% → 80% penalty (actively fading)
- Track B energy < 15% over 32s after entry → disqualified (silence)
- Track B energy 15–25% → 35% score (quiet but acceptable with long mix)

**Mix duration** defaults to 32 bars at track BPM (60s at 128 BPM), floored at 60s, capped at 120s. Can be overridden per-request via `preferred_mix_duration`.

**Strategy rotation** — the renderer passes `last_strategy` to the planner. If the scorer picks the same strategy as the previous transition, it switches to the second-best option if it scores within 15% of the top score. Prevents consecutive transitions from using identical techniques.

---

### `core/strategy_router.py`
Scores all available strategies against the current pair's properties and returns the best eligible one.

**Scoring model** evaluates six factors per strategy:

| Factor | What it measures |
|---|---|
| Harmonic | Key compatibility between tracks |
| Rhythmic | BPM delta tolerance |
| Energy | How well the transition point energy matches the strategy's ideal |
| Sync | Sync accuracy quality |
| Position | Track position (A is at 60–85%, B is at 0–20%) |
| Duration | How well the mix duration fits the strategy |

Each strategy has a weight vector across these six factors reflecting what matters most for that technique. `harmonic_mix` weights harmonic factor at 0.50. `percussion_blend` weights sync and energy highest.

**Eligibility requirements** are hard constraints checked before scoring:

| Strategy | Harmonic | Min sync | Max BPM delta | Min duration |
|---|---|---|---|---|
| `harmonic_mix` | Required | 0.40 | 10 BPM | 32s |
| `long_eq_blend` | Required | 0.40 | 8 BPM | 48s |
| `techno_filter_drive` | Must be False | 0.50 | 6 BPM | 16s |
| `phrase_mix` | Any | 0.50 | 10 BPM | 16s |
| `percussion_blend` | Any | 0.50 | 5 BPM | 16s |
| `hpf_sweep` | Any | 0.30 | 14 BPM | 12s |
| `lpf_sweep` | Any | 0.30 | 14 BPM | 12s |
| `breakdown_blend` | Any | 0.30 | 8 BPM | 16s |
| `energy_blend` | Any | 0.30 | 8 BPM | 16s |

**Removed from auto-selection:** `bass_swap` (unreliable in continuous mode), `reverb_wash` (sounds wrong on melodic electronic music), `auto_loop` (loop effect, not a blend strategy). All remain callable via the manual API.

---

### `core/transitions.py`
Audio rendering implementations for each strategy. All functions take `(segment_a, segment_b, sr, fx_parameters)` and return a mixed segment at the same length as the longer input.

**`harmonic_mix`** — Bass-delayed crossfade with equal-power gains. The bass frequencies of B are held back until 40% through the mix, preventing double-bass. Ideal for harmonically compatible tracks where both melodies should overlap clearly.

**`phrase_mix`** — Phrase-boundary-aware blend. Bass gate on A (low frequencies ducked progressively), equal-power crossfade on mids and highs. Designed for transitions where B enters on a phrase boundary while A is completing its phrase.

**`long_eq_blend`** — Three-band sequential crossfade. High frequencies cross first (15% of mix), mids cross at midpoint, bass crosses last (85%). Gives each frequency band its own transition timeline — the Anjunadeep signature technique. Requires 48s+ mix duration to pay off.

**`percussion_blend`** — High-frequency percussion comes through early (A's lows stay dominant until midpoint), then bass crosses in the second half. Groove is transferred before bass — sounds like a real DJ crossfade on a mixer.

**`energy_blend`** — Clean bass-delayed crossfade. B's bass is held until 60% through. Safe fallback for any condition; sounds musical but without character.

**`breakdown_blend`** — Designed for A in breakdown/outro. Aggressive sigmoid on A's exit with B leading the mix from the start. B's bass arrives very late (78% through) giving the incoming track space to establish.

**`hpf_sweep`** — STFT-domain high-pass filter sweeping up on A as it exits. Thins A's content progressively while B arrives underneath. Good for non-harmonic transitions where key clash needs to be masked.

**`lpf_sweep`** — Inverse: low-pass filter sweeping down on A, blurring it into the background. More musical than HPF for tracks with active melodies — sounds like A dissolves rather than being cut.

**`techno_filter_drive`** — Saturation + HPF sweep. Aggressive. Requires `harmonic=False` — the distortion destroys melodic content and would sound wrong on harmonically compatible tracks. Appropriate for peak-hour non-harmonic handoffs.

**`echo_out`** — Tempo-synced echo on A's tail, B enters clean. The echo is locked to beat duration so it stays rhythmic. Good for energetic exits where A should leave with impact rather than a fade.

**`loop_roll`** — Beat-synced roll exit. A is played with exponential envelope decay on repeated bar segments. High-energy technique for peak-to-peak transitions.

**`ambient_transition`** — Very soft sigmoid crossfade with late bass entry. For atmospheric, both-quiet transitions.

---

### `core/library.py`
Track metadata scanning and storage.

`scan_track_metadata` analyses a track and stores to `library_metadata.json`:
- BPM (madmom primary, librosa fallback)
- Key and Camelot wheel position
- Downbeat times and beat times
- Energy curve (times + values)
- Energy summary (mean, std, raw_rms_db, peak)
- Drop points (energy-jump candidates)
- Duration

Scans are cached — a track already in the library is not re-analysed unless forced.

---

### `core/queue_manager.py`
Track selection for automated set building.

`score_next_track` evaluates a candidate track given the current track and set position. Scores on:
- **Harmonic compatibility** — Camelot wheel distance (same key = 1.0, adjacent = 0.7, incompatible = 0.2)
- **BPM proximity** — exponential decay, ideal within ±3 BPM
- **Energy arc** — build phase prefers energy increases, peak phase prefers plateau, outro phase prefers decreases
- **Avoid repetition** — tracks recently played are penalised

`recommend_next_tracks` returns ranked candidates for the next position in the set.

---

### `core/smart_set_builder.py`
Automated full set construction.

`build_smart_track_order` uses `queue_manager` to greedily build a track ordering from a library, optimising for energy arc, harmonic flow, and BPM progression.

`build_and_render_smart_set` wraps the full pipeline: order tracks → plan all transitions → render to WAV.

---

### `core/renderer.py`
Continuous set rendering. Takes a list of ordered tracks with pre-computed transition plans and renders the full set as a single WAV.

**Per-segment normalisation** — each track segment is normalised to `TARGET_RMS = 0.06` before mixing. Prevents loudness inconsistencies across tracks with different mastering levels.

**Peak ceiling** — `PEAK_CEILING = 0.90`. Soft limiter applied after mixing each segment.

**Drift correction** — after sync and alignment, the renderer checks residual drift with `verify_beat_alignment`. Corrections up to `MAX_DRIFT_CORRECTION_MS = 200ms` are applied by shifting B's segment start. Larger drift indicates a wrong entry point, not a correctable offset.

**Strategy degradation** — when sync accuracy < 0.55 and the planned strategy is timing-sensitive (`percussion_blend`, `loop_roll`), the renderer automatically degrades to `energy_blend`. Logs the degradation.

**`_safe_strategy`** — hard enforcer. Blocks `bass_swap` and `drop_mix` in continuous mode (structural incompatibility). Blocks `harmonic_mix` when tracks are not harmonically compatible regardless of what the planner chose.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/library/scan` | POST | Scan a track file or folder into the library |
| `/v1/library/tracks` | GET | List all scanned tracks |
| `/v1/autodj/transition` | POST | Plan and render a single A→B transition |
| `/v1/autodj/set/render-queue-order` | POST | Render a pre-ordered track list as a continuous set |
| `/v1/autodj/set/build-and-render` | POST | Auto-order and render from a library |
| `/v1/autodj/recommend` | POST | Get recommended next tracks |

---

## Configuration

Key parameters and where to change them:

| Parameter | File | Default | Effect |
|---|---|---|---|
| `TARGET_RMS` | `renderer.py` | 0.06 | Per-segment loudness target |
| `PEAK_CEILING` | `renderer.py` | 0.90 | Hard output ceiling |
| `MAX_DRIFT_CORRECTION_MS` | `renderer.py` | 200ms | Maximum beat drift correction |
| `SYNC_TARGET` | `planner.py` | 0.55 | Sync confidence threshold for post-verify |
| `SYNC_MIN_ACCEPT` | `planner.py` | 0.35 | Minimum sync to attempt post-verify |
| Mix duration default | `planner.py` | 32 bars (60–120s) | `_default_mix_dur` formula |
| A exit zone | `planner.py` | 60–85% of track | `min_percent` / `max_percent` |
| B entry zone | `planner.py` | 0–20% of track | `max_percent=0.20` |
| B energy gate (hard) | `planner.py` | 0.15 | Below this = disqualified |
| B energy gate (soft) | `planner.py` | 0.25 | Below this = 35% score penalty |
| A silence penalty | `planner.py` | 0.40 normalised | Below this = 85% score penalty |

---

## Dependencies

- **madmom** — beat and downbeat detection
- **librosa** — audio analysis fallbacks, onset detection, STFT operations
- **numpy / scipy** — signal processing throughout
- **soundfile / sounddevice** — audio I/O
- **FastAPI / uvicorn** — API server
- **pyRubberBand or equivalent** — time-stretching for BPM matching

---

## Data Flow: Single Transition

```
Track A path ──┐
               ├─→ load_or_analyze_track() ──→ beats_a, downbeats_a, energy_a, key_a
Track B path ──┘                               beats_b, downbeats_b, energy_b, key_b

                      ↓
          camelot_compatible(key_a, key_b) → harmonic_ok

                      ↓
          phrase_boundary_candidates(downbeats_a) → candidates_a  [60–85% zone]
          intro_phrase_candidates(downbeats_b)    → candidates_b  [0–20% zone]

                      ↓
          for each (a_time, b_time) pair, sorted by pair_score ↓:
              score_transition_pair() → pair_score
              fine_sync_onset()       → sync_acc, synced_b_sample
              verify_beat_alignment() → passed_verify
              if passed_verify and better than current best → update winner

                      ↓
          find_best_downbeat_sync()   → refine to nearest bar  [±4 bars]
          align_beats_to_grid()       → sample-accurate lock

                      ↓
          choose_strategy_with_scores(energy_at_a, energy_at_b, sync, harmonic)
              → strategy + rotation check against last_strategy

                      ↓
          return plan {
              track_a_time, track_b_time, mix_duration,
              synced_b_sample, sync_accuracy,
              recommended_strategy, strategy_scores,
              harmonic_compatible, bpm_a, bpm_b, ...
          }

                      ↓
          _safe_strategy(plan) → validated strategy

                      ↓
          _render_transition_segment(y_a, y_b, plan)
              → normalise → apply_transition_strategy() → soft_limit → output
```

---

## Design Decisions

**Why pair score is primary, sync is a gate:** A musically strong pair at sync=0.70 will sound better than a musically weak pair at sync=0.92. Sync accuracy has diminishing returns above ~0.65 — the sub-beat alignment correction handles the residual. Musical quality (phrase position, energy, harmonic compatibility) should drive selection; sync just needs to be "acceptable."

**Why the post-verify uses the full mix duration window:** Short windows (4 bars = 6s) cannot distinguish bar positions in electronic music — kick patterns repeat every 4 bars and look identical. A 60-second window has enough melodic and harmonic variation that genuine misalignment produces measurably different drift from bar-accurate alignment.

**Why candidates_a uses phrase grouping but candidates_b uses phrase grouping too:** Phrase grouping on both sides keeps the candidate count small (~6–8 × ~5–6 = 30–48 pairs) while covering all musically valid transition points. Every 8-bar phrase boundary in the DJ exit zone (60–85%) is a valid A exit; every 8-bar phrase boundary in the intro zone (0–20%) is a valid B entry.

**Why Camelot compatible tracks get `harmonic_score = 1.0` and others get `0.40` rather than `0.0`:** Non-harmonic transitions are not bad — they just require different techniques. `techno_filter_drive` and `hpf_sweep` are specifically designed for them. Giving non-harmonic pairs a floor score of 0.40 keeps them in contention so these strategies can be selected, rather than forcing all non-harmonic transitions to fail scoring.

**Why madmom over librosa for beat detection:** madmom's probabilistic beat tracker handles the "double-tempo" problem (detecting 128 BPM as 64 BPM) much more reliably. Its downbeat model understands 4/4 bar structure, which is essential for phrase boundary detection. librosa is retained as a fallback for tracks where madmom fails.