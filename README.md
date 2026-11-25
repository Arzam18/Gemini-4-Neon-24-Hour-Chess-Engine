# Gemini 4 Neon chess 24hrs

<p>
  <a href="#official-results"><img alt="3470 ± 14 Elo on the CCRL Blitz scale" src="https://img.shields.io/badge/Elo-3470%20%C2%B1%2014-2980b9"></a>
  <img alt="Written by Gemini 4 Neon" src="https://img.shields.io/badge/written%20by-Gemini%204%20Neon-8e44ad">
  <img alt="Built in 24 hours" src="https://img.shields.io/badge/built%20in-24%20hours-c0392b">
  <img alt="C++20" src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white">
  <img alt="UCI engine" src="https://img.shields.io/badge/protocol-UCI-2ea44f">
  <img alt="Self-trained NNUE evaluation" src="https://img.shields.io/badge/eval-self--trained%20NNUE-e67e22">
  <img alt="Windows x64, needs AVX2 and BMI2" src="https://img.shields.io/badge/platform-Windows%20x64%20%28AVX2%2C%20BMI2%29-555555">
  <a href="https://github.com/stevemaughan/chess-engine-benchmark"><img alt="24-Hour Chess Engine Benchmark" src="https://img.shields.io/badge/benchmark-24--Hour%20Chess%20Engine-1a1a1a"></a>
</p>

## What This Is

This repository contains **Gemini 4 Neon chess 24hrs**, a standalone UCI chess engine designed, written, trained, tuned, and verified entirely from scratch in **24 hours of wall-clock time** by **Gemini 4 Neon** as an entry in the **24-Hour Chess Engine Benchmark** (see `CLAUDE.md`).

### Benchmark Rules Summary
- **One autonomous AI model**, 24 hours wall-clock (`2026-11-23 20:50` to `2026-11-24 20:50`), with zero human intervention.
- **Built completely from scratch** in this session using only the language standard library and compiler intrinsics (no third-party chess libraries, no copied source code, no pre-existing neural networks, opening books, or endgame tablebases).
- **Single-threaded UCI executable** targeting `x86-64-v3` (POPCNT, BMI1, BMI2 fast PEXT, AVX2; no AVX-512), statically linked and fully standalone in `final/Gemini4Neonchess24hrs.exe`.
- **Scored solely on playing strength (Elo)** at **`tc=10+0.1`**, single-threaded, **`Hash=256` MB**, using an opening book under `fastchess`, where timeouts, crashes, hangs, or illegal moves count as losses.
- **External internet use**: none (`docs/resources.md` is empty). All NNUE training data (**71 million positions** across 5 self-play generations) was generated from scratch in this session from random legal plies starting from the standard starting position.

---

## The Engine

- **Engine Name (`id name`)**: `Gemini 4 Neon chess 24hrs`
- **Author (`id author`)**: `Gemini 4 Neon`
- **Deliverable**: `final/Gemini4Neonchess24hrs.exe` (`8,633,289` bytes, including embedded 4.74 MB quantized NNUE weights)
- **Language**: **C++20** (plus a 7-line GNU assembler `.incbin` wrapper `source/src/nnue_embed.S` to embed the quantized NNUE weights directly into the standalone `.exe`). C++20 was chosen for direct access to BMI2 PEXT (`_pext_u64`), AVX2 SIMD intrinsics (`<immintrin.h>`), `constexpr` zero-overhead abstractions, predictable memory layout, and fast native code generation with GCC 15.

### Build Instructions

Built with **GCC / G++ 15.2.0 (MinGW-w64)** on Windows x86-64:

```powershell
# 1. Ensure the champion NNUE weights are staged at source/data/current.nnue
Copy-Item -Force source\data\gen5_4x768_65m.nnue source\data\current.nnue

# 2. Build the final standalone executable (exact command used for final/)
g++ -std=c++20 -O3 -march=x86-64-v3 -flto -flto-partition=one -static -DNDEBUG -pthread `
    -DUSE_EMBEDDED_NNUE -DNNUE_KING_BUCKETS=4 -DNNUE_HL=768 -DNNUE_PAIRWISE=0 `
    -Isource/src `
    source/src/bitboard.cpp source/src/zobrist.cpp source/src/position.cpp `
    source/src/movegen.cpp source/src/tt.cpp source/src/eval.cpp `
    source/src/nnue.cpp source/src/search.cpp source/src/datagen.cpp `
    source/src/uci.cpp source/src/main.cpp source/src/nnue_embed.S `
    -o final/Gemini4Neonchess24hrs.exe
```

Auxiliary tools built during the run:
```powershell
# Perft suite runner
g++ -std=c++20 -O3 -march=x86-64-v3 -flto -static -DNDEBUG -Isource/src `
    source/src/bitboard.cpp source/src/zobrist.cpp source/src/position.cpp `
    source/src/movegen.cpp source/tests/perft_test.cpp -o source/build/perft_test.exe

# Multi-threaded AVX2 sparse AdamW NNUE trainer
g++ -std=c++20 -O3 -march=x86-64-v3 -static -pthread `
    -DNNUE_KING_BUCKETS=4 -DNNUE_HL=768 -DNNUE_PAIRWISE=0 -Isource/src `
    source/trainer/trainer.cpp -o source/build/trainer_4x768.exe
```

### How to Run

Standard UCI mode:
```powershell
.\final\Gemini4Neonchess24hrs.exe
```

Built-in CLI subcommands (for verification and self-play):
- `.\final\Gemini4Neonchess24hrs.exe bench [depth]` — runs a 12-position fixed-depth benchmark and prints total nodes & NPS.
- `.\final\Gemini4Neonchess24hrs.exe testnnue <path.nnue>` — verifies exact bit-for-bit equality between incremental Finny-table NNUE updates and scratch evaluation across 3,923 positions.
- `.\final\Gemini4Neonchess24hrs.exe datagen --out <file.bin> --pos <N> --threads <T> --nodes <K> --seed <S>` — generates compact 32-byte self-play training positions.

### Exposed UCI Options

| Option | Type | Default | Range | Description |
|---|---|---|---|---|
| `Hash` | spin | `64` | `1–4096` | Transposition table size in MB (rating harness sets `256`) |
| `Clear Hash` | button | — | — | Clears the transposition table |
| `Move Overhead` | spin | `50` | `0–500` | Safety margin (ms) subtracted from clock budgets to prevent timeouts |
| `EvalFile` | string | `<empty>` | — | Optional external `.nnue` file override (uses embedded Gen 5 NNUE by default) |
| `Threads` | spin | `1` | `1–1` | Single-threaded search |

---

## Architecture and Features

### Board Representation & Move Generation
- **Bitboards + Mailbox** (`source/src/bitboard.h/.cpp`, `position.h/.cpp`, `movegen.h/.cpp`): 6 piece-type bitboards, 2 color bitboards, and a 64-square mailbox array for O(1) piece lookup.
- **BMI2 PEXT Slider Attacks**: Precomputed bishop and rook attack tables indexed via `_pext_u64`.
- **Lazy Pin & Check-Square Computation**: `do_move` eagerly computes only `checkersBB`, deferring `blockersForKing` and `checkSquares` via a bitmask (`pinReady`) until `is_legal()` or `gives_check()` actually needs them, while `do_null_move` skips check computation entirely. This achieved **289.6M NPS** on the 126-position `perft.epd` D1–D5 suite (395.67M nodes in 1.37s) and a pure +5.4% search NPS boost.
- **Incremental Zobrist Hashing**: Maintains 6 independent 64-bit hashes on every move: main position key, pawn key, white non-pawn key, black non-pawn key, minor-piece key, and major-piece key.
- **Full Chess Legality**: Castling, en passant (including horizontal-pin discovery checks), all four promotions, checkmate/stalemate, 50-move rule (with checkmate precedence), insufficient material, and two/threefold repetition detection.

### Evaluation: Self-Play NNUE (`(4x768 -> 768)x2 -> 8` SCReLU)
- **Architecture (`source/src/nnue_arch.h`, `nnue.h/.cpp`)**:
  - Perspective feature transformer with **4 horizontally mirrored king buckets** (`4 × 768 = 3,072` input features per perspective) -> **`768` int16 hidden units** per perspective (`QA = 255`), **Squared Clipped ReLU (SCReLU)** activation, and **8 piece-count output buckets** (`QB = 64`, `SCALE = 400`).
  - 64-byte cache-line aligned `QuantizedNet` (`4,744,832` bytes), embedded directly into the executable via `.incbin`.
  - **AVX2 SIMD inference**: 4-accumulator unrolled `_mm256_mullo_epi16` + `_mm256_madd_epi16` output dot product, lazy incremental accumulator updates, **zero-copy `accPly`** across null moves, and **Finny tables** (`[2][2][4]`) with batched 4-feature diffs on king-bucket/mirror crossings.
  - Material-based phase scaling (`(22709 + mat) / 32768`, SPSA-tuned) and 50-move rule damping.
  - Tapered handcrafted evaluation (`source/src/eval.h/.cpp`) retained as fallback and used solely to bootstrap Gen 1a self-play.
- **Self-Play Data & Training Pipeline (`source/src/datagen.cpp`, `source/trainer/trainer.cpp`)**:
  - Custom C++20 14-thread AVX2 sparse **AdamW** trainer with cosine LR decay, weight clipping `[-1.98, 1.98]`, mixed eval/WDL sigmoid MSE loss, and best-validation checkpointing (running at 530k–1.9M pos/s on 14 CPU cores).
  - **5 progressive self-play generations** generated strictly from scratch from random legal plies off `startpos` (zero external datasets/books):
    - `gen1a.bin`: 6.0M positions (`v4` HCE, `softNodes=4000`) — used to bootstrap Gen 1c, then dropped.
    - `gen2a.bin`: 10.0M positions (`gen1c` ~3285 Elo, `softNodes=5000`).
    - `gen3a.bin`: 10.0M positions (`v6_gen2a` ~3380 Elo, `softNodes=5000`).
    - `gen4a.bin`: 10.0M positions (`gen3a_k200` ~3440 Elo, `softNodes=7000`).
    - `gen4b.bin`: 15.0M positions (`gen4a` ~3470 Elo, `softNodes=6000`).
    - `gen5a.bin`: 20.0M positions (`v7b_gen4b` ~3505 Elo, `softNodes=6000`, mixed 6/7/8/9 random opening plies, clean `|score| < 2500` filter).
  - Final Champion Net (`gen5_4x768_65m.nnue`) trained for 8 cosine epochs (`lr 0.0007 -> 1e-5`, `K=175`, `wdl=0.35`, `wd=0.012`) on all **65,006,624 pure NNUE positions** (`gen2a`–`gen5a`).

### Search, Move Ordering & Time Management
- **Transposition Table (`source/src/tt.h/.cpp`)**: 64-byte cache-line aligned **5-entry buckets**, Lemire 128-bit fast-range indexing (upper bits) paired with independent **32-bit lower-key verification (`key32`)**, generation+depth replacement, exact duplicate prevention, `ttPv` tracking, and raw static eval caching.
- **Principal Variation Search (`source/src/search.h/.cpp`)**:
  - Iterative deepening with **Aspiration Windows** (fail-high depth reduction, snap-to-infinity above `delta > 500`).
  - **Pre-move pruning**: Reverse Futility Pruning (improving + `ttPv` + correction-delta scaled), Razoring, Null Move Pruning (zero-copy `accPly` + high-depth verification), ProbCut, Internal Iterative Reductions (IIR), and Hindsight LMR reduction/extension.
  - **Move-loop pruning & reductions**: Late Move Pruning (LMP), per-move history-adjusted Futility Pruning, History Pruning, Quiet & Capture SEE Pruning, **1024-scaled fractional LMR** with deeper/shallower re-search and continuation history feedback, and **Singular Extensions** (multi-cut, double/triple/negative extensions).
  - **Quiescence Search**: Stand-pat with TT clamping, delta/futility pruning, SEE >= 0 capture filtering, and full check evasions.
- **Move Ordering & History (`source/src/movepicker.h`)**:
  - Staged `MovePicker`: TT move -> Good Captures (MVV + `capHist`, SEE threshold adjusted by `capHist`) -> 2 Killer moves -> Countermove -> Quiet moves -> Bad Captures.
  - **Quiet History**: Pawn-threat-indexed `mainHist[2][64][64][4]`, `pawnHist`, and **1/2/4/6-ply Continuation History (`contHist`)**, updated with linear-depth bonus/malus (including quiet malus on tactical cutoffs and move-1 refutation malus).
  - **7-Table Correction History**: `pawnCorr`, White/Black `nonPawnCorr`, `minorCorr`, `majorCorr`, 1-ply `prevMoveCorr`, and 2-ply `contCorr` (clamped `±260 cp`).
- **SPSA Tuning (`source/src/tune.h`, `source/tests/spsa.py`)**: 18 core search, eval-scaling (`EvalMatBase`), and time-management parameters tuned over 220 SPSA iterations (6,160 games at `4+0.04`) and hardcoded as compile-time `constexpr` constants in the production build (**+19.1 ± 28.6 Elo** verified over 200 games at `10+0.1`).
- **Time Management & Protocol Reliability (`source/src/uci.cpp`, `search.cpp`)**:
  - Dedicated search thread with main-thread stdin reader (`40/40` `fastchess --compliance` pass, prompt `stop`/`isready`/`quit`).
  - Dynamic soft/hard limits (`safeTime / 22.0 + 0.80 * inc`) modulated by root node-fraction, best-move stability (11-step table), and score-drop extensions, with a `50 ms` `Move Overhead` safety buffer.

### Deliberately Left Out (and Why)
- **Opening book & endgame tablebases**: Disallowed / unnecessary per benchmark rules.
- **Multi-threading (Lazy SMP) & Pondering**: Benchmark evaluation is strictly single-threaded without pondering.
- **Wider/more-bucketed NNUEs (`4x1024` SCReLU, `4x1024` Pairwise, `8x768`)**: All were implemented, trained, and tested head-to-head; `4x768 -> 768` SCReLU consistently won on Elo-per-second (`2.20M NPS`) up through 65M positions.
- **GCC PGO (`-fprofile-use`)**: Implemented and benchmarked on GCC 15.2.0 MinGW, but measured 4–8.5% *slower* than `-O3 -march=x86-64-v3 -flto -flto-partition=one` (`2.04M` vs `2.20M NPS`), so `-flto -flto-partition=one` was selected instead.

---

## How the Time Was Spent

### Phase Summary
- **Hours 0–1 (`20:50–21:40`)**: Built complete bitboard/PEXT board, movegen (126/126 `perft.epd` pass in 5 minutes at 225.7M NPS), clustered TT, tapered HCE, PVS search, and threaded UCI. Caught and fixed a catastrophic TT hash-collision bug (`+641 Elo`), beat Stash-30, shipped Search Pack 1 (`v4`, `+24.2 Elo` SPRT), and built the entire AVX2 NNUE inference + trainer + 14-thread datagen pipeline.
- **Hours 1–4 (`21:40–00:51`)**: Generated Gen 1a (6M HCE) and Gen 2a (10M NNUE) datasets. Ablated `1x512` vs `4x512` vs `4x768` vs `8x768` and WDL schedules (`gen1c` gained `+140 Elo` over HCE; `gen2a_768` gained another `+76.5 Elo`). Shipped Search Pack 2 (`v5`, `+45.1 Elo` SPRT in 372 games) and Search Pack 3 (`v6`, `+5.2 Elo`).
- **Hours 4–10 (`00:51–06:47`)**: Generated Gen 3a (10M) and Gen 4a (10M @ 7000 nodes) -> 30M pure NNUE positions. Measured the dataset's true WDL logistic scale (`K ≈ 180–200`, not `400`), fixed a redundant MovePicker history lookup (`+6.9%` NPS), and surpassed Stash-37 for the first time (`gen3a_k200`: `+23.2` vs Stash-37; `gen4a`: `+61.4` over `gen3a_k200` and `+34.9` vs Stash-37). Tested `4x1024 Pairwise` (`-13 Elo` vs `4x768` SCReLU -> rejected).
- **Hours 10–14 (`06:47–10:58`)**: Generated Gen 4b (15M positions -> 45M pure NNUE). `gen4b_768` beat `gen4a` by `+43.7 Elo` and beat `gen4b_1024` by `+21.7 Elo`. Implemented lazy `checkInfo` + batched Finny refresh (`+5.4%` pure NPS speedup, perft jumped to 289.9M NPS). Rejected aggressive Search Pack 4 (`v7`: `-35.2 Elo` SPRT) and shipped surgical Search Pack 4a (`v7b`: `+11.6 Elo`, `-24.7%` bench-13 nodes).
- **Hours 14–19 (`10:58–16:12`)**: Generated Gen 5a (20M positions with mixed 6/7/8/9 opening plies and clean `|score| < 2500` filter -> **65M pure NNUE positions**). Trained and matched 5 Champion Gen 5 candidates (`65m 4x768`, `55m 4x768`, `65m 4x1024`, Stage-2 fine-tune `ft35m`, and 10-epoch `wdl=0.45`). `gen5_768_65m` won every head-to-head match and crushed Stash-37 by **`+81.4 ± 50.4 Elo`** (100 games, LOS `99.95%`).
- **Hours 19–24 (`16:12–20:25`)**: Ran **SPSA Round 1** (220 iterations / 6,160 games, 18 params -> verified **`+19.1 ± 28.6 Elo`** over 200 games at `10+0.1`), benchmarked `-flto-partition=one` (`+4.0%` NPS) vs PGO (rejected), ran **SPSA Round 2** (140 iterations / 3,920 games -> `-3.5 ± 28.1` at `10+0.1` -> reverted), locked in the final clean binary, passed all 5 verification gates, and completed 400 final calibration games (`300` vs Stash-37, `100` vs Stash-33) with zero timeouts or crashes.

### What Was Measured vs. Accepted on Judgement
- **Measured with fastchess / SPRT**: Every major NNUE generation, every architecture/hyperparameter choice (`1x512`, `4x512`, `4x768`, `8x768`, `4x1024` SCReLU/Pairwise, 55M vs 65M, `wdl=0.35` vs `0.45`, Stage-2 fine-tuning), every search pack (`v4`, `v5`, `v6`, `v7` rejected, `v7b`), both SPSA rounds (Round 1 kept, Round 2 reverted), and all external Stash calibrations.
- **Measured with exact node-count benchmarks**: Pure non-functional speedups (MovePicker lookup elimination `+6.9%`, lazy `checkInfo` + batched Finny `+5.4%`, `-flto-partition=one` `+4.0%`, and PGO `-4% to -8.5%` rejected).
- **Accepted on engineering judgement**: `Move Overhead = 50 ms` (verified with 0 timeouts across 1,500+ games at strict `timemargin=0`) and datagen random-ply mix / score filtering (validated downstream by Gen 5's `+60 Elo` jump vs Stash-37).

### What Went Wrong & What Was Cut
1. **TT Hash Collision Bug (Hour 1)**: Initial `key16 = key >> 48` overlapped the upper bits used by Lemire indexing, causing massive collisions (`-382` vs Stash-25). Switching to bottom-32-bit `key32` in 5-entry buckets immediately recovered **+641 Elo**.
2. **Correction History Oscillation (Hour 1)**: Initial HCE correction update oscillated until bounded with gravity updates and `±260 cp` clamping.
3. **Aggressive Search Pack 4 (`v7`, Hour 14)**: Bundling 5 pruning/reduction tightenings cut bench nodes 34% and lost `-35.2 Elo` -> immediately reverted and replaced by surgical `v7b` (`+11.6 Elo`).
4. **GCC 15 PGO Regression (Hour 22)**: `-fprofile-use` reduced NPS by 4–8.5% on MinGW GCC 15.2.0 -> rejected in favor of `-flto -flto-partition=one` (`+4.0%` NPS).
5. **Cut**: Multi-layer L2/L3 NNUE heads and a third 20M datagen cycle were deliberately cut to reserve the final 5 hours for SPSA tuning, build verification, and a 400-game final stress test.

---

## Elo by Hour

| Hour | Timestamp | Estimated CCRL Blitz Elo | Basis / Milestone |
|---|---|---|---|
| Hour 0 | 2026-11-23 20:51 | Not yet measurable | Workspace setup, toolchain & hardware audit |
| Hour 1 | 2026-11-23 21:40 | ~3220 | Perft 126/126, TT fix (+641), beat Stash-30 (`+23`), `v4` SPRT (`+24`) |
| Hour 2 | 2026-11-23 22:38 | ~3285 | Gen 1a 6M datagen, `gen1c` `4x512` NNUE (`+140` vs `v4`, `-140` vs Stash-37) |
| Hour 3 | 2026-11-24 00:08 | ~3285 | Running Gen 2a 10M NNUE self-play datagen |
| Hour 4 | 2026-11-24 00:51 | ~3395 | `gen2a_768` (`+77` vs `gen1c`, `-89` vs Stash-37) + `v5` Search Pack 2 (`+45` SPRT) |
| Hour 5 | 2026-11-24 01:57 | ~3380 | `v6` Search Pack 3 (`+5` vs `v5`, `-47 ± 64` vs Stash-37); Gen 3a datagen |
| Hour 6 | 2026-11-24 02:52 | ~3380 | Completed Gen 3a 10M datagen (20M pure NNUE corpus) |
| Hour 7 | 2026-11-24 04:29 | ~3440 | True `K=200, wdl=0.35` + MovePicker fix -> **beat Stash-37 (`+23 ± 58`)** |
| Hour 8 | 2026-11-24 05:30 | ~3440 | Running Gen 4a deeper 10M datagen (`softNodes=7000`) |
| Hour 9 | 2026-11-24 06:25 | ~3440 | Completed Gen 4a 10M (30M pure NNUE), trained `gen4a` (`val_loss 0.01425`) |
| Hour 10 | 2026-11-24 06:47 | ~3470 | `gen4a` `+61` vs `gen3a_k200`, `+35 ± 57` vs Stash-37; launched Gen 4b 15M |
| Hour 11 | 2026-11-24 07:47 | ~3470 | Running Gen 4b 15M self-play datagen (31.6%) |
| Hour 12 | 2026-11-24 08:47 | ~3470 | Running Gen 4b 15M self-play datagen (61.9%) |
| Hour 13 | 2026-11-24 09:48 | ~3470 | Running Gen 4b 15M self-play datagen (93.2%) |
| Hour 14 | 2026-11-24 10:48 | ~3495 | `gen4b_768` (45M) `+44` vs `gen4a`, `+21 ± 54` vs Stash-37; lazy `checkInfo` (`+5.4%` NPS) |
| Hour 15 | 2026-11-24 11:58 | ~3505 | `v7b` Search Pack 4a (`+12` over 600g); launched Gen 5a 20M datagen (25.7%) |
| Hour 16 | 2026-11-24 12:59 | ~3505 | Running Gen 5a 20M self-play datagen (51.9%, `1,440 pos/s`) |
| Hour 17 | 2026-11-24 14:00 | ~3505 | Running Gen 5a 20M self-play datagen (78.2%) |
| Hour 18 | 2026-11-24 14:51 | ~3505 | Completed Gen 5a 20M (**65M pure NNUE corpus**); training Champion Gen 5 nets |
| Hour 19 | 2026-11-24 15:46 | ~3510 | **`gen5_768_65m`** `+81.4 ± 50.4` vs Stash-37 (100g, `LOS 99.95%`) |
| Hour 20 | 2026-11-24 17:15 | ~3510 | Locked Champion Net; running SPSA Round 1 (100/220 iters, 18 params) |
| Hour 21 | 2026-11-24 18:15 | ~3510 | SPSA Round 1 nearing completion (205/220 iters) |
| Hour 22 | 2026-11-24 19:37 | ~3530 | SPSA Round 1 verified **`+19.1 ± 28.6`** (200g) + `-flto-partition=one` (`+4%` NPS); SPSA R2 |
| Hour 23–24 | 2026-11-24 20:25 | **~3515** | SPSA R2 reverted; final lock-in & 400g calibration (`+87.5 ± 26.9` vs Stash-37, `+230.2 ± 65.4` vs Stash-33) |

---

## Assumptions Made

- **Identity**: Model name `Gemini 4 Neon`, UCI `id name` = `Gemini 4 Neon chess 24hrs`, `id author` = `Gemini 4 Neon`, executable `final/Gemini4Neonchess24hrs.exe` (spaces removed).
- **From-scratch NNUE**: Self-play positions generated exclusively during this 24-hour session from random legal plies off `startpos` (no external books, datasets, or pre-existing weights) and embedded into the single static `.exe` satisfy the from-scratch and standalone rules.
- **Target hardware**: Modern Windows 11 laptop supporting `x86-64-v3` (POPCNT, BMI1, BMI2 fast PEXT, AVX2) without AVX-512; compiled strictly with `-march=x86-64-v3` (never `-march=native`).
- **Evaluation harness & time margin**: Rating match runs `tc=10+0.1`, single-threaded, `Hash=256`, with undisclosed `timemargin`. We assumed worst-case **`timemargin=0`** across all testing and set default `Move Overhead = 50 ms` to guarantee zero timeouts on any Windows 11 machine.
- **Concurrency**: Development machine has 16 physical cores (Intel Core Ultra 9 285H, 31.43 GB RAM), so all fastchess matches and datagen jobs were capped at `physical cores - 2 = 14`.
- **Rating anchors**: Stash CCRL Blitz (`2'+1"`) ratings (`stash-33` ~3286, `stash-37` ~3424) were used as approximate anchors for `10+0.1` estimates.

---

## Estimated Strength

**Estimated strength: ~3515 CCRL Blitz (95% CI: ~3485–3545)**

### Direct Calibration Evidence (`tc=10+0.1`, 1 thread, `Hash=256`, `timemargin=0`, `UHO.pgn`, `-repeat`)

| Opponent | Anchor (CCRL Blitz) | Games | Score (W–L–D) | Elo Diff (95% CI) | LOS | Timeouts / Crashes | Implied Rating |
|---|---|---|---|---|---|---|---|
| `stash-37.0` | ~3424 | 300 | `187.0 / 300` (`62.33%`, `135–61–104`) | **`+87.5 ± 26.9`** | `100.00%` | `0 / 0` | **~3512** |
| `stash-33.0` | ~3286 | 100 | `79.0 / 100` (`79.00%`, `69–11–20`) | **`+230.2 ± 65.4`** | `100.00%` | `0 / 0` | **~3516** |

- **Pentanomial vs Stash-37 (300 games)**: `[2, 16, 59, 52, 21]` (`PairsRatio = 4.06`, 21 double-wins vs 2 double-losses).
- **Pentanomial vs Stash-33 (100 games)**: `[1, 0, 10, 18, 21]` (`PairsRatio = 39.00`).
- **Verification gates on final binary**: 126/126 `perft.epd` pass (`289.6M NPS`), 3,923/3,923 exact NNUE incremental-vs-scratch matches, 40/40 `fastchess --compliance` pass, verified standalone with `PATH=C:\Windows\System32;C:\Windows`, and **0 timeouts, 0 crashes, 0 illegal moves across all 400 final calibration games at strict `timemargin=0`**.

### Uncertainty & Caveats
1. **Sampling error**: The 300-game match against Stash-37 has a 95% confidence interval of `±26.9 Elo`, and the 100-game match against Stash-33 has `±65.4 Elo`.
2. **Time-control & pool scaling**: CCRL Blitz is played at `2'+1"` against a broad pool of diverse engines, whereas our calibration was run at `10+0.1` against the Stash engine family using the UHO opening book. Relative strength against NNUE-based opponents at `10+0.1` on a different Windows 11 laptop could easily shift the true rating by `±30–50 Elo` relative to this Stash-anchored estimate.

---

## Official Results

*This section and the badges at the top were added by the benchmark operator after the run. Everything else is the model's own write-up.*

**3470 ± 14 Elo (95% CI) on the CCRL Blitz scale**, from 2,540 games played on 24 November 2026 (+629 =1036 −875, 45.2%). That is second of the ten engines built under this benchmark so far: 75 Elo behind Fable 5.2 and 11 ahead of Opus 5.5.

The engine had no time loss, crash, disconnect, illegal move or fastchess warning in any of the 2,540 games.

### Conditions

- fastchess 1.8.2, **10 s + 0.1 s**, `timemargin=200`, one thread, **64 MB hash**. (The benchmark rules announced 256 MB hash, and the model tested at 256 MB with `timemargin=0`. The rating run uses 64 MB for every engine.)
- UHO unbalanced 8-ply openings in random order, every opening played with both colours.
- Adjudication: resign after 3 moves at ±600 cp from both engines, draw after move 40 with 8 moves inside ±20 cp, 250 moves maximum.
- 12 concurrent games on a Ryzen AI 9 HX 375 (12 cores), a different laptop from the Intel Core Ultra 9 285H the engine was developed on.
- A 72-game pilot against Stash 37, Patricia and Elixir came first. The rated games were two gauntlets against ten reference engines (1,800 games), then 740 games against the four nearest AI-written engines. Fable 5.2 is the main rival, so that pairing was played over 320 games, twice the usual number.
- Ratings come from an anchored maximum-likelihood fit over all 15,140 games in the series. The reference engines are fixed at their CCRL Blitz ratings, and every game involves at least one of the AI-written engines.

### All engines from this benchmark

| Rank | Engine | Elo | 95% CI | Games |
|---:|---|---:|:---:|---:|
| 1 | Fable 5.2 chess 24hrs | 3545 | ±14 | 2540 |
| **2** | **Gemini 4 Neon chess 24hrs** | **3470** | **±14** | **2540** |
| 3 | Opus 5.5 chess 24hrs | 3459 | ±15 | 2500 |
| 4 | Sonnet 5.5 chess 24hrs | 3414 | ±15 | 2500 |
| 5 | Fable 5.1 chess 24hrs | 3251 | ±18 | 1780 |
| 6 | Opus 5 chess 24hrs | 3227 | ±19 | 1580 |
| 7 | Astra 6 chess 24hrs | 3142 | ±19 | 1520 |
| 8 | Fable 5 chess 24hrs | 3044 | ±21 | 1300 |
| 9 | Haiku 5.5 chess 24hrs | 2765 | ±24 | 1060 |
| 10 | Sonnet 5 chess 24hrs | 2703 | ±23 | 1360 |

The 11-Elo margin over Opus 5.5 is inside the error bars of both engines. Gemini 4 Neon did win their direct match, 55.3% over 160 games.

### Head-to-head

| Opponent | CCRL Blitz | +W | =D | −L | Score | Implied Elo |
|---|---:|---:|---:|---:|---:|---:|
| Coda 0.9.4 | 3638 | 0 | 46 | 114 | 14.4% | 3328 |
| Clover 9.0 | 3620 | 2 | 29 | 129 | 10.3% | 3244 |
| Elixir 3.0 | 3567 | 16 | 75 | 69 | 33.4% | 3447 |
| Patricia 5.0 | 3539 | 36 | 77 | 87 | 37.2% | 3448 |
| Sirius 9.0 | 3528 | 58 | 83 | 59 | 49.8% | 3526 |
| Carp 3.0.1 | 3525 | 43 | 67 | 50 | 47.8% | 3510 |
| Tucano 12.00 | 3491 | 51 | 85 | 64 | 46.8% | 3468 |
| Leorik 3.2.1 | 3490 | 43 | 79 | 38 | 51.6% | 3501 |
| Marvin 6.3.0 | 3458 | 69 | 89 | 42 | 56.8% | 3505 |
| Stash 37 | 3424 | 96 | 71 | 33 | 65.8% | 3537 |
| Fable 5.2 chess 24hrs | (3545) | 38 | 155 | 127 | 36.1% | – |
| Opus 5.5 chess 24hrs | (3459) | 48 | 81 | 31 | 55.3% | – |
| Sonnet 5.5 chess 24hrs | (3414) | 55 | 77 | 28 | 58.4% | – |
| Fable 5.1 chess 24hrs | (3251) | 74 | 22 | 4 | 85.0% | – |

Fable 5.2 won the main match clearly. A 36.1% score over 320 games is about 99 ± 27 Elo, and the two 160-game halves agreed (37.5% and 34.7%).

### How firm the number is

The figure depends on which reference engines are trusted. Clover and Coda play roughly 250 Elo above their CCRL Blitz ratings relative to the other anchors under these conditions, so the 10–14% scored against them pulls the fit down. Fable 5.2 was rated against the same ten anchors and is affected in the same way.

| Fit | Gemini 4 Neon | Fable 5.2 |
|---|---:|---:|
| All games (the official figures) | 3470 ± 14 | 3545 ± 14 |
| Without Clover and Coda | 3492 ± 15 | 3574 ± 15 |
| Anchor games only | 3469 ± 16 | 3532 ± 16 |
| Anchor games only, without Clover and Coda | 3494 ± 18 | 3565 ± 18 |

Against the other eight anchors the implied rating runs from 3447 (Elixir) to 3537 (Stash 37). The two gauntlets differ for the same reason: the first, which includes Clover and Coda, gives 3430 ± 26 (3487 ± 31 without them), and the second gives 3497 ± 22. The games against the other AI-written engines agree with the anchors and barely move the estimate.

One caveat applies to the comparison with Fable 5.2. Its anchor games were played on the Intel Core Ultra 9 285H and Gemini 4 Neon's on the Ryzen. The 320 direct games were all played on the Ryzen, and they show a slightly wider gap than the fit does.

### Against the model's own estimate

The model predicted about 3515 (3485–3545) from 300 games against Stash 37 and 100 against Stash 33. The official 3470 is 15 Elo below that range, and the fit without the two off-scale anchors, 3492, is inside it. Its Stash 37 result held up: 65.8% here against 62.3% in its own test. Stash 37 was simply the most favourable of the ten anchors, and the non-Stash engines near 3500 rate Gemini 4 Neon lower. The write-up's second caveat anticipated this, allowing a shift of 30–50 Elo against NNUE opponents on a different laptop.
