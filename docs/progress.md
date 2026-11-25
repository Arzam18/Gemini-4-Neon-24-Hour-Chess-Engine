# 24-Hour Chess Engine Benchmark — Progress Log

## Identity & Environment
- **Model**: Gemini 4 Neon
- **Engine Name (`id name`)**: `Gemini 4 Neon chess 24hrs`
- **Author (`id author`)**: `Gemini 4 Neon`
- **Executable**: `final/Gemini4Neonchess24hrs.exe`
- **Start Time**: `2026-11-23T20:50:17.9929705-05:00`
- **Deadline**: `2026-11-24T20:50:17-05:00`
- **Hardware**: Intel Core Ultra 9 285H (16 physical cores / 16 logical processors), 31.43 GB RAM, NVIDIA GeForce RTX 5080 Laptop GPU (16 GB VRAM)
- **Max Concurrency (`physical cores - 2`)**: `14`
- **Compiler**: GCC / G++ 15.2.0 (MinGW-w64), target `-march=x86-64-v3 -O3 -flto -static`

## Assumptions
- Model name derived from benchmark workspace and configuration (`Gemini 4 Neon`); executable name `Gemini4Neonchess24hrs.exe` follows the rule of removing spaces and dots.
- Self-play training data for any NNUE will be generated strictly from scratch in this session using random-ply openings from startpos (no external datasets, pre-existing weights, or opening books used for training), and weights will be embedded directly into the standalone executable.
- Target machine supports `x86-64-v3` (POPCNT, BMI1, BMI2 fast PEXT, AVX2) without AVX-512.
- Rating harness runs `tc=10+0.1` with `setoption name Hash value 256` and undisclosed `timemargin`, so we will test with strict `timemargin=0` to guarantee zero timeouts.

## Milestones ("Firsts")
- **First full perft suite pass (`resources/perft/perft.epd`)**: `2026-11-23 20:55` (126/126 D1–D5, 395.67M nodes in 1.75s = 225.7M NPS)
- **First full legal chess game played**: `2026-11-23 21:01` (v1 vs Stash-25 at 10+0.1)
- **First verified build deployed to `final/`**: `2026-11-23 21:10` (v2, 0 timeouts/crashes), upgraded to `v3` at `2026-11-23 21:15` (+23 ± 56 Elo vs Stash-30, ~3190 Elo)

---

## Hourly Log

### Hour 0 — 2026-11-23 20:51
- Initialized workspace (`docs/`, `source/`, `final/`), recorded start time, verified GCC 15.2.0, fastchess 1.8.2, Stash calibration ladder, and 16 physical cores (`-concurrency 14`).
- Planned architecture: C++20 bitboard engine with BMI2 PEXT slider attacks, modern PVS search + NNUE self-play bootstrapping pipeline (HCE -> NNUE Gen 1 -> Gen 2 -> Gen 3).
- Current Elo estimate: not yet measurable.
- Plan for Hour 1: Implement board representation, PEXT move generator, Zobrist hashing, make/unmake, and pass all 126 positions in `resources/perft/perft.epd`.

### Hour 1 — 2026-11-23 21:40
- Built full bitboard + BMI2 PEXT move generator (126/126 perft D1–D5 passed in 1.75s = 225.7M NPS).
- Built clustered TT, tapered HCE, staged MovePicker, PVS search (LMR, NMP, RFP, ProbCut, Singular Extensions, Correction History), and threaded UCI (40/40 fastchess compliance).
- Diagnosed and fixed a critical TT hash-collision bug (`key16 = key >> 48` overlapped Lemire index top bits) -> jumped **+641 Elo** (from -382 to +260 vs Stash-25) and beat **Stash-30** (`+23.2 ± 55.7 Elo`, 32/60, 0 timeouts/crashes at `10+0.1` `Hash=256` `timemargin=0`).
- Shipped Search Upgrade Pack 1 (`v4`: Continuation CorrHist, 1024-scaled fractional LMR, Pawn History, NMP verification, calibrated TM, `Move Overhead=50ms`) -> **SPRT H1 accepted (+24.2 ± 15.5 Elo over `v3`, 834 games, LLR=3.00)**; deployed to `final/Gemini4Neonchess24hrs.exe` and snapshotted.
- Built & verified full NNUE pipeline: AVX2 SCReLU `(4x768 -> 512)x2 -> 8` with Finny tables (3,923/3,923 exact incremental-vs-scratch matches, ~2.4M NPS), C++20 multi-threaded AVX2 AdamW trainer, and 14-thread self-play `datagen` (~2,500 clean pos/s).
- Current Elo estimate: **~3220 CCRL Blitz** (based on `v3` +23 vs Stash-30 ~3166 plus `v4` +24 SPRT gain).
- Plan for Hour 2: Complete Gen 1a self-play datagen (6M positions, ~40 min), train NNUE Gen 1a (`(4x768 -> 512)x2 -> 8`), embed and test against `v4` HCE and Stash-33 (~3286) / Stash-37 (~3424), then launch NNUE-powered Gen 2 datagen.

### Hour 2 — 2026-11-23 22:38
- Completed Gen 1a HCE self-play datagen: **6,001,451** filtered positions from 71,038 games in 38 min (`2,605 pos/s`).
- Ran controlled NNUE architecture/schedule ablation on the 6M dataset (training at 1.1M–1.9M pos/s on 14 AVX2 cores):
  - `gen1a` (`4x768 -> 512`, 20ep un-annealed Ep4, `wdl=0.25`): `+23.2 ± 60.4 Elo` vs `v4` HCE (`2.28M NPS`).
  - `gen1b` (`1x768 -> 512`, 5ep cosine, `wdl=0.50`): `-58.5 ± 77.6 Elo` vs `v4` HCE (`2.50M NPS`) — proved king buckets + lower WDL weight are essential.
  - `gen1c` (`4x768 -> 512`, 6ep cosine `0.0012->1e-5`, `wdl=0.10`, `wd=0.012`): **`+140.4 ± 62.5 Elo` vs `v4` HCE** (`41.5/60`, 0 pair losses, `2.34M NPS`, fewer depth-12 nodes than HCE!).
- Deployed `gen1c` (standalone embedded binary) to `final/Gemini4Neonchess24hrs.exe`, verified 40/40 compliance + standalone execution, saved snapshot.
- Calibrated `gen1c` against **Stash-37 (~3424 CCRL Blitz)** at `10+0.1`, `Hash=256`, `timemargin=0`: **`-140.4 ± 62.5 Elo`** (`18.5/60`, 0 timeouts/crashes) -> **~3285 CCRL Blitz** (Stash-33 level) at Hour 1.8!
- Current Elo estimate: **~3285 CCRL Blitz** (`-140.4 ± 62.5` vs Stash-37 ~3424).
- Plan for Hour 3: Run Gen 2a NNUE-powered self-play datagen (`10,000,000` positions with `gen1c` at `softNodes=5000`, ~70 min), followed by Gen 2b (another 10M positions -> 20M pure NNUE positions) and train a stronger Gen 2 `(8x768 -> 768)` / `(4x768 -> 768)` network to surpass Stash-37!

### Hour 3 — 2026-11-24 00:08
- Continued running Gen 2a NNUE-powered self-play datagen (`10,000,000` positions at `softNodes=5000` on 14 cores using `gen1c`).
- Current deployable build in `final/`: `gen1c` (`~3285` CCRL Blitz, `-140.4 ± 62.5` vs Stash-37).
- Plan for Hour 4: Finish Gen 2a datagen, immediately train Gen 2a NNUE (`4x768->512`, `4x768->768`, `8x768->768` ablations), test vs `gen1c` and Stash-37, deploy new best net to `final/`, and upgrade search (Search Pack 2) before launching Gen 2b/Gen 3 datagen.

### Hour 4 — 2026-11-24 00:51
- Completed Gen 2a NNUE self-play datagen: **10,001,313** filtered positions from 115,388 games (`softNodes=5000`, `1,684 pos/s`).
- Added 64-byte cache-line alignment to `QuantizedNet`; trained 3 Gen 2a candidates and selected best validation loss: `4x768 -> 768` on 16M combined positions (`gen1a + gen2a`, `val_loss=0.005658`).
- `gen2a_768` beat `gen1c` by **`+76.5 ± 55.6 Elo`** (`36.5/60`, `LOS 99.75%`), and improved vs **Stash-37** to **`-88.7 ± 69.0 Elo`** (`22.5/60`, 0 timeouts/crashes) -> ~3345 CCRL Blitz.
- Shipped **Search Pack 2 (`v5`)** (4-accumulator unrolled AVX2 `forward_l1`, zero-copy `accPly` across null moves, 6-ply contHist, `(ss-4)` improving fallback, hindsight LMR, fixed FP `skip_quiets`, capHist LMR/SEE, NNUE qsearch delta, draw randomization, aspiration snap) -> **SPRT H1 accepted (`+45.1 ± 21.4 Elo` over `gen2a_768` in 372 games, `LLR=2.96`, 0 timeouts)**; deployed `v5` to `final/Gemini4Neonchess24hrs.exe` and snapshotted.
- Currently testing **Search Pack 3 (`v6`)** (pawn-threat-indexed quiet history, material-scaled NNUE, tighter singular margin + triple extensions, move-1 refutation malus) via SPRT vs `v5`.
- Current Elo estimate: **~3390–3400 CCRL Blitz** (based on `gen2a_768` -89 vs Stash-37 ~3424 plus `v5` +45 SPRT gain).
- Plan for Hour 5+: Finalize `v6` (keep if passing, revert if not), calibrate `final/` vs Stash-37 (expecting ~even / beating Stash-37!), then launch **Gen 3 massive self-play datagen** (~30M–50M positions from our ~3400+ Elo engine in clean chunks) to push to 3550–3650+ Elo!

### Hour 5 — 2026-11-24 01:57
- Finalized **Search Pack 3 (`v6`)**: `+5.2 ± 16.0 Elo` vs `v5` (800 games, `4+0.04`, 0 timeouts/crashes), deployed to `final/Gemini4Neonchess24hrs.exe` and snapshotted.
- Calibrated `v6_gen2a` vs **Stash-37 (~3424 CCRL Blitz)** at `10+0.1`, `Hash=256`, `timemargin=0`: **`-46.6 ± 63.9 Elo`** (`26.0/60 = 43.3%`, 0 timeouts by us, 1 timeout by Stash-37).
- Running **Gen 3a** self-play datagen (`10,000,000` positions with ~3380 Elo `v6_gen2a` at `softNodes=5000` on 14 cores).
- Current Elo estimate: **~3380 CCRL Blitz** (`-46.6 ± 63.9` vs Stash-37 ~3424).
- Plan for Hour 6: Complete `gen3a.bin` (10M positions), then immediately launch `gen3b.bin` (another 10M positions -> 30M pure NNUE positions total with `gen2a`), and prepare Gen 3 `(8x768 -> 768)` / `(10x768 -> 1024)` training.

### Hour 6 — 2026-11-24 02:52
- Completed **Gen 3a** self-play datagen: **10,001,340** filtered positions from 113,696 games of ~3380 Elo `v6_gen2a` (`1,457 pos/s`).
- Total dataset now: **20,002,653** pure NNUE positions (`gen2a` 10M + `gen3a` 10M), plus 6M initial HCE positions (`gen1a`).
- Current deployable build in `final/`: `v6_gen2a` (~3380 CCRL Blitz, `-46.6 ± 63.9` vs Stash-37).
- Plan for Hour 7: Train Gen 3a (`8x768 -> 768` vs `4x768 -> 768`) on 20M pure NNUE positions (dropping HCE data completely), benchmark & match vs `v6_gen2a` and **Stash-37 (~3424)**, deploy winner to `final/`, and launch Gen 3b datagen with the stronger Gen 3a engine.

### Hour 7 — 2026-11-24 04:29
- Trained Gen 3a NNUE (`4x768 -> 768` vs `8x768 -> 768`) on 20M pure NNUE positions (`gen2a + gen3a`, dropping HCE data): `4x768 -> 768` won validation (`0.005411` vs `0.005524`) and beat `v6_gen2a` by `+29.0 ± 56.7 Elo` (`32.5/60`).
- Measured true WDL logistic scale of `gen3a.bin` (`K ≈ 180–220`, NOT `400`, with WDL-MSE dropping 16% from `0.0603` in `gen2a` to `0.0509` in `gen3a`). Eliminated redundant `quiet_history_score` lookup in `search.cpp` (`+6.9%` NPS).
- Trained properly cosine-annealed `gen3a_k200_wdl35` (`--scale 200 --wdl 0.35 --lr 0.0006 --wd 0.02`, 5 epochs, smooth monotonic val drop to `0.014390`): **BEAT Stash-37 (~3424 CCRL Blitz) by `+23.2 ± 58.1 Elo`** (`22W 18L 20D = 32.0/60`, 0 timeouts/crashes at `10+0.1` `Hash=256` `timemargin=0`)!
- Deployed `gen3a_k200_wdl35` to `final/Gemini4Neonchess24hrs.exe`, verified 40/40 compliance, saved snapshot, and launched **Gen 4a** deeper self-play datagen (`10,000,000` positions at `softNodes=7000` across 14 cores).
- Updated `nnue_arch.h` `KING_BUCKETS == 8` to a back-rank-focused layout (4 dedicated 1st-rank buckets, 2 2nd-rank buckets, 1 mid-rank, 1 endgame bucket) for Gen 4 training.
- Current Elo estimate: **~3435–3445 CCRL Blitz** (`+23.2 ± 58.1` vs Stash-37 ~3424).
- Plan for Hours 8–10: Finish `gen4a.bin` (10M @ 7000 nodes), launch `gen4b.bin` (another 10M @ 7000 nodes -> 40M total pure NNUE positions), then train Gen 4 (`8_backrank x 768 -> 1024` vs `4x768 -> 1024` vs `8_backrank x 768 -> 768`) to push past **3500+ CCRL Blitz**!

### Hour 8 — 2026-11-24 05:30
- Continued **Gen 4a** deeper self-play datagen (`10,000,000` positions at `softNodes=7000` across 14 cores using `gen3a_k200_wdl35`).
- Current deployable build in `final/`: `gen3a_k200_wdl35` (~3435–3445 CCRL Blitz, `+23.2 ± 58.1` vs Stash-37).
- Plan for Hour 9: Complete `gen4a.bin` (~23 min left -> **30M total pure NNUE positions** across `gen2a + gen3a + gen4a`!), immediately train Gen 4a comparing `8_backrank x 768 -> 768` vs `4x768 -> 1024` vs `8_backrank x 768 -> 1024`, test vs `gen3a_k200` and Stash-37, deploy new best net, and launch Gen 4b datagen with the stronger Gen 4a engine!

### Hour 9 — 2026-11-24 06:25
- Completed **Gen 4a** deeper self-play datagen: **10,001,223** filtered positions from 110,182 games (`softNodes=7000`, `1,155 pos/s`). Dataset WDL-MSE improved another 10% (`0.0603` -> `0.0509` -> **`0.0457`**).
- Tested `8_backrank x 768` vs `4x768 -> 768` on 30M pure NNUE positions (`gen2a + gen3a + gen4a`):
  - `8_backrank x 768` (9.46 MB, `val_loss=0.014646`, `1.76M NPS`) lost `-23.2` vs `gen3a_k200`.
  - `4x768 -> 768` on 30M (`K=180, wdl=0.35`, 6ep cosine) converged monotonically every epoch down to **`val_loss=0.014247`** (beating 20M `gen3a+gen4a` `0.014749` by `-0.00050`!).
- Currently running 80-game matches at `10+0.1`, `Hash=256`, `timemargin=0` of `gen4a` (`4x768` 30M) vs `gen3a_k200` and **Stash-37 (~3424)**.
- Current Elo estimate: **~3435–3445 CCRL Blitz** (`final/` = `gen3a_k200_wdl35`, awaiting `gen4a` results).
- Plan for Hours 10–14: Deploy `gen4a` to `final/` if stronger, scale `4x768` vs `4x1024` and push to **50M–70M pure NNUE positions** (`gen4b`, `gen5a`, `gen5b`) + **Search Pack 4 + SPSA tuning + PGO**!

### Hour 10 — 2026-11-24 06:47
- Verified **`gen4a` (`4x768 -> 768` SCReLU on 30M pure NNUE positions, `K=180, wdl=0.35`, `val_loss=0.014247`, `2.01M NPS`)**:
  - **`+61.4 ± 50.9 Elo` vs `gen3a_k200`** (`47.0/80 = 58.75%`, `LOS 99.25%`, 0 timeouts/crashes).
  - **`+34.9 ± 56.5 Elo` vs Stash-37 (~3424 CCRL Blitz)** (`44.0/80 = 55.0%`, `LOS 89.0%`, 0 timeouts/crashes at `10+0.1` `Hash=256` `timemargin=0`).
  - Deployed `gen4a` to `final/Gemini4Neonchess24hrs.exe`, verified 40/40 compliance + standalone execution, saved snapshot `2026-11-24_0632_gen4a_4x768_30m`.
- Tested `gen4a_pw1024` (`4x768 -> 1024` Pairwise CReLU, 6.31 MB, `val_loss=0.014244`, `1.91M NPS`) vs `gen4a`: `-13.0 ± 44.4 Elo` (`38.5/80`, 0 timeouts) — kept `gen4a` SCReLU 768 in `final/`, will re-test wider/pairwise nets once corpus reaches 45M–65M positions.
- Launched **Gen 4b** self-play datagen (`15,000,000` positions with ~3470 Elo `engine_gen4a.exe` at `softNodes=6000` across 14 cores -> bringing pure NNUE corpus to **45M positions**).
- Current Elo estimate: **~3465–3475 CCRL Blitz** (`+34.9 ± 56.5` vs Stash-37 ~3424, `+61.4` over `gen3a_k200`).
- Plan for Hours 11–14: Complete `gen4b.bin` (15M positions, ~3h), train & evaluate Gen 4b (`4x768 -> 768` vs `4x768 -> 1024` SCReLU on 45M pos), deploy winner to `final/`, ship Search Pack 4, then run Gen 5a datagen (20M positions -> 65M total).

### Hour 11 — 2026-11-24 07:47
- **Gen 4b** self-play datagen (`gen4b.bin`, `softNodes=6000`, 14 cores using ~3470 Elo `engine_gen4a.exe`) running smoothly: **4.74M / 15.0M** positions (31.6%, 53,358 games, steady `1,311 pos/s`, ETA ~2h 10m).
- Current deployable build in `final/`: `gen4a` (`4x768 -> 768` SCReLU, ~3465–3475 CCRL Blitz, `+34.9 ± 56.5` vs Stash-37).
- Plan for Hour 12: Continue `gen4b.bin` datagen (~63% by Hour 12, completion ~09:57 AM -> 45M pure NNUE positions ready for Gen 4b training + Search Pack 4).

### Hour 12 — 2026-11-24 08:47
- **Gen 4b** self-play datagen (`gen4b.bin`, `softNodes=6000`, 14 cores) at **9.28M / 15.0M** positions (61.9%, 104,478 games, `1,283 pos/s`, ETA ~74 min -> completion ~10:02 AM).
- Current deployable build in `final/`: `gen4a` (`4x768 -> 768` SCReLU, ~3465–3475 CCRL Blitz, `+34.9 ± 56.5` vs Stash-37).
- Plan for Hour 13: Complete `gen4b.bin` (**45M pure NNUE positions** total across `gen2a+gen3a+gen4a+gen4b`), measure `gen4b` WDL-MSE / optimal `K`, train & compare `4x768 -> 768` vs `4x768 -> 1024` SCReLU, test winner vs `gen4a` and Stash-37, deploy to `final/`, then implement & SPRT Search Pack 4 before Gen 5a datagen.

### Hour 13 — 2026-11-24 09:48
- **Gen 4b** self-play datagen (`gen4b.bin`, `softNodes=6000`, 14 cores) nearing completion: **13.97M / 15.0M** positions (93.2%, 157,295 games, `1,287 pos/s`, ETA ~13 min).
- Pre-compiled `trainer_4x1024.exe` (`4x768 -> 1024` SCReLU) so 45M-position Gen 4b training starts immediately upon completion.
- Current deployable build in `final/`: `gen4a` (`4x768 -> 768` SCReLU, ~3465–3475 CCRL Blitz, `+34.9 ± 56.5` vs Stash-37).
- Plan for Hour 14: Complete `gen4b.bin` (45M pure NNUE positions), train & match `4x768 -> 768` vs `4x768 -> 1024` SCReLU, deploy winner to `final/`, implement & SPRT Search Pack 4 (including mean-1.0 material scale normalization), and launch Gen 5a 20M datagen (~65M total corpus).

### Hour 14 — 2026-11-24 10:48
- Completed **Gen 4b** self-play datagen: **15,001,629** clean positions from 168,829 games (`1,287 pos/s`, WDL-MSE `0.0461`, balanced W/D/L `32.5/34.5/33.0%`) -> **45.0M total pure NNUE positions**.
- Trained and compared `4x768 -> 768` SCReLU (`val_loss=0.013871`, down `-0.00038` vs `gen4a`, `2.08M NPS`) vs `4x768 -> 1024` SCReLU (`val_loss=0.013949`, `1.88M NPS`):
  - **`gen4b_768` beat `gen4a` by `+43.7 ± 40.3 Elo`** (`45.0/80 = 56.25%`, `LOS 98.45%`, 0 pair losses, 0 timeouts/crashes), and beat `gen4b_1024` by `+21.7 ± 44.1 Elo`.
  - Calibrated `gen4b_768` vs **Stash-37 (~3424 CCRL Blitz)** across 100 games (`10+0.1`, `Hash=256`, `timemargin=0`): **`+20.9 ± 53.9 Elo`** (`53.0/100`, 0 timeouts by us, 1 by Stash-37).
- Implemented lazy `blockersForKing`/`checkSquares` in `position.cpp` (plus zero-cost `do_null_move` and batched Finny refresh in `nnue.cpp`): verified 126/126 perft pass (**289.9M NPS**, +28.4%), 3,923/3,923 `testnnue`, and identical `bench 13` node count at **2.12M NPS (+5.4% pure speedup)**; deployed `gen4b_fast` to `final/Gemini4Neonchess24hrs.exe` (40/40 compliance) and snapshotted `2026-11-24_1032_gen4b_4x768_45m`.
- Tested aggressive Search Pack 4 (`v7`) via SPRT (`-35.2 ± 22.5 Elo`, H0 accepted -> immediately reverted); now SPRT-testing surgical Search Pack 4a (`v7b`: linear history bonus, tactical-cutoff quiet malus, 1-ply `prevMoveCorr`, TT eval preservation + exact duplicate prevention, aspiration fail-high depth reduction).
- Current Elo estimate: **~3485–3505 CCRL Blitz** (`gen4b_768` +44 over `gen4a` + `+5.4%` pure NPS speedup; combined `gen3a_k200/gen4a/gen4b` vs Stash-37 = `+26` across 240 games).
- Plan for Hours 15–18: Finalize `v7b` (and quick material-scale test), then launch **Gen 5a** 20M self-play datagen (mixed 6/8/9 random plies + cleaner score cap -> **65M total pure NNUE positions**!) before Champion Gen 5 training, SPSA tuning, and PGO.

### Hour 15 — 2026-11-24 11:58
- Finalized **Search Pack 4a (`v7b`)**: **`+11.6 ± 17.0 Elo`** over `v6_gen4b_fast` across 600 games (`310.0/600 = 51.67%`, `LOS 90.90%`, 0 timeouts/crashes, `-24.7%` bench-13 nodes); deployed to `final/Gemini4Neonchess24hrs.exe` (40/40 compliance + standalone PATH verified) and snapshotted `2026-11-24_1058_v7b_gen4b_4x768`.
- **Gen 5a** self-play datagen (`gen5a.bin`, 20M positions, `softNodes=6000`, mixed 6/7/8/9 random plies, `|score| < 2500` filter, 14 cores) running fast at **`1,432 pos/s`**: **5.14M / 20.0M** positions (25.7%, 60,051 games, ETA ~2h 53m -> completion ~14:51).
- Current Elo estimate: **~3500–3515 CCRL Blitz** (`gen4b_768` + `+5.4%` lazy-checkInfo speedup + `v7b` +12 Elo).
- Plan for Hour 16: Continue `gen5a.bin` datagen (~51% by Hour 16, completion ~14:51 -> **65.0M pure NNUE positions** total).

### Hour 16 — 2026-11-24 12:59
- **Gen 5a** self-play datagen (`gen5a.bin`, 20M positions, `softNodes=6000`, 14 cores) past halfway: **10.39M / 20.0M** positions (51.9%, 121,558 games, steady **`1,440 pos/s`**, ETA ~1h 51m -> completion ~14:50).
- Current deployable build in `final/`: `v7b_gen4b` (`4x768 -> 768` SCReLU, ~3500–3515 CCRL Blitz).
- Plan for Hour 17: Continue `gen5a.bin` datagen (~78% by Hour 17, completion at Hour 18.0 -> **65.0M pure NNUE positions** ready for Champion Gen 5 training, SPSA search/TM tuning, and PGO).

### Hour 17 — 2026-11-24 14:00
- **Gen 5a** self-play datagen (`gen5a.bin`, 20M positions, `softNodes=6000`, 14 cores) nearing completion: **15.65M / 20.0M** positions (78.2%, 183,043 games, **`1,442 pos/s`**, ETA ~50 min -> completion at ~14:50, exact Hour 18.0).
- Current deployable build in `final/`: `v7b_gen4b` (`4x768 -> 768` SCReLU, ~3500–3515 CCRL Blitz).
- Plan for Hours 18–20: Complete `gen5a.bin` (**65.0M pure NNUE positions**!), train & match Champion Gen 5 NNUE candidates (`4x768 -> 768` vs `4x768 -> 1024` SCReLU with stronger weight decay, 65M vs 55M high-Elo subset, optimal `K`/`wdl`), deploy Champion Net, then run automated SPSA Search + Eval-Scale + TM tuning (200–300 iters) + PGO build + final 200-game Stash-37 calibration!

### Hour 18 — 2026-11-24 14:51
- Completed **Gen 5a** self-play datagen: **20,001,119** clean positions from 233,967 games of `v7b_gen4b` (`1,440 pos/s`, balanced W/D/L `32.2/34.6/33.1%`, `|sc|<50 = 21.7%`, clean `|score|<2500` filter) -> **65.0M total pure NNUE positions** (`gen2a + gen3a + gen4a + gen4b + gen5a`).
- Launched **Champion Gen 5 NNUE** 8-epoch cosine training suite (comparing 65M `4x768`, 55M high-Elo `4x768`, and 65M `4x1024` SCReLU with stronger `wd=0.025`, all validated on the identical 200k `gen5a` holdout).
- Current deployable build in `final/`: `v7b_gen4b` (`4x768 -> 768` SCReLU, ~3500–3515 CCRL Blitz).
- Plan for Hour 19: Finish Gen 5 NNUE training + WDL-weight ablation (`wdl=0.45`), run 100-game matches at `10+0.1` (`Hash=256`) to pick and deploy the Champion Gen 5 Net, calibrate vs Stash-37, and start automated SPSA Search + Eval-Scale + Time Management tuning!

### Hour 19 — 2026-11-24 15:46
- Evaluated all three Gen 5 NNUE candidates (`10+0.1`, `Hash=256`, `timemargin=0`):
  - `gen5_768_55m` (`val_loss=0.014860`) beat `v7b_gen4b` by **`+43.7 ± 43.9 Elo`** (`45.0/80 = 56.25%`, `LOS 97.62%`, 0 pair losses).
  - **`gen5_768_65m`** (`val_loss=0.014895`, all 65M positions) beat `gen5_768_55m` by **another `+43.7 ± 47.3 Elo`** (`45.0/80 = 56.25%`, `LOS 96.72%`, `PairsRatio 3.00`) — keeping all 65M multi-generation positions dramatically improves generalization!
  - `gen5_768_55m` beat `gen5_1024_65m` (`val_loss=0.014847`, `1.86M NPS`) by `+13.0 ± 37.1 Elo`.
- Deployed **`gen5_768_65m`** to `final/Gemini4Neonchess24hrs.exe` (40/40 compliance + standalone verified), snapshotted `2026-11-24_1541_gen5_4x768_65m`, and calibrated vs **Stash-37 (~3424 CCRL Blitz)** across 100 games (`10+0.1`, `Hash=256`, `timemargin=0`):
  - **CRUSHED Stash-37 by `+81.4 ± 50.4 Elo`** (**`61.5 / 100`**, `45W 22L 33D`, **`LOS 99.95%`**, `PairsRatio 3.00`, `Ptnml: [1, 7, 18, 16, 8]`, **0 timeouts/crashes**)!
- Now training two final Champion Gen 5 NNUE polish candidates (`gen5_ft35m`: 4-epoch low-LR Stage-2 fine-tune on 45M strongest `gen4a+gen4b+gen5a` with `wdl=0.40`; and `gen5_65m_10ep_wdl45`: 10-epoch cosine on all 65M with `wdl=0.45`).
- Current Elo estimate: **~3505–3525 CCRL Blitz** (`+81.4 ± 50.4` vs Stash-37 ~3424 across 100 games, `LOS 99.95%`).
- Plan for Hours 20–24: Test Gen 5 polish nets & lock in Champion Net -> run **220–250 iterations of automated SPSA Search + Eval-Scale + TM tuning** on 14 cores -> verify via 200-game SPRT -> build **PGO binary** (+10–15% NPS) -> final 200-game Stash-37 calibration & zero-timeout stress test -> lock in `final/` and write `README.md`!

### Hour 20 — 2026-11-24 17:15
- Locked in **`gen5_4x768_65m.nnue`** as our undisputed Champion Gen 5 NNUE (`gen5_ft35m` scored `-17.4 ± 52.7` and `gen5_10ep_wdl45` scored `-34.9 ± 46.1` against it at `10+0.1`, `Hash=256`).
- Wired 18 search, eval-scale (`EvalMatBase`), and time-management parameters behind `#ifdef SPSA_TUNE` in `tune.h` (compile-time `constexpr` in production builds, zero runtime overhead) and launched a **220-iteration SPSA tuning run** (`6,160` games at `4+0.04`, `Hash=64`, `timemargin=0`, 14 cores).
- SPSA progress at Hour 20.4: **100 / 220 iterations** (`2,800` games, `~35s/iter`, ETA ~70 min -> completion ~18:25); clear directional trends already visible (`EvalMatBase 24500->23073`, `LmrBase 75->72`, `LmrDiv 220->226`, `HistPruneMult 2000->2251`, `FutBase 100->107`, `TmDiv 23->22`).
- Current deployable build in `final/`: `gen5_768_65m` (`~3505–3525 CCRL Blitz`, `+81.4 ± 50.4` vs Stash-37).
- Plan for Hours 21–24: Finish SPSA tuning (~18:25) -> hardcode tuned constants into production `tune.h` -> verify via 200-game `10+0.1` match vs untuned `gen5_768_65m` -> build **PGO binary** (+10–15% NPS) & match-verify -> final 200-game Stash-37 calibration & zero-timeout stress test (`timemargin=0`, `Hash=256`) -> compliance/standalone/perft verification -> final snapshot & `README.md` after `20:50`.

### Hour 21 — 2026-11-24 18:15
- **SPSA tuning** (`6,160` games, 18 search/eval/TM parameters on 14 cores) nearly complete at **205 / 220 iterations** (`5,740` games played, ETA ~8 min); parameters tightly converged (`EvalMatBase 24500->22779`, `LmrBase 75->72`, `LmrDiv 220->227`, `HistPruneMult 2000->2351`, `FutBase 100->109`, `FutMult 90->87`, `SeeQuietMult 22->20`, `NmpEvalDiv 180->175`, `SingMarginNum 15->16`, `TmDiv 23->22`, `TmSoftPct 72->73`).
- Current deployable build in `final/`: `gen5_768_65m` (`~3505–3525 CCRL Blitz`, `+81.4 ± 50.4` vs Stash-37).
- Plan for Hours 22–24: Hardcode SPSA-tuned constants into clean production `tune.h` -> run 200-game `10+0.1` verification match vs untuned `gen5_768_65m` -> deploy if stronger -> build **Profile-Guided Optimization (PGO)** standalone binary -> verify NPS & match strength -> run **200-game final calibration & zero-timeout stress test vs Stash-37 (`10+0.1`, `Hash=256`, `timemargin=0`)** -> final compliance/standalone/perft checks & snapshot -> `README.md` after `20:50`.

### Hour 22 — 2026-11-24 19:37
- Completed **SPSA Round 1** (`220` iterations, `6,160` games, 18 search/eval/TM parameters) and verified hardcoded production binary `gen5_spsa` vs untuned `gen5_base` across **200 games at `10+0.1`, `Hash=256`, `timemargin=0`**:
  - **`+19.1 ± 28.6 Elo`** (`105.5/200 = 52.75%`, `LOS 90.60%`, 5 double-wins vs 1 double-loss, 0 timeouts/crashes); deployed to `final/` and snapshotted `2026-11-24_1831_gen5_spsa_4x768_65m`.
- Benchmarked compiler optimization flags on `bench 15` (identical `2,386,183` nodes):
  - `-flto`: `2.12M NPS`; `-flto-partition=one -fprofile-use` (GCC 15 PGO): `2.04M NPS` (rejected — slower); **`-flto -flto-partition=one`: `2.20M NPS` (+4.02% pure free speedup!)** -> deployed to `final/Gemini4Neonchess24hrs.exe` (40/40 compliance).
- Launched **SPSA Round 2** on 12 additional untuned parameters (LMR modifiers, QSearch futility, CorrHist weights, TM hard/node scaling): currently at **100 / 140 iterations** (`2,800` games, ETA ~23 min -> completion ~19:59).
- Current Elo estimate: **~3525–3545 CCRL Blitz** (`gen5_768_65m` +81 vs Stash-37 + SPSA Round 1 `+19` + `-flto-partition=one` `+4%` NPS).
- Plan for Hours 23–24: Finish SPSA Round 2 (~19:59) -> verify via 200-game `10+0.1` match vs Round 1 -> deploy winner to `final/` -> run **240-game final calibration & zero-timeout stress test vs Stash-37 (`10+0.1`, `Hash=256`, `timemargin=0`)** -> full compliance / standalone PATH / `perft.epd` 126/126 verification & final snapshot -> lock in `final/` and write `README.md` at `20:50`!

### Hour 23–24 (Final Lock-in) — 2026-11-24 20:25
- Completed **SPSA Round 2** (`140` iterations, `3,920` games, 12 additional LMR/QSearch/CorrHist/TM parameters) and verified against SPSA Round 1 across 200 games (`10+0.1`, `Hash=256`, `timemargin=0`): `-3.5 ± 28.1 Elo` (`99.0/200`, 0 timeouts) -> reverted Round 2 parameters to the proven Round 1 SPSA baseline.
- Built and locked the clean final standalone executable at `final/Gemini4Neonchess24hrs.exe` (`8,633,289` bytes, `g++ 15.2.0`, `-std=c++20 -O3 -march=x86-64-v3 -flto -flto-partition=one -static -DNDEBUG -pthread`, embedded `gen5_4x768_65m.nnue`) and saved final snapshot `source/snapshots/2026-11-24_2009_final_gen5_spsa_65m_lto1`.
- Passed **every final verification gate**:
  - `resources/perft/perft.epd` (D1–D5): **126 / 126 passed**, `395,673,136` nodes in `1.37s` (`289.6M NPS`).
  - NNUE incremental-vs-scratch (`testnnue`): **3,923 / 3,923 exact matches** (0 mismatches).
  - `fastchess --compliance final/Gemini4Neonchess24hrs.exe`: **40 / 40 passed**.
  - Standalone stripped-`PATH` (`C:\Windows\System32;C:\Windows`) test: exact `id name Gemini 4 Neon chess 24hrs`, `id author Gemini 4 Neon`, `uciok`, `readyok`, `bestmove`.
- Completed **400-game final calibration & zero-timeout stress test** at exact evaluation settings (`tc=10+0.1`, `1t`, `Hash=256`, `timemargin=0`, `UHO.pgn`, `-repeat`):
  - **vs Stash-37 (~3424 CCRL Blitz)** (300 games): **`+87.5 ± 26.9 Elo`** (`187.0/300 = 62.33%`, `135W 61L 104D`, **`LOS 100.00%`**, `Ptnml: [2, 16, 59, 52, 21]`, **0 timeouts, 0 crashes, 0 illegal moves**) -> `~3512 CCRL Blitz`.
  - **vs Stash-33 (~3286 CCRL Blitz)** (100 games): **`+230.2 ± 65.4 Elo`** (`79.0/100 = 79.00%`, `69W 11L 20D`, **`LOS 100.00%`**, `Ptnml: [1, 0, 10, 18, 21]`, **0 timeouts, 0 crashes, 0 illegal moves**) -> `~3516 CCRL Blitz`.
- Final Elo estimate: **~3515 CCRL Blitz** (95% CI: **~3485–3545**, zero timeouts/crashes across all 400 final calibration games).
- Final deliverable locked in `final/Gemini4Neonchess24hrs.exe`; wrote public `README.md`.
