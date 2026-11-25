#include "datagen.h"
#include "dataformat.h"
#include "position.h"
#include "movegen.h"
#include "search.h"
#include "eval.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <iomanip>

namespace Datagen {

namespace {

const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

PackedPos pack_position(const Position& pos, int scoreWhite, uint8_t wdl, uint16_t ply) {
    PackedPos pp{};
    Bitboard occ = pos.pieces();
    pp.occ = occ;
    int idx = 0;
    while (occ && idx < 32) {
        Square sq = pop_lsb(occ);
        uint8_t pc = uint8_t(pos.piece_on(sq));
        if ((idx & 1) == 0) pp.pieces[idx >> 1] = pc & 0xF;
        else                pp.pieces[idx >> 1] |= uint8_t((pc & 0xF) << 4);
        idx++;
    }
    pp.stm = uint8_t(pos.side_to_move());
    pp.wdl = wdl;
    pp.score = int16_t(std::clamp(scoreWhite, -30000, 30000));
    pp.ply = ply;
    pp.reserved = 0;
    return pp;
}

} // namespace

void run(const std::string& outPath, uint64_t targetPositions,
         int numThreads, uint64_t softNodes, uint64_t seed) {
    std::ofstream out(outPath, std::ios::binary | std::ios::app);
    if (!out) {
        std::cerr << "Datagen: cannot open output file " << outPath << "\n";
        return;
    }

    std::atomic<uint64_t> totalPos{0};
    std::atomic<uint64_t> totalGames{0};
    std::atomic<bool> done{false};
    std::mutex fileMtx;

    auto tStart = std::chrono::steady_clock::now();

    std::cout << "Datagen starting: target=" << targetPositions
              << " threads=" << numThreads
              << " softNodes=" << softNodes
              << " seed=" << seed
              << " out=" << outPath << "\n" << std::flush;

    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    for (int tid = 0; tid < numThreads; ++tid) {
        workers.emplace_back([&, tid]() {
            std::mt19937_64 rng(seed + uint64_t(tid + 1) * 0x9E3779B97F4A7C15ULL);
            Searcher searcher;
            searcher.set_hash(2); // 2 MB per worker thread (fits in L2/L3 cache, instant clear!)

            std::vector<StateInfo> history;
            history.reserve(1024);
            std::vector<PackedPos> gameBuf;
            gameBuf.reserve(256);
            std::vector<PackedPos> flushBuf;
            flushBuf.reserve(16384);

            SearchLimits lim;
            lim.softNodes = softNodes;
            lim.nodes = softNodes * 6; // safety hard cap
            lim.depth = 20;

            while (!done.load(std::memory_order_relaxed)) {
                Position pos;
                history.clear();
                history.push_back(StateInfo{});
                pos.set_fen(START_FEN, &history.back());

                // Mixed opening diversity: 40% 6-7 random plies (structured middlegames) + 60% 8-9 plies (tactical diversity)
                int rPick = int(rng() % 10);
                int randPlies = (rPick < 2) ? 6 : (rPick < 4 ? 7 : (rPick < 7 ? 8 : 9));
                bool validStart = true;
                for (int r = 0; r < randPlies; ++r) {
                    MoveList ml;
                    MoveGen::generate_legal(pos, ml);
                    if (ml.size() == 0) { validStart = false; break; }
                    Move rm = ml.moves[rng() % ml.size()].move;
                    history.push_back(StateInfo{});
                    pos.do_move(rm, history.back());
                }
                if (!validStart || pos.is_draw(0)) continue;

                // Quick filter: reject already-crushed random starts
                SearchLimits filterLim;
                filterLim.depth = 5;
                filterLim.softNodes = 1500;
                filterLim.nodes = 5000;
                searcher.clear_hash();
                SearchResult fRes = searcher.search(pos, filterLim, true);
                if (fRes.bestMove == MOVE_NONE || std::abs(fRes.score) > 650) continue;
                searcher.clear_state();

                gameBuf.clear();
                int winStreak = 0;
                int drawStreak = 0;
                uint8_t gameWdl = 1; // default draw
                int gamePly = 0;

                while (gamePly < 400) {
                    MoveList ml;
                    MoveGen::generate_legal(pos, ml);
                    if (ml.size() == 0) {
                        if (pos.in_check()) {
                            gameWdl = (pos.side_to_move() == WHITE) ? 0 : 2;
                        } else {
                            gameWdl = 1;
                        }
                        break;
                    }
                    if (pos.is_draw(0)) {
                        gameWdl = 1;
                        break;
                    }

                    SearchResult res = searcher.search(pos, lim, true);
                    if (res.bestMove == MOVE_NONE) {
                        gameWdl = 1;
                        break;
                    }

                    int scoreWhite = (pos.side_to_move() == WHITE) ? res.score : -res.score;

                    // Record clean quiet non-check informative positions
                    if (!pos.in_check() && !is_tactical(res.bestMove)
                        && std::abs(res.score) < 2500 && gamePly >= 2) {
                        gameBuf.push_back(pack_position(pos, scoreWhite, 1, uint16_t(randPlies + gamePly)));
                    }

                    // Win adjudication
                    if (scoreWhite >= 1800) {
                        winStreak = (winStreak > 0) ? winStreak + 1 : 1;
                        if (winStreak >= 4) { gameWdl = 2; break; }
                    } else if (scoreWhite <= -1800) {
                        winStreak = (winStreak < 0) ? winStreak - 1 : -1;
                        if (winStreak <= -4) { gameWdl = 0; break; }
                    } else {
                        winStreak = 0;
                    }

                    // Draw adjudication
                    if (gamePly >= 50 && std::abs(scoreWhite) <= 12) {
                        drawStreak++;
                        if (drawStreak >= 8) { gameWdl = 1; break; }
                    } else {
                        drawStreak = 0;
                    }

                    history.push_back(StateInfo{});
                    pos.do_move(res.bestMove, history.back());
                    gamePly++;
                }

                // Stamp final game WDL onto all recorded positions from this game
                for (auto& pp : gameBuf) {
                    pp.wdl = gameWdl;
                    flushBuf.push_back(pp);
                }

                totalGames.fetch_add(1, std::memory_order_relaxed);
                uint64_t newTotal = totalPos.fetch_add(gameBuf.size(), std::memory_order_relaxed) + gameBuf.size();

                if (flushBuf.size() >= 8192 || newTotal >= targetPositions) {
                    std::lock_guard<std::mutex> lock(fileMtx);
                    out.write(reinterpret_cast<const char*>(flushBuf.data()),
                              flushBuf.size() * sizeof(PackedPos));
                    flushBuf.clear();
                }

                if (newTotal >= targetPositions) {
                    done.store(true, std::memory_order_relaxed);
                    break;
                }
            }

            if (!flushBuf.empty()) {
                std::lock_guard<std::mutex> lock(fileMtx);
                out.write(reinterpret_cast<const char*>(flushBuf.data()),
                          flushBuf.size() * sizeof(PackedPos));
                flushBuf.clear();
            }
        });
    }

    // Monitor progress every 10 seconds
    while (!done.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        uint64_t posCnt = totalPos.load(std::memory_order_relaxed);
        uint64_t gCnt   = totalGames.load(std::memory_order_relaxed);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
        double rate = posCnt / std::max(secs, 1e-3);
        double eta  = (targetPositions > posCnt && rate > 1.0) ? (targetPositions - posCnt) / rate : 0.0;
        std::cout << "[Datagen] games=" << gCnt << " pos=" << posCnt << "/" << targetPositions
                  << " (" << std::fixed << std::setprecision(1) << (100.0 * posCnt / targetPositions) << "%)"
                  << " rate=" << uint64_t(rate) << " pos/s"
                  << " elapsed=" << int(secs) << "s ETA=" << int(eta) << "s\n" << std::flush;
    }

    for (auto& w : workers) w.join();
    out.flush();
    out.close();

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
    std::cout << "Datagen finished: " << totalPos.load() << " positions from "
              << totalGames.load() << " games in " << secs << "s ("
              << uint64_t(totalPos.load() / std::max(secs, 1e-3)) << " pos/s)\n" << std::flush;
}

} // namespace Datagen
