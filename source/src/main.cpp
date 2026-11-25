#include "bitboard.h"
#include "zobrist.h"
#include "eval.h"
#include "nnue.h"
#include "uci.h"
#include "datagen.h"
#include "position.h"
#include "movegen.h"
#include <iostream>
#include <string>
#include <random>
#include <memory>

static int test_nnue_incremental(const std::string& netPath) {
    if (!NNUE::load_file(netPath)) {
        std::cerr << "Failed to load NNUE: " << netPath << "\n";
        return 1;
    }
    auto accStack = std::make_unique<NNUE::AccumulatorStack>();
    std::mt19937_64 rng(999);
    int checks = 0, mismatches = 0;

    for (int g = 0; g < 50; ++g) {
        Position pos;
        StateInfo hist[256];
        pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &hist[0]);
        accStack->reset_finny();
        accStack->reset_root(pos);

        for (int ply = 0; ply < 80; ++ply) {
            int inc = accStack->evaluate(pos, ply);
            int scr = NNUE::evaluate_scratch(pos);
            checks++;
            if (inc != scr) {
                std::cerr << "MISMATCH g=" << g << " ply=" << ply << " inc=" << inc << " scr=" << scr
                          << " fen=" << pos.to_fen() << "\n";
                mismatches++;
                if (mismatches >= 5) return 1;
            }
            MoveList ml;
            MoveGen::generate_legal(pos, ml);
            if (ml.size() == 0) break;
            Move m = ml.moves[rng() % ml.size()].move;
            accStack->push_move(pos, m, ply);
            pos.do_move(m, hist[ply + 1]);
        }
    }
    std::cout << "NNUE incremental test: " << checks << " positions checked, " << mismatches << " mismatches.\n";
    return mismatches == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Bitboards::init();
    Zobrist::init();
    Eval::init();
    NNUE::init_embedded();

    if (argc > 1) {
        std::string cmd = argv[1];
        if (cmd == "bench") {
            int d = (argc > 2) ? std::stoi(argv[2]) : 13;
            UCI::run_bench(d);
            return 0;
        } else if (cmd == "datagen") {
            std::string outPath = "source/data/gen1.bin";
            uint64_t targetPos = 15000000;
            int threads = 14;
            uint64_t softNodes = 5000;
            uint64_t seed = 20261123ULL;
            std::string netPath;
            for (int i = 2; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--out" && i + 1 < argc) outPath = argv[++i];
                else if ((a == "--pos" || a == "--positions") && i + 1 < argc) targetPos = std::stoull(argv[++i]);
                else if (a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
                else if (a == "--nodes" && i + 1 < argc) softNodes = std::stoull(argv[++i]);
                else if (a == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
                else if (a == "--net" && i + 1 < argc) netPath = argv[++i];
            }
            if (!netPath.empty()) {
                if (!NNUE::load_file(netPath)) {
                    std::cerr << "Failed to load NNUE for datagen: " << netPath << "\n";
                    return 1;
                }
                std::cout << "Datagen loaded NNUE: " << netPath << "\n";
            }
            Datagen::run(outPath, targetPos, threads, softNodes, seed);
            return 0;
        } else if (cmd == "testnnue" && argc > 2) {
            return test_nnue_incremental(argv[2]);
        }
    }

    UCI::loop();
    return 0;
}
