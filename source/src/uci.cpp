#include "uci.h"
#include "tune.h"
#include "position.h"
#include "search.h"
#include "eval.h"
#include "nnue.h"
#include "movegen.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cctype>

namespace UCI {

namespace {

const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

std::string to_lower(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

class UciEngine {
public:
    UciEngine() {
        history.reserve(8192);
        reset_position(START_FEN);
    }

    ~UciEngine() {
        stop_and_join();
    }

    void stop_and_join() {
        searcher.request_stop();
        if (worker.joinable()) worker.join();
    }

    void reset_position(const std::string& fen) {
        history.clear();
        history.push_back(StateInfo{});
        pos.set_fen(fen, &history.back());
    }

    void handle_position(std::istringstream& iss) {
        stop_and_join();
        std::string token;
        if (!(iss >> token)) return;

        std::string fen;
        if (token == "startpos") {
            fen = START_FEN;
            iss >> token; // possibly "moves"
        } else if (token == "fen") {
            while (iss >> token && token != "moves") {
                if (!fen.empty()) fen += ' ';
                fen += token;
            }
        } else {
            return;
        }

        reset_position(fen);

        if (token == "moves") {
            while (iss >> token) {
                Move m = pos.parse_uci_move(token);
                if (m == MOVE_NONE) break;
                // Reserve safely so pointers don't invalidate
                if (history.size() + 2 >= history.capacity()) {
                    // Rebuild linked pointers safely by storing moves; 4096 capacity handles any chess game
                    break;
                }
                history.push_back(StateInfo{});
                pos.do_move(m, history.back());
            }
        }
    }

    void handle_go(std::istringstream& iss) {
        stop_and_join();

        SearchLimits limits;
        std::string token;
        while (iss >> token) {
            if (token == "wtime")      { iss >> limits.time[WHITE]; limits.useTime = true; }
            else if (token == "btime") { iss >> limits.time[BLACK]; limits.useTime = true; }
            else if (token == "winc")  { iss >> limits.inc[WHITE]; }
            else if (token == "binc")  { iss >> limits.inc[BLACK]; }
            else if (token == "movestogo") { iss >> limits.movestogo; }
            else if (token == "movetime")  { iss >> limits.movetime; }
            else if (token == "depth")     { iss >> limits.depth; }
            else if (token == "nodes")     { iss >> limits.nodes; }
            else if (token == "infinite")  { limits.infinite = true; }
        }

        // Launch single search thread
        worker = std::thread([this, limits]() {
            SearchResult res = searcher.search(pos, limits, false, &outMtx);
            std::string line = "bestmove " + move_to_uci(res.bestMove);
            if (res.ponderMove != MOVE_NONE) {
                line += " ponder " + move_to_uci(res.ponderMove);
            }
            line += '\n';
            std::lock_guard<std::mutex> lock(outMtx);
            std::cout << line << std::flush;
        });
    }

    void handle_setoption(std::istringstream& iss) {
        stop_and_join();
        std::string token, name, value;
        bool inName = false, inValue = false;
        while (iss >> token) {
            if (token == "name") { inName = true; inValue = false; continue; }
            if (token == "value") { inName = false; inValue = true; continue; }
            if (inName) {
                if (!name.empty()) name += ' ';
                name += token;
            } else if (inValue) {
                if (!value.empty()) value += ' ';
                value += token;
            }
        }

        std::string lname = to_lower(name);
        if (lname == "hash") {
            try {
                int mb = std::stoi(value);
                searcher.set_hash(size_t(std::clamp(mb, 1, 4096)));
            } catch (...) {}
        } else if (lname == "clear hash") {
            searcher.clear_hash();
        } else if (lname == "move overhead") {
            try {
                int ms = std::stoi(value);
                searcher.set_overhead(ms);
            } catch (...) {}
        } else if (lname == "evalfile") {
            if (!value.empty() && value != "<empty>") {
                if (NNUE::load_file(value)) {
                    searcher.clear_hash();
                    searcher.clear_state();
                }
            }
        }
#ifdef SPSA_TUNE
        else {
            try {
                int v = std::stoi(value);
                if (lname == "evalmatbase") Tune::EvalMatBase = v;
                else if (lname == "lmrbase") { Tune::LmrBase = v; searcher.init_tables(); }
                else if (lname == "lmrdiv") { Tune::LmrDiv = v; searcher.init_tables(); }
                else if (lname == "lmrhistdiv") Tune::LmrHistDiv = std::max(1, v);
                else if (lname == "rfpimp") Tune::RfpImp = v;
                else if (lname == "rfpnonimp") Tune::RfpNonImp = v;
                else if (lname == "razorbase") Tune::RazorBase = v;
                else if (lname == "razormult") Tune::RazorMult = v;
                else if (lname == "nmpevaldiv") Tune::NmpEvalDiv = std::max(10, v);
                else if (lname == "probcutmargin") Tune::ProbCutMargin = v;
                else if (lname == "futbase") Tune::FutBase = v;
                else if (lname == "futmult") Tune::FutMult = v;
                else if (lname == "histprunemult") Tune::HistPruneMult = v;
                else if (lname == "seequietmult") Tune::SeeQuietMult = v;
                else if (lname == "seecapmult") Tune::SeeCapMult = v;
                else if (lname == "singmarginnum") Tune::SingMarginNum = v;
                else if (lname == "tmdiv") Tune::TmDiv = std::max(5, v);
                else if (lname == "tmsoftpct") Tune::TmSoftPct = v;
                else if (lname == "lmrcutnopv") Tune::LmrCutNoPv = v;
                else if (lname == "lmrttcap") Tune::LmrTtCap = v;
                else if (lname == "lmrcheck") Tune::LmrCheck = v;
                else if (lname == "lmrkiller") Tune::LmrKiller = v;
                else if (lname == "lmrcaphistdiv") Tune::LmrCapHistDiv = std::max(2, v);
                else if (lname == "qsfutmargin") Tune::QsFutMargin = v;
                else if (lname == "corrpawnw") Tune::CorrPawnW = v;
                else if (lname == "corrnonpawnw") Tune::CorrNonPawnW = v;
                else if (lname == "corrminmajw") Tune::CorrMinMajW = v;
                else if (lname == "corrcontw") Tune::CorrContW = v;
                else if (lname == "tmhardmult") Tune::TmHardMult = v;
                else if (lname == "tmnodescale") Tune::TmNodeScale = v;
            } catch (...) {}
        }
#endif
    }

    void send_line(const std::string& s) {
        std::lock_guard<std::mutex> lock(outMtx);
        std::cout << s << '\n' << std::flush;
    }

    void run() {
        history.reserve(8192);
        reset_position(START_FEN);

        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream iss(line);
            std::string cmd;

            // UCI spec: skip unknown leading tokens on the same line
            while (iss >> cmd) {
                if (cmd == "uci") {
                    send_line("id name Gemini 4 Neon chess 24hrs");
                    send_line("id author Gemini 4 Neon");
                    send_line("option name Hash type spin default 64 min 1 max 4096");
                    send_line("option name Clear Hash type button");
                    send_line("option name Move Overhead type spin default 50 min 0 max 500");
                    send_line("option name EvalFile type string default <empty>");
                    send_line("option name Threads type spin default 1 min 1 max 1");
#ifdef SPSA_TUNE
                    send_line("option name EvalMatBase type spin default 24500 min 18000 max 32000");
                    send_line("option name LmrBase type spin default 75 min 40 max 120");
                    send_line("option name LmrDiv type spin default 220 min 165 max 290");
                    send_line("option name LmrHistDiv type spin default 6 min 3 max 14");
                    send_line("option name RfpImp type spin default 58 min 35 max 95");
                    send_line("option name RfpNonImp type spin default 80 min 50 max 130");
                    send_line("option name RazorBase type spin default 250 min 120 max 420");
                    send_line("option name RazorMult type spin default 210 min 110 max 330");
                    send_line("option name NmpEvalDiv type spin default 180 min 100 max 280");
                    send_line("option name ProbCutMargin type spin default 185 min 115 max 285");
                    send_line("option name FutBase type spin default 100 min 45 max 185");
                    send_line("option name FutMult type spin default 90 min 45 max 150");
                    send_line("option name HistPruneMult type spin default 2000 min 900 max 3800");
                    send_line("option name SeeQuietMult type spin default 22 min 10 max 42");
                    send_line("option name SeeCapMult type spin default 85 min 45 max 140");
                    send_line("option name SingMarginNum type spin default 15 min 9 max 28");
                    send_line("option name TmDiv type spin default 22 min 15 max 35");
                    send_line("option name TmSoftPct type spin default 73 min 50 max 95");
                    send_line("option name LmrCutNoPv type spin default 1800 min 800 max 2800");
                    send_line("option name LmrTtCap type spin default 900 min 200 max 1800");
                    send_line("option name LmrCheck type spin default 900 min 200 max 1800");
                    send_line("option name LmrKiller type spin default 900 min 200 max 1800");
                    send_line("option name LmrCapHistDiv type spin default 10 min 4 max 22");
                    send_line("option name QsFutMargin type spin default 140 min 60 max 260");
                    send_line("option name CorrPawnW type spin default 24 min 6 max 50");
                    send_line("option name CorrNonPawnW type spin default 20 min 6 max 44");
                    send_line("option name CorrMinMajW type spin default 16 min 4 max 38");
                    send_line("option name CorrContW type spin default 20 min 4 max 44");
                    send_line("option name TmHardMult type spin default 240 min 150 max 360");
                    send_line("option name TmNodeScale type spin default 135 min 85 max 200");
#endif
                    send_line("uciok");
                    break;
                } else if (cmd == "isready") {
                    send_line("readyok");
                    break;
                } else if (cmd == "ucinewgame") {
                    stop_and_join();
                    searcher.clear_hash();
                    searcher.clear_state();
                    break;
                } else if (cmd == "setoption") {
                    handle_setoption(iss);
                    break;
                } else if (cmd == "position") {
                    handle_position(iss);
                    break;
                } else if (cmd == "go") {
                    handle_go(iss);
                    break;
                } else if (cmd == "stop") {
                    searcher.request_stop();
                    break;
                } else if (cmd == "quit") {
                    stop_and_join();
                    return;
                } else if (cmd == "bench") {
                    stop_and_join();
                    int d = 12;
                    if (iss >> d) {}
                    run_bench(d);
                    break;
                } else if (cmd == "eval") {
                    stop_and_join();
                    int ev = NNUE::is_loaded() ? NNUE::evaluate_scratch(pos) : Eval::evaluate(pos);
                    send_line("eval cp " + std::to_string(ev));
                    break;
                } else if (cmd == "d") {
                    stop_and_join();
                    send_line(pos.to_fen());
                    break;
                }
            }
        }
        stop_and_join();
    }

private:
    Position pos;
    std::vector<StateInfo> history;
    Searcher searcher;
    std::thread worker;
    std::mutex outMtx;
};

} // namespace

void loop() {
    UciEngine engine;
    engine.run();
}

void run_bench(int depth) {
    static const char* benchFens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "r1bqk2r/pp1nbppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 7",
        "2r3k1/1q1nbppp/r3p3/3pP3/pPpP4/P1Q2N2/2RN1PPP/2R4K b - b3 0 23",
        "r2q1rk1/pb1nbppp/1p2pn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
        "5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27",
        "8/4k3/1p2p1p1/pP1pP1P1/P2P1P2/8/4K3/8 w - - 0 1",
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 6"
    };

    Searcher s;
    s.set_hash(64);
    SearchLimits lim;
    lim.depth = depth;

    auto t0 = std::chrono::steady_clock::now();
    U64 totalNodes = 0;

    for (const char* fen : benchFens) {
        Position pos;
        StateInfo st;
        pos.set_fen(fen, &st);
        s.clear_hash();
        s.clear_state();
        SearchResult res = s.search(pos, lim, true);
        totalNodes += res.nodes;
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Bench depth " << depth << ": " << totalNodes << " nodes in "
              << secs << "s (" << U64(totalNodes / std::max(secs, 1e-6)) << " NPS)\n" << std::flush;
}

} // namespace UCI
