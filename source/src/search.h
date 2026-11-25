#pragma once

#include "position.h"
#include "tt.h"
#include "movepicker.h"
#include "nnue.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

struct SearchLimits {
    int64_t time[COLOR_NB] = {0, 0};
    int64_t inc[COLOR_NB]  = {0, 0};
    int movestogo = 0;
    int64_t movetime = 0;
    int depth = 0;
    U64 nodes = 0;
    U64 softNodes = 0;
    bool infinite = false;
    bool useTime = false;
};

struct SearchResult {
    Move bestMove = MOVE_NONE;
    Move ponderMove = MOVE_NONE;
    int score = 0;
    int depth = 0;
    U64 nodes = 0;
};

class Searcher {
public:
    Searcher();

    void init_tables();
    void clear_state();
    void set_hash(size_t mb) { tt.resize(mb); }
    void clear_hash() { tt.clear(); }
    void set_overhead(int ms) { moveOverhead = std::clamp(ms, 0, 500); }
    void request_stop() { stopFlag.store(true, std::memory_order_relaxed); }

    SearchResult search(Position& pos, const SearchLimits& limits, bool silent = false, std::mutex* outMtx = nullptr);

    U64 total_nodes() const { return nodes; }

private:
    int negamax(Position& pos, SearchStack* ss, int alpha, int beta, int depth, int ply, int accPly, bool cutNode);
    int qsearch(Position& pos, SearchStack* ss, int alpha, int beta, int ply, int accPly);
    void update_quiet_stats(const Position& pos, SearchStack* ss, Move bestMove,
                            const Move* quietsTried, int quietCount, int depth);
    void update_cap_stats(const Position& pos, Move bestMove,
                          const Move* capsTried, int capCount, int depth);
    void check_time();
    int64_t elapsed_ms() const;
    void print_info(int depth, int score, const Move* pv, int pvLen, std::mutex* outMtx);
    int evaluate_pos(const Position& pos, int ply);

    TranspositionTable tt;
    std::unique_ptr<SearchTables> tablesPtr;
    SearchTables& tables;
    std::unique_ptr<NNUE::AccumulatorStack> accStackPtr;
    NNUE::AccumulatorStack& accStack;
    bool useNNUE = false;

    std::atomic<bool> stopFlag{false};
    bool aborted = false;
    bool silentMode = false;
    int moveOverhead = 50;

    U64 nodes = 0;
    U64 nodeLimit = 0;
    int selDepth = 0;

    std::chrono::steady_clock::time_point startTime;
    int64_t softLimitMs = 0;
    int64_t hardLimitMs = 0;
    bool hasTimeLimit = false;
    bool isInfinite = false;

    U64 rootMoveNodes[64][64];
    int lmrTable[64][64];
};
