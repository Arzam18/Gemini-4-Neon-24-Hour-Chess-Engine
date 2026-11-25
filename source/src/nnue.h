#pragma once

#include "position.h"
#include "nnue_arch.h"
#include <immintrin.h>
#include <array>
#include <string>

namespace NNUE {

struct DirtyPiece {
    int numAdded = 0;
    int numRemoved = 0;
    Piece addedPc[2];
    Square addedSq[2];
    Piece removedPc[2];
    Square removedSq[2];
};

struct alignas(64) Accumulator {
    alignas(64) int16_t acc[COLOR_NB][NNUEArch::HL];
    bool computed[COLOR_NB] = {false, false};
    DirtyPiece dp;
};

struct FinnyEntry {
    alignas(64) int16_t acc[NNUEArch::HL];
    Bitboard byColor[COLOR_NB] = {0, 0};
    Bitboard byType[PIECE_TYPE_NB] = {};
    bool valid = false;
};

struct AccumulatorStack {
    Accumulator stack[MAX_PLY + 8];
    FinnyEntry finny[COLOR_NB][2][NNUEArch::KING_BUCKETS]; // [persp][mirror][bucket]

    void reset_finny();
    void reset_root(const Position& pos);
    void push_move(const Position& posBefore, Move m, int ply);
    void push_null(int ply) {
        Accumulator& child = stack[ply + 1];
        child.computed[WHITE] = false;
        child.computed[BLACK] = false;
        child.dp.numAdded = 0;
        child.dp.numRemoved = 0;
    }
    void ensure_computed(const Position& pos, int ply);
    int evaluate(const Position& pos, int ply);

private:
    void refresh_perspective(const Position& pos, Color persp, int ply);
    void update_perspective(const Position& pos, Color persp, int ply);
};

// Global network pointer (loaded from embedded data or file)
extern const NNUEArch::QuantizedNet* gNet;

bool init_embedded();
bool load_file(const std::string& path);
bool is_loaded();

// Direct uncached evaluation (for testing / root / fallback)
int evaluate_scratch(const Position& pos);

} // namespace NNUE
