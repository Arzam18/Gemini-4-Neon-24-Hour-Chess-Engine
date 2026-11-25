#pragma once

#include "types.h"
#include <vector>
#include <cstring>

enum Bound : U8 {
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,
    BOUND_LOWER = 2,
    BOUND_EXACT = 3
};

struct TTEntry {
    U32 key32;
    Move move;
    int16_t score;
    int16_t eval;
    U8 depth;
    U8 genBound; // bits 0..1: bound, bit 2: ttPv, bits 3..7: generation

    Bound bound() const { return Bound(genBound & 3); }
    bool is_pv() const { return (genBound & 4) != 0; }
    U8 gen() const { return genBound >> 3; }
};

static_assert(sizeof(TTEntry) == 12, "TTEntry must be 12 bytes");

struct alignas(64) TTBucket {
    TTEntry entries[5];
    U32 padding;
};

static_assert(sizeof(TTBucket) == 64, "TTBucket must be 64 bytes");

class TranspositionTable {
public:
    TranspositionTable() { resize(64); }

    void resize(size_t mb);
    void clear();
    void new_search() { generation = (generation + 1) & 31; }

    inline size_t index(U64 key) const {
        return size_t((__uint128_t(key) * __uint128_t(bucketCount)) >> 64);
    }

    inline void prefetch(U64 key) const {
        __builtin_prefetch(&buckets[index(key)]);
    }

    TTEntry* probe(U64 key, bool& found) const;
    void store(U64 key, Move move, int score, int eval, int depth, Bound bound, bool isPv);
    int hashfull() const;

    static int score_to_tt(int score, int ply) {
        if (score >= SCORE_MATE_IN_MAX)  return score + ply;
        if (score <= SCORE_MATED_IN_MAX) return score - ply;
        return score;
    }

    static int score_from_tt(int score, int ply) {
        if (score >= SCORE_MATE_IN_MAX)  return score - ply;
        if (score <= SCORE_MATED_IN_MAX) return score + ply;
        return score;
    }

private:
    std::vector<TTBucket> buckets;
    size_t bucketCount = 0;
    U8 generation = 0;
};
