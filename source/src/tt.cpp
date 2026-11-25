#include "tt.h"
#include <algorithm>

void TranspositionTable::resize(size_t mb) {
    mb = std::clamp<size_t>(mb, 1, 4096);
    bucketCount = (mb * 1024ULL * 1024ULL) / sizeof(TTBucket);
    if (bucketCount < 1024) bucketCount = 1024;
    buckets.assign(bucketCount, TTBucket{});
    generation = 0;
}

void TranspositionTable::clear() {
    std::memset(buckets.data(), 0, buckets.size() * sizeof(TTBucket));
    generation = 0;
}

TTEntry* TranspositionTable::probe(U64 key, bool& found) const {
    TTBucket& bucket = const_cast<TTBucket&>(buckets[index(key)]);
    U32 key32 = U32(key); // Lower 32 bits (independent of upper bits used in index())

    for (int i = 0; i < 5; ++i) {
        if (bucket.entries[i].key32 == key32 && bucket.entries[i].bound() != BOUND_NONE) {
            found = true;
            return &bucket.entries[i];
        }
    }
    found = false;
    return nullptr;
}

void TranspositionTable::store(U64 key, Move move, int score, int eval, int depth, Bound bound, bool isPv) {
    TTBucket& bucket = buckets[index(key)];
    U32 key32 = U32(key);
    U8 depth8 = U8(std::clamp(depth, 0, 255));

    TTEntry* matched = nullptr;
    TTEntry* empty = nullptr;
    TTEntry* worst = &bucket.entries[0];
    int worstVal = 1000000;

    for (int i = 0; i < 5; ++i) {
        TTEntry& e = bucket.entries[i];
        if (e.key32 == key32 && e.bound() != BOUND_NONE) {
            matched = &e;
            break;
        }
        if (e.bound() == BOUND_NONE) {
            if (!empty) empty = &e;
            continue;
        }
        int ageDiff = (32 + generation - e.gen()) & 31;
        int val = int(e.depth) - 8 * ageDiff;
        if (val < worstVal) {
            worstVal = val;
            worst = &e;
        }
    }
    TTEntry* replace = matched ? matched : (empty ? empty : worst);

    bool sameKey = (replace->key32 == key32 && replace->bound() != BOUND_NONE);
    if (sameKey) {
        if (move == MOVE_NONE) move = replace->move;
        if (eval == SCORE_NONE) eval = replace->eval;
        if (bound != BOUND_EXACT
            && replace->gen() == generation
            && int(depth8) + 4 + 2 * int(isPv) < int(replace->depth)) {
            return;
        }
    }

    replace->key32 = key32;
    replace->move = move;
    replace->score = int16_t(score);
    replace->eval = int16_t(eval);
    replace->depth = depth8;
    replace->genBound = U8((generation << 3) | (isPv ? 4 : 0) | bound);
}

int TranspositionTable::hashfull() const {
    int used = 0;
    size_t sample = std::min<size_t>(1000, bucketCount);
    for (size_t i = 0; i < sample; ++i) {
        for (int j = 0; j < 5; ++j) {
            const auto& e = buckets[i].entries[j];
            if (e.bound() != BOUND_NONE && e.gen() == generation)
                used++;
        }
    }
    return int((used * 1000ULL) / (sample * 5));
}
