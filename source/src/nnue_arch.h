#pragma once

#include <cstdint>

namespace NNUEArch {

// Architecture constants (configurable per generation)
// Gen 1: KING_BUCKETS = 4, HL = 512, OUT_BUCKETS = 8
// Gen 2+: KING_BUCKETS = 8, HL = 768 or 1024, OUT_BUCKETS = 8
#ifndef NNUE_KING_BUCKETS
#define NNUE_KING_BUCKETS 4
#endif
#ifndef NNUE_HL
#define NNUE_HL 512
#endif
#ifndef NNUE_PAIRWISE
#define NNUE_PAIRWISE 0
#endif

constexpr int KING_BUCKETS = NNUE_KING_BUCKETS;
constexpr int HL           = NNUE_HL;
constexpr bool PAIRWISE    = (NNUE_PAIRWISE != 0);
constexpr int L1_PER_PERSP = PAIRWISE ? (HL / 2) : HL;
constexpr int OUT_BUCKETS  = 8;
constexpr int INPUT_SIZE   = KING_BUCKETS * 768;

constexpr int QA    = 255;
constexpr int QB    = 64;
constexpr int SCALE = 400;

// Maps relative king square (0..63, AFTER horizontal mirroring to files a..d) -> bucket 0..KING_BUCKETS-1
inline int king_bucket(int mirroredRelKingSq) {
    int f = mirroredRelKingSq & 7; // 0..3
    int r = mirroredRelKingSq >> 3; // 0..7
    if constexpr (KING_BUCKETS == 1) {
        return 0;
    } else if constexpr (KING_BUCKETS == 4) {
        if (r <= 1) return (f <= 1) ? 0 : 1;
        if (r <= 3) return 2;
        return 3;
    } else if constexpr (KING_BUCKETS == 8) {
        // Back-rank-focused 8-bucket layout (mirrored files a..d)
        if (r == 0) return f;                // a1/h1=0, b1/g1=1, c1/f1=2, d1/e1=3
        if (r == 1) return (f <= 1) ? 4 : 5; // castled luft=4, center rank2=5
        if (r <= 3) return 6;                // ranks 3-4=6
        return 7;                            // ranks 5-8=7
    } else {
        // Fallback: 10 buckets (Stockfish-like back-rank granularity)
        static constexpr int Map10[32] = {
            0, 1, 2, 3,
            4, 4, 5, 5,
            6, 6, 7, 7,
            6, 6, 7, 7,
            8, 8, 9, 9,
            8, 8, 9, 9,
            8, 8, 9, 9,
            8, 8, 9, 9
        };
        return Map10[r * 4 + f];
    }
}

// Compute feature index for perspective `persp` (0=WHITE, 1=BLACK)
// `perspKingSq` is actual square 0..63 of persp's king.
// `piece` is 0..11 (W_PAWN..B_KING), `sq` is actual square 0..63.
inline int feature_index(int persp, int perspKingSq, int piece, int sq) {
    int relKing = perspKingSq ^ (persp * 56);
    int mir = ((relKing & 7) >= 4) ? 7 : 0;
    int bucket = king_bucket(relKing ^ mir);

    int pcColor = (piece >= 6) ? 1 : 0;
    int pt = (piece >= 6) ? (piece - 6) : piece;
    int relColor = (pcColor == persp) ? 0 : 1;
    int relSq = (sq ^ (persp * 56)) ^ mir;
    return bucket * 768 + relColor * 384 + pt * 64 + relSq;
}

inline int output_bucket(int pieceCount) {
    int b = (pieceCount - 2) / 4;
    if (b < 0) b = 0;
    if (b >= OUT_BUCKETS) b = OUT_BUCKETS - 1;
    return b;
}

// Quantized network binary layout (all arrays 64-byte cache-line aligned)
struct alignas(64) QuantizedNet {
    uint32_t magic;       // 0x4E4E5545 ('NNUE') or 0x4E4E5057 ('NNPW' for pairwise)
    uint32_t kingBuckets;
    uint32_t hl;
    uint32_t outBuckets;
    uint32_t pad[12];     // Pad header to 64 bytes
    alignas(64) int16_t  ftWeights[INPUT_SIZE * HL];
    alignas(64) int16_t  ftBias[HL];
    alignas(64) int16_t  l1Weights[OUT_BUCKETS][2 * L1_PER_PERSP];
    alignas(64) int32_t  l1Bias[OUT_BUCKETS]; // scaled by QA * QB
};

constexpr uint32_t EXPECTED_MAGIC = PAIRWISE ? 0x4E4E5057U : 0x4E4E5545U;

} // namespace NNUEArch
