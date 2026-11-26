#include "nnue.h"
#include "tune.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <memory>

#if defined(__aarch64__)
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

namespace NNUE {

const NNUEArch::QuantizedNet* gNet = nullptr;
static std::unique_ptr<NNUEArch::QuantizedNet> gOwnedNet;

#ifdef USE_EMBEDDED_NNUE
extern "C" {
    extern const unsigned char gEmbeddedNNUEData[];
    extern const unsigned char gEmbeddedNNUEEnd[];
}
#endif

bool init_embedded() {
#ifdef USE_EMBEDDED_NNUE
    size_t sz = size_t(gEmbeddedNNUEEnd - gEmbeddedNNUEData);
    if (sz >= sizeof(NNUEArch::QuantizedNet)) {
        const auto* net = reinterpret_cast<const NNUEArch::QuantizedNet*>(gEmbeddedNNUEData);
        if (net->magic == NNUEArch::EXPECTED_MAGIC && net->kingBuckets == NNUEArch::KING_BUCKETS
            && net->hl == NNUEArch::HL && net->outBuckets == NNUEArch::OUT_BUCKETS) {
            gNet = net;
            return true;
        }
    }
#endif
    return false;
}

bool load_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    auto net = std::make_unique<NNUEArch::QuantizedNet>();
    in.read(reinterpret_cast<char*>(net.get()), sizeof(NNUEArch::QuantizedNet));
    if (!in || net->magic != NNUEArch::EXPECTED_MAGIC || net->kingBuckets != NNUEArch::KING_BUCKETS
        || net->hl != NNUEArch::HL || net->outBuckets != NNUEArch::OUT_BUCKETS) {
        return false;
    }
    gOwnedNet = std::move(net);
    gNet = gOwnedNet.get();
    return true;
}

bool is_loaded() {
    return gNet != nullptr;
}

namespace {

#if defined(__aarch64__)

inline void acc_add1(int16_t* dst, const int16_t* src, const int16_t* wAdd) {
    constexpr int N = NNUEArch::HL;
    for (int i = 0; i < N; i += 8)
        vst1q_s16(dst + i, vaddq_s16(vld1q_s16(src + i), vld1q_s16(wAdd + i)));
}

inline void acc_sub1(int16_t* dst, const int16_t* src, const int16_t* wSub) {
    constexpr int N = NNUEArch::HL;
    for (int i = 0; i < N; i += 8)
        vst1q_s16(dst + i, vsubq_s16(vld1q_s16(src + i), vld1q_s16(wSub + i)));
}

inline void acc_add1_sub1(int16_t* __restrict dst, const int16_t* __restrict src,
                          const int16_t* __restrict wAdd, const int16_t* __restrict wSub) {
    constexpr int N = NNUEArch::HL;
    for (int i = 0; i < N; i += 8) {
        int16x8_t a = vld1q_s16(src + i);
        a = vaddq_s16(a, vld1q_s16(wAdd + i));
        a = vsubq_s16(a, vld1q_s16(wSub + i));
        vst1q_s16(dst + i, a);
    }
}

inline void acc_add1_sub2(int16_t* __restrict dst, const int16_t* __restrict src,
                          const int16_t* __restrict wAdd,
                          const int16_t* __restrict wSub1, const int16_t* __restrict wSub2) {
    constexpr int N = NNUEArch::HL;
    for (int i = 0; i < N; i += 8) {
        int16x8_t a = vld1q_s16(src + i);
        a = vaddq_s16(a, vld1q_s16(wAdd + i));
        a = vsubq_s16(a, vld1q_s16(wSub1 + i));
        a = vsubq_s16(a, vld1q_s16(wSub2 + i));
        vst1q_s16(dst + i, a);
    }
}

inline void acc_add2_sub2(int16_t* __restrict dst, const int16_t* __restrict src,
                          const int16_t* __restrict wAdd1, const int16_t* __restrict wAdd2,
                          const int16_t* __restrict wSub1, const int16_t* __restrict wSub2) {
    constexpr int N = NNUEArch::HL;
    for (int i = 0; i < N; i += 8) {
        int16x8_t a = vld1q_s16(src + i);
        a = vaddq_s16(a, vld1q_s16(wAdd1 + i));
        a = vaddq_s16(a, vld1q_s16(wAdd2 + i));
        a = vsubq_s16(a, vld1q_s16(wSub1 + i));
        a = vsubq_s16(a, vld1q_s16(wSub2 + i));
        vst1q_s16(dst + i, a);
    }
}

// Add products of two 8-lane vectors to two 4-lane int32 accumulators.
// `lhs * weights` intentionally wraps to int16, matching AVX2 mullo.
inline void madd16x8(int32x4_t& sumLo, int32x4_t& sumHi,
                     int16x8_t lhs, int16x8_t weights, int16x8_t rhs) {
    const int16x8_t product = vmulq_s16(lhs, weights);
    sumLo = vmlal_s16(sumLo, vget_low_s16(product), vget_low_s16(rhs));
    sumHi = vmlal_s16(sumHi, vget_high_s16(product), vget_high_s16(rhs));
}

inline int16x8_t clipped_acc(const int16_t* p) {
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t qa = vdupq_n_s16(NNUEArch::QA);
    return vminq_s16(qa, vmaxq_s16(zero, vld1q_s16(p)));
}

inline int forward_l1(const int16_t* usAcc, const int16_t* themAcc, int outBucket) {
    constexpr int N = NNUEArch::HL;
    constexpr int HALF = N / 2;
    const int16_t* wUs = gNet->l1Weights[outBucket];
    const int16_t* wThem = wUs + NNUEArch::L1_PER_PERSP;

    int32x4_t us0Lo = vdupq_n_s32(0), us0Hi = vdupq_n_s32(0);
    int32x4_t us1Lo = vdupq_n_s32(0), us1Hi = vdupq_n_s32(0);
    int32x4_t th0Lo = vdupq_n_s32(0), th0Hi = vdupq_n_s32(0);
    int32x4_t th1Lo = vdupq_n_s32(0), th1Hi = vdupq_n_s32(0);

    if constexpr (NNUEArch::PAIRWISE) {
        for (int i = 0; i < HALF; i += 32) {
            for (int j = 0; j < 16; j += 8) {
                int16x8_t ua = clipped_acc(usAcc + i + j);
                int16x8_t ub = clipped_acc(usAcc + HALF + i + j);
                madd16x8(us0Lo, us0Hi, ua, vld1q_s16(wUs + i + j), ub);
                int16x8_t ta = clipped_acc(themAcc + i + j);
                int16x8_t tb = clipped_acc(themAcc + HALF + i + j);
                madd16x8(th0Lo, th0Hi, ta, vld1q_s16(wThem + i + j), tb);
            }
            for (int j = 16; j < 32; j += 8) {
                int16x8_t ua = clipped_acc(usAcc + i + j);
                int16x8_t ub = clipped_acc(usAcc + HALF + i + j);
                madd16x8(us1Lo, us1Hi, ua, vld1q_s16(wUs + i + j), ub);
                int16x8_t ta = clipped_acc(themAcc + i + j);
                int16x8_t tb = clipped_acc(themAcc + HALF + i + j);
                madd16x8(th1Lo, th1Hi, ta, vld1q_s16(wThem + i + j), tb);
            }
        }
    } else {
        for (int i = 0; i < N; i += 32) {
            for (int j = 0; j < 16; j += 8) {
                int16x8_t u = clipped_acc(usAcc + i + j);
                madd16x8(us0Lo, us0Hi, u, vld1q_s16(wUs + i + j), u);
                int16x8_t t = clipped_acc(themAcc + i + j);
                madd16x8(th0Lo, th0Hi, t, vld1q_s16(wThem + i + j), t);
            }
            for (int j = 16; j < 32; j += 8) {
                int16x8_t u = clipped_acc(usAcc + i + j);
                madd16x8(us1Lo, us1Hi, u, vld1q_s16(wUs + i + j), u);
                int16x8_t t = clipped_acc(themAcc + i + j);
                madd16x8(th1Lo, th1Hi, t, vld1q_s16(wThem + i + j), t);
            }
        }
    }

    const int64_t sum64 =
        int64_t(vaddlvq_s32(us0Lo)) + vaddlvq_s32(us0Hi) +
        vaddlvq_s32(us1Lo) + vaddlvq_s32(us1Hi) +
        vaddlvq_s32(th0Lo) + vaddlvq_s32(th0Hi) +
        vaddlvq_s32(th1Lo) + vaddlvq_s32(th1Hi);
    const int32_t sum = static_cast<int32_t>(sum64);
    const int32_t output = (sum / NNUEArch::QA) + gNet->l1Bias[outBucket];
    return int((int64_t(output) * NNUEArch::SCALE) / (NNUEArch::QA * NNUEArch::QB));
}

#else

inline void acc_add1(int16_t* dst, const int16_t* src, const int16_t* wAdd) {
    constexpr int N = NNUEArch::HL;
    for (int i = 0; i < N; i += 16) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wAdd + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_add_epi16(a, w));
    }
}

inline void acc_sub1(int16_t* dst, const int16_t* src, const int16_t* wSub) {
    constexpr int N = NNUEArch::HL;
    for (int i = 0; i < N; i += 16) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wSub + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_sub_epi16(a, w));
    }
}

inline void acc_add1_sub1(int16_t* __restrict dst, const int16_t* __restrict src,
                          const int16_t* __restrict wAdd, const int16_t* __restrict wSub) {
    constexpr int N = NNUEArch::HL;
    for (int i = 0; i < N; i += 16) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        a = _mm256_add_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wAdd + i)));
        a = _mm256_sub_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wSub + i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), a);
    }
}

inline void acc_add1_sub2(int16_t* __restrict dst, const int16_t* __restrict src,
                          const int16_t* __restrict wAdd,
                          const int16_t* __restrict wSub1, const int16_t* __restrict wSub2) {
    constexpr int N = NNUEArch::HL;
    for (int i = 0; i < N; i += 16) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        a = _mm256_add_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wAdd + i)));
        a = _mm256_sub_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wSub1 + i)));
        a = _mm256_sub_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wSub2 + i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), a);
    }
}

inline void acc_add2_sub2(int16_t* __restrict dst, const int16_t* __restrict src,
                          const int16_t* __restrict wAdd1, const int16_t* __restrict wAdd2,
                          const int16_t* __restrict wSub1, const int16_t* __restrict wSub2) {
    constexpr int N = NNUEArch::HL;
    for (int i = 0; i < N; i += 16) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        a = _mm256_add_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wAdd1 + i)));
        a = _mm256_add_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wAdd2 + i)));
        a = _mm256_sub_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wSub1 + i)));
        a = _mm256_sub_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wSub2 + i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), a);
    }
}

inline int forward_l1(const int16_t* usAcc, const int16_t* themAcc, int outBucket) {
    constexpr int N = NNUEArch::HL;
    constexpr int HALF = N / 2;
    const __m256i vZero = _mm256_setzero_si256();
    const __m256i vQA   = _mm256_set1_epi16(NNUEArch::QA);
    const int16_t* wUs   = gNet->l1Weights[outBucket];
    const int16_t* wThem = wUs + NNUEArch::L1_PER_PERSP;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();

    if constexpr (NNUEArch::PAIRWISE) {
        for (int i = 0; i < HALF; i += 32) {
            __m256i ua0 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(usAcc + i))));
            __m256i ub0 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(usAcc + HALF + i))));
            __m256i wu0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wUs + i));
            sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(_mm256_mullo_epi16(ua0, wu0), ub0));

            __m256i ua1 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(usAcc + i + 16))));
            __m256i ub1 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(usAcc + HALF + i + 16))));
            __m256i wu1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wUs + i + 16));
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(_mm256_mullo_epi16(ua1, wu1), ub1));

            __m256i ta0 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(themAcc + i))));
            __m256i tb0 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(themAcc + HALF + i))));
            __m256i wt0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wThem + i));
            sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(_mm256_mullo_epi16(ta0, wt0), tb0));

            __m256i ta1 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(themAcc + i + 16))));
            __m256i tb1 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(themAcc + HALF + i + 16))));
            __m256i wt1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wThem + i + 16));
            sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(_mm256_mullo_epi16(ta1, wt1), tb1));
        }
    } else {
        for (int i = 0; i < N; i += 32) {
            __m256i u0 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(usAcc + i))));
            __m256i wu0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wUs + i));
            sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(_mm256_mullo_epi16(u0, wu0), u0));

            __m256i u1 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(usAcc + i + 16))));
            __m256i wu1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wUs + i + 16));
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(_mm256_mullo_epi16(u1, wu1), u1));

            __m256i t0 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(themAcc + i))));
            __m256i wt0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wThem + i));
            sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(_mm256_mullo_epi16(t0, wt0), t0));

            __m256i t1 = _mm256_min_epi16(vQA, _mm256_max_epi16(vZero, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(themAcc + i + 16))));
            __m256i wt1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wThem + i + 16));
            sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(_mm256_mullo_epi16(t1, wt1), t1));
        }
    }

    __m256i totalVec = _mm256_add_epi32(_mm256_add_epi32(sum0, sum1), _mm256_add_epi32(sum2, sum3));
    __m128i lo = _mm256_castsi256_si128(totalVec);
    __m128i hi = _mm256_extracti128_si256(totalVec, 1);
    __m128i s4 = _mm_add_epi32(lo, hi);
    __m128i s2 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t sum = _mm_cvtsi128_si32(s1);

    int32_t output = (sum / NNUEArch::QA) + gNet->l1Bias[outBucket];
    return int((int64_t(output) * NNUEArch::SCALE) / (NNUEArch::QA * NNUEArch::QB));
}

#endif // AArch64 NEON / x86 AVX2

inline bool needs_refresh(Color persp, Square from, Square to) {
    int relFrom = int(from) ^ (int(persp) * 56);
    int relTo   = int(to)   ^ (int(persp) * 56);
    int mirFrom = ((relFrom & 7) >= 4) ? 7 : 0;
    int mirTo   = ((relTo   & 7) >= 4) ? 7 : 0;
    if (mirFrom != mirTo) return true;
    return NNUEArch::king_bucket(relFrom ^ mirFrom) != NNUEArch::king_bucket(relTo ^ mirTo);
}

} // namespace

void AccumulatorStack::reset_finny() {
    for (int c = 0; c < 2; ++c)
        for (int m = 0; m < 2; ++m)
            for (int b = 0; b < NNUEArch::KING_BUCKETS; ++b)
                finny[c][m][b].valid = false;
}

void AccumulatorStack::refresh_perspective(const Position& pos, Color persp, int ply) {
    Square ksq = pos.king_square(persp);
    int relK = int(ksq) ^ (int(persp) * 56);
    int mirIdx = ((relK & 7) >= 4) ? 1 : 0;
    int mir = mirIdx ? 7 : 0;
    int bucket = NNUEArch::king_bucket(relK ^ mir);

    FinnyEntry& fe = finny[persp][mirIdx][bucket];
    if (!fe.valid) {
        std::memcpy(fe.acc, gNet->ftBias, NNUEArch::HL * sizeof(int16_t));
        std::memset(fe.byColor, 0, sizeof(fe.byColor));
        std::memset(fe.byType, 0, sizeof(fe.byType));
        fe.valid = true;
    }

    // Diff current board against FinnyEntry (batched 4-at-a-time for 4x less acc memory traffic)
    int addFeats[32], remFeats[32];
    int nAdd = 0, nRem = 0;
    const int baseOff = bucket * 768;
    const int perspFlip = int(persp) * 56;

    for (int c = 0; c < 2; ++c) {
        int relColorOff = baseOff + ((c == int(persp)) ? 0 : 384);
        for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
            Bitboard cur = pos.pieces(Color(c), PieceType(pt));
            Bitboard old = fe.byColor[c] & fe.byType[pt];
            Bitboard added   = cur & ~old;
            Bitboard removed = old & ~cur;
            int ptOff = relColorOff + pt * 64;

            while (added) {
                Square sq = pop_lsb(added);
                addFeats[nAdd++] = ptOff + ((int(sq) ^ perspFlip) ^ mir);
            }
            while (removed) {
                Square sq = pop_lsb(removed);
                remFeats[nRem++] = ptOff + ((int(sq) ^ perspFlip) ^ mir);
            }
        }
    }

    int ia = 0, ir = 0;
    while (ia + 2 <= nAdd && ir + 2 <= nRem) {
        acc_add2_sub2(fe.acc, fe.acc,
                      &gNet->ftWeights[size_t(addFeats[ia]) * NNUEArch::HL],
                      &gNet->ftWeights[size_t(addFeats[ia + 1]) * NNUEArch::HL],
                      &gNet->ftWeights[size_t(remFeats[ir]) * NNUEArch::HL],
                      &gNet->ftWeights[size_t(remFeats[ir + 1]) * NNUEArch::HL]);
        ia += 2; ir += 2;
    }
    while (ia < nAdd && ir < nRem) {
        acc_add1_sub1(fe.acc, fe.acc,
                      &gNet->ftWeights[size_t(addFeats[ia++]) * NNUEArch::HL],
                      &gNet->ftWeights[size_t(remFeats[ir++]) * NNUEArch::HL]);
    }
    while (ia < nAdd) {
        acc_add1(fe.acc, fe.acc, &gNet->ftWeights[size_t(addFeats[ia++]) * NNUEArch::HL]);
    }
    while (ir < nRem) {
        acc_sub1(fe.acc, fe.acc, &gNet->ftWeights[size_t(remFeats[ir++]) * NNUEArch::HL]);
    }

    fe.byColor[WHITE] = pos.pieces(WHITE);
    fe.byColor[BLACK] = pos.pieces(BLACK);
    for (int pt = 0; pt < PIECE_TYPE_NB; ++pt)
        fe.byType[pt] = pos.pieces(PieceType(pt));

    std::memcpy(stack[ply].acc[persp], fe.acc, NNUEArch::HL * sizeof(int16_t));
    stack[ply].computed[persp] = true;
}

void AccumulatorStack::reset_root(const Position& pos) {
    if (!gNet) return;
    refresh_perspective(pos, WHITE, 0);
    refresh_perspective(pos, BLACK, 0);
}

void AccumulatorStack::push_move(const Position& posBefore, Move m, int ply) {
    // Store dirty piece changes into stack[ply + 1]
    Accumulator& child = stack[ply + 1];
    child.computed[WHITE] = false;
    child.computed[BLACK] = false;

    DirtyPiece& dp = child.dp;
    dp.numAdded = 0;
    dp.numRemoved = 0;

    Color us = posBefore.side_to_move();
    Color them = ~us;
    Square from = from_sq(m);
    Square to = to_sq(m);
    MoveFlag flag = move_flag(m);
    Piece pc = posBefore.piece_on(from);

    if (flag == MF_CASTLE_KS) {
        Square rFrom = (us == WHITE ? SQ_H1 : SQ_H8);
        Square rTo   = (us == WHITE ? SQ_F1 : SQ_F8);
        Piece rook = make_piece(us, ROOK);
        dp.removedPc[0] = pc;   dp.removedSq[0] = from;
        dp.removedPc[1] = rook; dp.removedSq[1] = rFrom;
        dp.numRemoved = 2;
        dp.addedPc[0] = pc;     dp.addedSq[0] = to;
        dp.addedPc[1] = rook;   dp.addedSq[1] = rTo;
        dp.numAdded = 2;
    } else if (flag == MF_CASTLE_QS) {
        Square rFrom = (us == WHITE ? SQ_A1 : SQ_A8);
        Square rTo   = (us == WHITE ? SQ_D1 : SQ_D8);
        Piece rook = make_piece(us, ROOK);
        dp.removedPc[0] = pc;   dp.removedSq[0] = from;
        dp.removedPc[1] = rook; dp.removedSq[1] = rFrom;
        dp.numRemoved = 2;
        dp.addedPc[0] = pc;     dp.addedSq[0] = to;
        dp.addedPc[1] = rook;   dp.addedSq[1] = rTo;
        dp.numAdded = 2;
    } else {
        Piece finalPc = is_promotion(m) ? make_piece(us, promo_type(m)) : pc;
        dp.removedPc[0] = pc; dp.removedSq[0] = from;
        dp.numRemoved = 1;
        if (flag == MF_EP_CAPTURE) {
            Square capSq = Square(to + (us == WHITE ? -8 : 8));
            dp.removedPc[1] = make_piece(them, PAWN);
            dp.removedSq[1] = capSq;
            dp.numRemoved = 2;
        } else if (is_capture(m)) {
            dp.removedPc[1] = posBefore.piece_on(to);
            dp.removedSq[1] = to;
            dp.numRemoved = 2;
        }
        dp.addedPc[0] = finalPc; dp.addedSq[0] = to;
        dp.numAdded = 1;
    }
}

void AccumulatorStack::update_perspective(const Position& pos, Color persp, int ply) {
    if (stack[ply].computed[persp]) return;

    // Walk back to nearest computed ancestor, checking if king crossed bucket/mirror
    int p = ply;
    while (p > 0 && !stack[p].computed[persp]) {
        const DirtyPiece& dp = stack[p].dp;
        if (dp.numRemoved > 0 && dp.removedPc[0] == make_piece(persp, KING)) {
            if (needs_refresh(persp, dp.removedSq[0], dp.addedSq[0])) {
                refresh_perspective(pos, persp, ply);
                return;
            }
        }
        p--;
    }

    if (!stack[p].computed[persp]) {
        refresh_perspective(pos, persp, ply);
        return;
    }

    // Apply incremental deltas forward from p to ply
    Square ksq = pos.king_square(persp);
    for (int cur = p + 1; cur <= ply; ++cur) {
        const DirtyPiece& dp = stack[cur].dp;
        int16_t* dst = stack[cur].acc[persp];
        const int16_t* src = stack[cur - 1].acc[persp];

        if (dp.numAdded == 1 && dp.numRemoved == 1) {
            int fA = NNUEArch::feature_index(persp, ksq, dp.addedPc[0], dp.addedSq[0]);
            int fS = NNUEArch::feature_index(persp, ksq, dp.removedPc[0], dp.removedSq[0]);
            acc_add1_sub1(dst, src,
                          &gNet->ftWeights[size_t(fA) * NNUEArch::HL],
                          &gNet->ftWeights[size_t(fS) * NNUEArch::HL]);
        } else if (dp.numAdded == 1 && dp.numRemoved == 2) {
            int fA  = NNUEArch::feature_index(persp, ksq, dp.addedPc[0], dp.addedSq[0]);
            int fS1 = NNUEArch::feature_index(persp, ksq, dp.removedPc[0], dp.removedSq[0]);
            int fS2 = NNUEArch::feature_index(persp, ksq, dp.removedPc[1], dp.removedSq[1]);
            acc_add1_sub2(dst, src,
                          &gNet->ftWeights[size_t(fA) * NNUEArch::HL],
                          &gNet->ftWeights[size_t(fS1) * NNUEArch::HL],
                          &gNet->ftWeights[size_t(fS2) * NNUEArch::HL]);
        } else if (dp.numAdded == 2 && dp.numRemoved == 2) {
            int fA1 = NNUEArch::feature_index(persp, ksq, dp.addedPc[0], dp.addedSq[0]);
            int fA2 = NNUEArch::feature_index(persp, ksq, dp.addedPc[1], dp.addedSq[1]);
            int fS1 = NNUEArch::feature_index(persp, ksq, dp.removedPc[0], dp.removedSq[0]);
            int fS2 = NNUEArch::feature_index(persp, ksq, dp.removedPc[1], dp.removedSq[1]);
            acc_add2_sub2(dst, src,
                          &gNet->ftWeights[size_t(fA1) * NNUEArch::HL],
                          &gNet->ftWeights[size_t(fA2) * NNUEArch::HL],
                          &gNet->ftWeights[size_t(fS1) * NNUEArch::HL],
                          &gNet->ftWeights[size_t(fS2) * NNUEArch::HL]);
        } else {
            // Null move (0 added, 0 removed): copy pointer/contents
            std::memcpy(dst, src, NNUEArch::HL * sizeof(int16_t));
        }
        stack[cur].computed[persp] = true;
    }
}

void AccumulatorStack::ensure_computed(const Position& pos, int ply) {
    update_perspective(pos, WHITE, ply);
    update_perspective(pos, BLACK, ply);
}

int AccumulatorStack::evaluate(const Position& pos, int ply) {
    ensure_computed(pos, ply);
    Color stm = pos.side_to_move();
    int outBucket = NNUEArch::output_bucket(popcount(pos.pieces()));
    int raw = forward_l1(stack[ply].acc[stm], stack[ply].acc[~stm], outBucket);

    // Material-based phase scaling (gently dampens noisy low-material endgames)
    int mat = popcount(pos.pieces(PAWN)) * 100
            + popcount(pos.pieces(KNIGHT)) * 320
            + popcount(pos.pieces(BISHOP)) * 330
            + popcount(pos.pieces(ROOK)) * 500
            + popcount(pos.pieces(QUEEN)) * 950;
    raw = raw * (Tune::EvalMatBase + mat) / 32768;

    // 50-move rule damping
    raw = raw * (200 - pos.rule50_count()) / 200;
    return std::clamp(raw, SCORE_MATED_IN_MAX + 100, SCORE_MATE_IN_MAX - 100);
}

int evaluate_scratch(const Position& pos) {
    if (!gNet) return 0;
    alignas(64) int16_t accW[NNUEArch::HL];
    alignas(64) int16_t accB[NNUEArch::HL];
    std::memcpy(accW, gNet->ftBias, NNUEArch::HL * sizeof(int16_t));
    std::memcpy(accB, gNet->ftBias, NNUEArch::HL * sizeof(int16_t));

    Square wk = pos.king_square(WHITE);
    Square bk = pos.king_square(BLACK);
    Bitboard occ = pos.pieces();
    int pcCount = 0;
    while (occ) {
        Square sq = pop_lsb(occ);
        Piece pc = pos.piece_on(sq);
        int fw = NNUEArch::feature_index(WHITE, wk, pc, sq);
        int fb = NNUEArch::feature_index(BLACK, bk, pc, sq);
        acc_add1(accW, accW, &gNet->ftWeights[size_t(fw) * NNUEArch::HL]);
        acc_add1(accB, accB, &gNet->ftWeights[size_t(fb) * NNUEArch::HL]);
        pcCount++;
    }

    Color stm = pos.side_to_move();
    int outBucket = NNUEArch::output_bucket(pcCount);
    int raw = forward_l1(stm == WHITE ? accW : accB, stm == WHITE ? accB : accW, outBucket);
    int mat = popcount(pos.pieces(PAWN)) * 100
            + popcount(pos.pieces(KNIGHT)) * 320
            + popcount(pos.pieces(BISHOP)) * 330
            + popcount(pos.pieces(ROOK)) * 500
            + popcount(pos.pieces(QUEEN)) * 950;
    raw = raw * (Tune::EvalMatBase + mat) / 32768;
    raw = raw * (200 - pos.rule50_count()) / 200;
    return std::clamp(raw, SCORE_MATED_IN_MAX + 100, SCORE_MATE_IN_MAX - 100);
}

} // namespace NNUE
