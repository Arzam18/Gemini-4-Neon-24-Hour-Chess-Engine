// High-speed C++20 AVX2 Multi-Threaded Sparse NNUE Trainer (AdamW + SCReLU + Cosine LR)
// Compiles cleanly with GCC 15:
// g++ -std=c++20 -O3 -march=x86-64-v3 -pthread -Isource/src source/trainer/trainer.cpp -o source/build/trainer.exe

#include "dataformat.h"
#include "nnue_arch.h"

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <random>
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <memory>

using namespace NNUEArch;

namespace {

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

struct FeatureList {
    int wFeats[32];
    int bFeats[32];
    int count;
    int stm;
    int outBucket;
    float target;
};

void unpack_pos(const PackedPos& p, FeatureList& fl, float wdlLambda, float evalScale) {
    uint64_t occ = p.occ;
    int pieces[32];
    int squares[32];
    int n = 0;
    int wKingSq = 4, bKingSq = 60;

    while (occ && n < 32) {
        int sq = __builtin_ctzll(occ);
        occ &= occ - 1;
        int pc = (n & 1) ? (p.pieces[n >> 1] >> 4) : (p.pieces[n >> 1] & 0xF);
        pieces[n] = pc;
        squares[n] = sq;
        if (pc == 5) wKingSq = sq;
        else if (pc == 11) bKingSq = sq;
        n++;
    }

    fl.count = n;
    fl.stm = p.stm;
    fl.outBucket = output_bucket(n);

    for (int i = 0; i < n; ++i) {
        fl.wFeats[i] = feature_index(0, wKingSq, pieces[i], squares[i]);
        fl.bFeats[i] = feature_index(1, bKingSq, pieces[i], squares[i]);
    }

    // Perspective target: from side-to-move's POV!
    float wdlWhite = p.wdl * 0.5f; // 0.0, 0.5, 1.0
    float scoreWhite = float(p.score);
    float stmScore = (p.stm == 0) ? scoreWhite : -scoreWhite;
    float stmWdl   = (p.stm == 0) ? wdlWhite   : (1.0f - wdlWhite);

    float evalProb = sigmoid(stmScore / evalScale);
    fl.target = (1.0f - wdlLambda) * evalProb + wdlLambda * stmWdl;
}

// Thread-local gradient buffer
struct GradBuffer {
    std::vector<float> ftW; // INPUT_SIZE * HL
    std::vector<float> ftB; // HL
    std::vector<float> l1W; // OUT_BUCKETS * 2 * HL
    std::vector<float> l1B; // OUT_BUCKETS
    std::vector<int>   dirtyFeats;
    std::vector<uint8_t> featTouched;
    double lossSum = 0.0;

    GradBuffer()
        : ftW(size_t(INPUT_SIZE) * HL, 0.0f),
          ftB(HL, 0.0f),
          l1W(OUT_BUCKETS * 2 * L1_PER_PERSP, 0.0f),
          l1B(OUT_BUCKETS, 0.0f),
          featTouched(INPUT_SIZE, 0) {
        dirtyFeats.reserve(INPUT_SIZE);
    }

    void clear_fast() {
        for (int f : dirtyFeats) {
            std::memset(&ftW[size_t(f) * HL], 0, HL * sizeof(float));
            featTouched[f] = 0;
        }
        dirtyFeats.clear();
        std::fill(ftB.begin(), ftB.end(), 0.0f);
        std::fill(l1W.begin(), l1W.end(), 0.0f);
        std::fill(l1B.begin(), l1B.end(), 0.0f);
        lossSum = 0.0;
    }

    inline void mark_feat(int f) {
        if (!featTouched[f]) {
            featTouched[f] = 1;
            dirtyFeats.push_back(f);
        }
    }
};

// AVX2 helper: dst[i] += src[i]
inline void vec_add(float* __restrict dst, const float* __restrict src, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(dst + i);
        __m256 b = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(a, b));
    }
    for (; i < n; ++i) dst[i] += src[i];
}

// AVX2 helper: dst[i] += scale * src[i]
inline void vec_fma(float* __restrict dst, const float* __restrict src, float scale, int n) {
    __m256 s = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(dst + i);
        __m256 b = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(b, s, a));
    }
    for (; i < n; ++i) dst[i] += scale * src[i];
}

// AdamW parameter block
struct AdamParam {
    std::vector<float> w;
    std::vector<float> m;
    std::vector<float> v;

    explicit AdamParam(size_t n) : w(n, 0.0f), m(n, 0.0f), v(n, 0.0f) {}

    void step_range(const float* grad, size_t begin, size_t end,
                    float invBatch, float lr, float wd, float clipVal,
                    float bc1, float bc2) {
        constexpr float beta1 = 0.9f;
        constexpr float beta2 = 0.999f;
        constexpr float eps = 1e-8f;
        float lrAdj = lr * std::sqrt(bc2) / bc1;

        for (size_t i = begin; i < end; ++i) {
            float g = grad[i] * invBatch;
            float mi = beta1 * m[i] + (1.0f - beta1) * g;
            float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
            m[i] = mi;
            v[i] = vi;
            float wi = w[i] * (1.0f - lr * wd) - lrAdj * mi / (std::sqrt(vi) + eps);
            if (wi > clipVal) wi = clipVal;
            else if (wi < -clipVal) wi = -clipVal;
            w[i] = wi;
        }
    }
};

void save_quantized(const std::string& path,
                    const AdamParam& ftW, const AdamParam& ftB,
                    const AdamParam& l1W, const AdamParam& l1B) {
    auto net = std::make_unique<QuantizedNet>();
    net->magic = EXPECTED_MAGIC;
    net->kingBuckets = KING_BUCKETS;
    net->hl = HL;
    net->outBuckets = OUT_BUCKETS;
    std::memset(net->pad, 0, sizeof(net->pad));

    for (size_t i = 0; i < size_t(INPUT_SIZE) * HL; ++i) {
        float v = std::round(ftW.w[i] * QA);
        net->ftWeights[i] = int16_t(std::clamp(v, -32000.0f, 32000.0f));
    }
    for (int i = 0; i < HL; ++i) {
        float v = std::round(ftB.w[i] * QA);
        net->ftBias[i] = int16_t(std::clamp(v, -32000.0f, 32000.0f));
    }
    for (int ob = 0; ob < OUT_BUCKETS; ++ob) {
        for (int i = 0; i < 2 * L1_PER_PERSP; ++i) {
            float v = std::round(l1W.w[ob * 2 * L1_PER_PERSP + i] * QB);
            net->l1Weights[ob][i] = int16_t(std::clamp(v, -127.0f, 127.0f));
        }
        net->l1Bias[ob] = int32_t(std::round(l1B.w[ob] * QA * QB));
    }

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(net.get()), sizeof(QuantizedNet));
    out.close();
    std::cout << "Saved quantized net to " << path << " (" << sizeof(QuantizedNet) << " bytes)\n" << std::flush;
}

void save_checkpoint(const std::string& path,
                     const AdamParam& ftW, const AdamParam& ftB,
                     const AdamParam& l1W, const AdamParam& l1B) {
    std::ofstream out(path, std::ios::binary);
    auto writeVec = [&](const std::vector<float>& v) {
        out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    };
    writeVec(ftW.w); writeVec(ftB.w); writeVec(l1W.w); writeVec(l1B.w);
    out.close();
}

bool load_checkpoint(const std::string& path,
                     AdamParam& ftW, AdamParam& ftB,
                     AdamParam& l1W, AdamParam& l1B) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    auto readVec = [&](std::vector<float>& v) {
        in.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(float));
    };
    readVec(ftW.w); readVec(ftB.w); readVec(l1W.w); readVec(l1B.w);
    return bool(in);
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> dataPaths;
    std::string outPath = "net.nnue";
    std::string resumePath;
    int threads = 14;
    int epochs = 20;
    int batchSize = 16384;
    float lrStart = 0.001f;
    float lrEnd   = 0.00002f;
    float wdlLambda = 0.25f;
    float evalScale = 400.0f;
    float weightDecay = 0.005f;
    size_t maxPos = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--data" && i + 1 < argc) dataPaths.push_back(argv[++i]);
        else if (a == "--out" && i + 1 < argc) outPath = argv[++i];
        else if (a == "--resume" && i + 1 < argc) resumePath = argv[++i];
        else if (a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (a == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) batchSize = std::stoi(argv[++i]);
        else if (a == "--lr" && i + 1 < argc) lrStart = std::stof(argv[++i]);
        else if (a == "--lr-end" && i + 1 < argc) lrEnd = std::stof(argv[++i]);
        else if (a == "--wdl" && i + 1 < argc) wdlLambda = std::stof(argv[++i]);
        else if (a == "--wd" && i + 1 < argc) weightDecay = std::stof(argv[++i]);
        else if (a == "--scale" && i + 1 < argc) evalScale = std::stof(argv[++i]);
        else if (a == "--max-pos" && i + 1 < argc) maxPos = std::stoull(argv[++i]);
    }

    if (dataPaths.empty()) {
        std::cerr << "Usage: trainer --data <file.bin> [--data <file2.bin> ...] [--out net.nnue] [--threads 14] [--epochs 20] [--wdl 0.25]\n";
        return 1;
    }

    std::vector<PackedPos> dataset;
    for (const auto& dp : dataPaths) {
        std::ifstream in(dp, std::ios::binary | std::ios::ate);
        if (!in) {
            std::cerr << "Cannot open dataset: " << dp << "\n";
            return 1;
        }
        size_t fileBytes = size_t(in.tellg());
        size_t cnt = fileBytes / sizeof(PackedPos);
        in.seekg(0);
        size_t oldSz = dataset.size();
        dataset.resize(oldSz + cnt);
        in.read(reinterpret_cast<char*>(dataset.data() + oldSz), cnt * sizeof(PackedPos));
        in.close();
        std::cout << "Loaded " << cnt << " positions from " << dp << "\n";
    }
    if (maxPos > 0 && dataset.size() > maxPos) dataset.resize(maxPos);
    size_t totalPos = dataset.size();

    std::cout << "NNUE Trainer: (" << KING_BUCKETS << "x768 -> " << HL
              << (PAIRWISE ? " Pairwise" : " SCReLU") << ")x2 -> " << OUT_BUCKETS
              << " | Total " << totalPos << " positions (" << (totalPos * 32 / (1024*1024)) << " MB)\n" << std::flush;

    // Reserve last 2% (capped at 200k) for validation
    size_t valCount = std::min<size_t>(200000, totalPos / 50);
    size_t trainCount = totalPos - valCount;
    std::cout << "Train: " << trainCount << ", Val: " << valCount
              << " | threads=" << threads << " epochs=" << epochs
              << " batch=" << batchSize << " lr=" << lrStart << "->" << lrEnd
              << " wdl=" << wdlLambda << "\n" << std::flush;

    // Initialize weights (He / Kaiming small uniform)
    AdamParam ftW(size_t(INPUT_SIZE) * HL);
    AdamParam ftB(HL);
    AdamParam l1W(OUT_BUCKETS * 2 * L1_PER_PERSP);
    AdamParam l1B(OUT_BUCKETS);

    std::mt19937 rng(42);
    {
        float ftBound = 1.0f / std::sqrt(32.0f);
        std::uniform_real_distribution<float> distFt(-ftBound * 0.25f, ftBound * 0.25f);
        for (float& x : ftW.w) x = distFt(rng);
        for (float& x : ftB.w) x = 0.0f;

        float l1Bound = 1.0f / std::sqrt(float(2 * L1_PER_PERSP));
        std::uniform_real_distribution<float> distL1(-l1Bound, l1Bound);
        for (float& x : l1W.w) x = distL1(rng);
        for (float& x : l1B.w) x = 0.0f;
    }

    if (!resumePath.empty()) {
        if (load_checkpoint(resumePath, ftW, ftB, l1W, l1B))
            std::cout << "Resumed float weights from " << resumePath << "\n";
    }

    // Allocate thread-local gradient buffers & master gradient buffer
    std::vector<std::unique_ptr<GradBuffer>> tGrads(threads);
    for (int t = 0; t < threads; ++t) tGrads[t] = std::make_unique<GradBuffer>();

    std::vector<float> sumFtW(size_t(INPUT_SIZE) * HL, 0.0f);
    std::vector<float> sumFtB(HL, 0.0f);
    std::vector<float> sumL1W(OUT_BUCKETS * 2 * L1_PER_PERSP, 0.0f);
    std::vector<float> sumL1B(OUT_BUCKETS, 0.0f);

    std::vector<uint32_t> indices(trainCount);
    for (size_t i = 0; i < trainCount; ++i) indices[i] = uint32_t(i);

    size_t batchesPerEpoch = trainCount / batchSize;
    size_t totalSteps = batchesPerEpoch * epochs;
    size_t globalStep = 0;

    auto tTrainStart = std::chrono::steady_clock::now();
    double bestValLoss = 1e9;

    for (int ep = 0; ep < epochs; ++ep) {
        auto tEpStart = std::chrono::steady_clock::now();
        std::shuffle(indices.begin(), indices.end(), rng);
        double epochLoss = 0.0;

        for (size_t b = 0; b < batchesPerEpoch; ++b) {
            globalStep++;
            size_t bStart = b * batchSize;

            // Cosine LR schedule
            float progress = float(globalStep) / float(std::max<size_t>(1, totalSteps));
            float lr = lrEnd + 0.5f * (lrStart - lrEnd) * (1.0f + std::cos(3.14159265f * progress));

            // 1. Parallel forward + backward across threads
            std::vector<std::thread> workers;
            workers.reserve(threads);
            for (int tid = 0; tid < threads; ++tid) {
                workers.emplace_back([&, tid]() {
                    GradBuffer& gb = *tGrads[tid];
                    gb.clear_fast();

                    size_t chunk = (batchSize + threads - 1) / threads;
                    size_t lo = bStart + tid * chunk;
                    size_t hi = std::min(bStart + batchSize, lo + chunk);

                    alignas(32) float accW[HL], accB[HL];
                    alignas(32) float actUs[HL], actThem[HL];
                    alignas(32) float dAccUs[HL], dAccThem[HL];
                    FeatureList fl;

                    const __m256 vZero = _mm256_setzero_ps();
                    const __m256 vOne  = _mm256_set1_ps(1.0f);
                    const __m256 vTwo  = _mm256_set1_ps(2.0f);

                    for (size_t idx = lo; idx < hi; ++idx) {
                        unpack_pos(dataset[indices[idx]], fl, wdlLambda, evalScale);

                        // Accumulator forward
                        std::memcpy(accW, ftB.w.data(), HL * sizeof(float));
                        std::memcpy(accB, ftB.w.data(), HL * sizeof(float));
                        for (int k = 0; k < fl.count; ++k) {
                            vec_add(accW, &ftW.w[size_t(fl.wFeats[k]) * HL], HL);
                            vec_add(accB, &ftW.w[size_t(fl.bFeats[k]) * HL], HL);
                        }

                        const float* usAcc   = (fl.stm == 0) ? accW : accB;
                        const float* themAcc = (fl.stm == 0) ? accB : accW;
                        const float* wUs     = &l1W.w[fl.outBucket * 2 * L1_PER_PERSP];
                        const float* wThem   = wUs + L1_PER_PERSP;

                        __m256 vDot = _mm256_setzero_ps();
                        if constexpr (PAIRWISE) {
                            constexpr int HALF = HL / 2;
                            for (int i = 0; i < HALF; i += 8) {
                                __m256 ua = _mm256_min_ps(vOne, _mm256_max_ps(vZero, _mm256_loadu_ps(usAcc + i)));
                                __m256 ub = _mm256_min_ps(vOne, _mm256_max_ps(vZero, _mm256_loadu_ps(usAcc + HALF + i)));
                                __m256 ta = _mm256_min_ps(vOne, _mm256_max_ps(vZero, _mm256_loadu_ps(themAcc + i)));
                                __m256 tb = _mm256_min_ps(vOne, _mm256_max_ps(vZero, _mm256_loadu_ps(themAcc + HALF + i)));
                                __m256 uab = _mm256_mul_ps(ua, ub);
                                __m256 tab = _mm256_mul_ps(ta, tb);
                                _mm256_storeu_ps(actUs + i, uab);
                                _mm256_storeu_ps(actThem + i, tab);
                                vDot = _mm256_fmadd_ps(uab, _mm256_loadu_ps(wUs + i), vDot);
                                vDot = _mm256_fmadd_ps(tab, _mm256_loadu_ps(wThem + i), vDot);
                            }
                        } else {
                            for (int i = 0; i < HL; i += 8) {
                                __m256 u = _mm256_min_ps(vOne, _mm256_max_ps(vZero, _mm256_loadu_ps(usAcc + i)));
                                __m256 t = _mm256_min_ps(vOne, _mm256_max_ps(vZero, _mm256_loadu_ps(themAcc + i)));
                                __m256 u2 = _mm256_mul_ps(u, u);
                                __m256 t2 = _mm256_mul_ps(t, t);
                                _mm256_storeu_ps(actUs + i, u2);
                                _mm256_storeu_ps(actThem + i, t2);
                                vDot = _mm256_fmadd_ps(u2, _mm256_loadu_ps(wUs + i), vDot);
                                vDot = _mm256_fmadd_ps(t2, _mm256_loadu_ps(wThem + i), vDot);
                            }
                        }
                        float tmp[8];
                        _mm256_storeu_ps(tmp, vDot);
                        float out = l1B.w[fl.outBucket] + tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];

                        float pred = sigmoid(out * (SCALE / evalScale));
                        float diff = pred - fl.target;
                        gb.lossSum += double(diff * diff);

                        float dOut = 2.0f * diff * pred * (1.0f - pred) * (SCALE / evalScale);
                        gb.l1B[fl.outBucket] += dOut;

                        float* gL1Us   = &gb.l1W[fl.outBucket * 2 * L1_PER_PERSP];
                        float* gL1Them = gL1Us + L1_PER_PERSP;
                        __m256 vDOut = _mm256_set1_ps(dOut);

                        if constexpr (PAIRWISE) {
                            constexpr int HALF = HL / 2;
                            for (int i = 0; i < HALF; i += 8) {
                                __m256 aU = _mm256_loadu_ps(actUs + i);
                                __m256 aT = _mm256_loadu_ps(actThem + i);
                                _mm256_storeu_ps(gL1Us + i,   _mm256_fmadd_ps(aU, vDOut, _mm256_loadu_ps(gL1Us + i)));
                                _mm256_storeu_ps(gL1Them + i, _mm256_fmadd_ps(aT, vDOut, _mm256_loadu_ps(gL1Them + i)));

                                __m256 dwU = _mm256_mul_ps(vDOut, _mm256_loadu_ps(wUs + i));
                                __m256 dwT = _mm256_mul_ps(vDOut, _mm256_loadu_ps(wThem + i));

                                __m256 uaRaw = _mm256_loadu_ps(usAcc + i);
                                __m256 ubRaw = _mm256_loadu_ps(usAcc + HALF + i);
                                __m256 uaC = _mm256_min_ps(vOne, _mm256_max_ps(vZero, uaRaw));
                                __m256 ubC = _mm256_min_ps(vOne, _mm256_max_ps(vZero, ubRaw));
                                __m256 uaM = _mm256_and_ps(_mm256_cmp_ps(uaRaw, vZero, _CMP_GT_OQ), _mm256_cmp_ps(uaRaw, vOne, _CMP_LT_OQ));
                                __m256 ubM = _mm256_and_ps(_mm256_cmp_ps(ubRaw, vZero, _CMP_GT_OQ), _mm256_cmp_ps(ubRaw, vOne, _CMP_LT_OQ));
                                _mm256_storeu_ps(dAccUs + i,        _mm256_and_ps(uaM, _mm256_mul_ps(dwU, ubC)));
                                _mm256_storeu_ps(dAccUs + HALF + i, _mm256_and_ps(ubM, _mm256_mul_ps(dwU, uaC)));

                                __m256 taRaw = _mm256_loadu_ps(themAcc + i);
                                __m256 tbRaw = _mm256_loadu_ps(themAcc + HALF + i);
                                __m256 taC = _mm256_min_ps(vOne, _mm256_max_ps(vZero, taRaw));
                                __m256 tbC = _mm256_min_ps(vOne, _mm256_max_ps(vZero, tbRaw));
                                __m256 taM = _mm256_and_ps(_mm256_cmp_ps(taRaw, vZero, _CMP_GT_OQ), _mm256_cmp_ps(taRaw, vOne, _CMP_LT_OQ));
                                __m256 tbM = _mm256_and_ps(_mm256_cmp_ps(tbRaw, vZero, _CMP_GT_OQ), _mm256_cmp_ps(tbRaw, vOne, _CMP_LT_OQ));
                                _mm256_storeu_ps(dAccThem + i,        _mm256_and_ps(taM, _mm256_mul_ps(dwT, tbC)));
                                _mm256_storeu_ps(dAccThem + HALF + i, _mm256_and_ps(tbM, _mm256_mul_ps(dwT, taC)));
                            }
                        } else {
                            for (int i = 0; i < HL; i += 8) {
                                __m256 aU = _mm256_loadu_ps(actUs + i);
                                __m256 aT = _mm256_loadu_ps(actThem + i);
                                _mm256_storeu_ps(gL1Us + i,   _mm256_fmadd_ps(aU, vDOut, _mm256_loadu_ps(gL1Us + i)));
                                _mm256_storeu_ps(gL1Them + i, _mm256_fmadd_ps(aT, vDOut, _mm256_loadu_ps(gL1Them + i)));

                                __m256 uRaw = _mm256_loadu_ps(usAcc + i);
                                __m256 tRaw = _mm256_loadu_ps(themAcc + i);
                                __m256 uMask = _mm256_and_ps(_mm256_cmp_ps(uRaw, vZero, _CMP_GT_OQ),
                                                             _mm256_cmp_ps(uRaw, vOne,  _CMP_LT_OQ));
                                __m256 tMask = _mm256_and_ps(_mm256_cmp_ps(tRaw, vZero, _CMP_GT_OQ),
                                                             _mm256_cmp_ps(tRaw, vOne,  _CMP_LT_OQ));
                                __m256 gU = _mm256_and_ps(uMask, _mm256_mul_ps(_mm256_mul_ps(vTwo, uRaw),
                                                                               _mm256_mul_ps(vDOut, _mm256_loadu_ps(wUs + i))));
                                __m256 gT = _mm256_and_ps(tMask, _mm256_mul_ps(_mm256_mul_ps(vTwo, tRaw),
                                                                               _mm256_mul_ps(vDOut, _mm256_loadu_ps(wThem + i))));
                                _mm256_storeu_ps(dAccUs + i, gU);
                                _mm256_storeu_ps(dAccThem + i, gT);
                            }
                        }

                        const float* dW = (fl.stm == 0) ? dAccUs : dAccThem;
                        const float* dB = (fl.stm == 0) ? dAccThem : dAccUs;

                        vec_add(gb.ftB.data(), dW, HL);
                        vec_add(gb.ftB.data(), dB, HL);
                        for (int k = 0; k < fl.count; ++k) {
                            int fw = fl.wFeats[k];
                            int fb = fl.bFeats[k];
                            gb.mark_feat(fw);
                            gb.mark_feat(fb);
                            vec_add(&gb.ftW[size_t(fw) * HL], dW, HL);
                            vec_add(&gb.ftW[size_t(fb) * HL], dB, HL);
                        }
                    }
                });
            }
            for (auto& w : workers) w.join();

            // 2. Reduce gradients & parallel AdamW update
            for (int t = 0; t < threads; ++t) epochLoss += tGrads[t]->lossSum;

            std::fill(sumFtB.begin(), sumFtB.end(), 0.0f);
            std::fill(sumL1W.begin(), sumL1W.end(), 0.0f);
            std::fill(sumL1B.begin(), sumL1B.end(), 0.0f);
            for (int t = 0; t < threads; ++t) {
                vec_add(sumFtB.data(), tGrads[t]->ftB.data(), HL);
                vec_add(sumL1W.data(), tGrads[t]->l1W.data(), OUT_BUCKETS * 2 * L1_PER_PERSP);
                vec_add(sumL1B.data(), tGrads[t]->l1B.data(), OUT_BUCKETS);
            }

            float invBatch = 1.0f / float(batchSize);
            float bc1 = 1.0f - std::pow(0.9f, float(globalStep));
            float bc2 = 1.0f - std::pow(0.999f, float(globalStep));

            // Small layers update on main thread
            ftB.step_range(sumFtB.data(), 0, HL, invBatch, lr, 0.0f, 1.98f, bc1, bc2);
            l1W.step_range(sumL1W.data(), 0, OUT_BUCKETS * 2 * L1_PER_PERSP, invBatch, lr, weightDecay, 1.98f, bc1, bc2);
            l1B.step_range(sumL1B.data(), 0, OUT_BUCKETS, invBatch, lr, 0.0f, 10.0f, bc1, bc2);

            // Parallel reduction + AdamW on FT weights (partitioned by feature index!)
            workers.clear();
            for (int tid = 0; tid < threads; ++tid) {
                workers.emplace_back([&, tid]() {
                    int fPerT = (INPUT_SIZE + threads - 1) / threads;
                    int fLo = tid * fPerT;
                    int fHi = std::min(INPUT_SIZE, fLo + fPerT);
                    for (int f = fLo; f < fHi; ++f) {
                        size_t off = size_t(f) * HL;
                        float* dst = &sumFtW[off];
                        bool any = false;
                        for (int t = 0; t < threads; ++t) {
                            if (tGrads[t]->featTouched[f]) {
                                if (!any) {
                                    std::memcpy(dst, &tGrads[t]->ftW[off], HL * sizeof(float));
                                    any = true;
                                } else {
                                    vec_add(dst, &tGrads[t]->ftW[off], HL);
                                }
                            }
                        }
                        if (!any) {
                            std::memset(dst, 0, HL * sizeof(float));
                        }
                        ftW.step_range(sumFtW.data(), off, off + HL, invBatch, lr, weightDecay, 1.98f, bc1, bc2);
                    }
                });
            }
            for (auto& w : workers) w.join();
        }

        // Validation loss
        double valLoss = 0.0;
        {
            alignas(32) float accW[HL], accB[HL];
            FeatureList fl;
            for (size_t idx = trainCount; idx < totalPos; ++idx) {
                unpack_pos(dataset[idx], fl, wdlLambda, evalScale);
                std::memcpy(accW, ftB.w.data(), HL * sizeof(float));
                std::memcpy(accB, ftB.w.data(), HL * sizeof(float));
                for (int k = 0; k < fl.count; ++k) {
                    vec_add(accW, &ftW.w[size_t(fl.wFeats[k]) * HL], HL);
                    vec_add(accB, &ftW.w[size_t(fl.bFeats[k]) * HL], HL);
                }
                const float* usAcc = (fl.stm == 0) ? accW : accB;
                const float* thAcc = (fl.stm == 0) ? accB : accW;
                const float* wUs   = &l1W.w[fl.outBucket * 2 * L1_PER_PERSP];
                const float* wTh   = wUs + L1_PER_PERSP;
                float out = l1B.w[fl.outBucket];
                if constexpr (PAIRWISE) {
                    constexpr int HALF = HL / 2;
                    for (int i = 0; i < HALF; ++i) {
                        float ua = std::clamp(usAcc[i], 0.0f, 1.0f);
                        float ub = std::clamp(usAcc[HALF + i], 0.0f, 1.0f);
                        float ta = std::clamp(thAcc[i], 0.0f, 1.0f);
                        float tb = std::clamp(thAcc[HALF + i], 0.0f, 1.0f);
                        out += ua * ub * wUs[i] + ta * tb * wTh[i];
                    }
                } else {
                    for (int i = 0; i < HL; ++i) {
                        float u = std::clamp(usAcc[i], 0.0f, 1.0f);
                        float t = std::clamp(thAcc[i], 0.0f, 1.0f);
                        out += u * u * wUs[i] + t * t * wTh[i];
                    }
                }
                float pred = sigmoid(out * (SCALE / evalScale));
                float diff = pred - fl.target;
                valLoss += double(diff * diff);
            }
            valLoss /= double(std::max<size_t>(1, valCount));
        }

        auto tEpEnd = std::chrono::steady_clock::now();
        double epSecs = std::chrono::duration<double>(tEpEnd - tEpStart).count();
        double avgTrainLoss = epochLoss / double(batchesPerEpoch * batchSize);

        std::cout << "Epoch " << std::setw(2) << (ep + 1) << "/" << epochs
                  << " | train_loss=" << std::fixed << std::setprecision(6) << avgTrainLoss
                  << " | val_loss=" << valLoss
                  << " | " << std::setprecision(1) << epSecs << "s ("
                  << size_t(double(batchesPerEpoch * batchSize) / std::max(epSecs, 1e-3)) << " pos/s)"
                  << (valLoss < bestValLoss ? " [BEST]" : "") << "\n" << std::flush;

        if (valLoss < bestValLoss || ep + 1 == epochs) {
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                save_quantized(outPath, ftW, ftB, l1W, l1B);
            }
        }
    }

    save_checkpoint(outPath + ".ckpt", ftW, ftB, l1W, l1B);
    auto tTrainEnd = std::chrono::steady_clock::now();
    std::cout << "Training complete in "
              << std::chrono::duration<double>(tTrainEnd - tTrainStart).count() << "s!\n";
    return 0;
}
