#include "eval.h"
#include <algorithm>
#include <cstdlib>

namespace Eval {

namespace {

// Packed MG/EG score representation
using PScore = int32_t;
constexpr PScore S(int mg, int eg) {
    return static_cast<PScore>((static_cast<uint32_t>(eg) << 16) + mg);
}
constexpr int mg_val(PScore s) { return static_cast<int16_t>(s); }
constexpr int eg_val(PScore s) { return static_cast<int16_t>((s + 0x8000) >> 16); }

// Material values (MG, EG)
constexpr PScore PieceValue[PIECE_TYPE_NB] = {
    S( 95, 105), // PAWN
    S(325, 315), // KNIGHT
    S(345, 330), // BISHOP
    S(490, 545), // ROOK
    S(995, 1030),// QUEEN
    S(  0,   0)  // KING
};

constexpr int PhaseWeight[PIECE_TYPE_NB] = { 0, 1, 1, 2, 4, 0 };
constexpr int MAX_PHASE = 24;

// Piece-Square Tables (from Black's perspective a8..h1, flipped for White using flip_rank)
// Wait: let's define directly in rank 0 (rank 1) .. rank 7 (rank 8) for WHITE!
// Actually easier to read in standard chess diagram order (rank 8 at top, rank 1 at bottom),
// then mirror into rank 0..7 in init()!

constexpr int PawnMG_Top[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     88, 104,  88,  95, 105,  90,  55,  30,
     18,  25,  38,  48,  52,  42,  25,  15,
      2,   8,  14,  24,  28,  18,   8,  -2,
     -6,   2,   8,  18,  20,  10,   2,  -6,
    -10,  -4,   2,   5,   8,   2,   4,  -8,
    -16,  -8, -10, -14, -14,   8,  14, -10,
      0,   0,   0,   0,   0,   0,   0,   0
};

constexpr int PawnEG_Top[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    160, 155, 145, 135, 135, 145, 155, 160,
     85,  80,  70,  62,  62,  70,  80,  85,
     35,  32,  24,  18,  18,  24,  32,  35,
     14,  12,   6,   2,   2,   6,  12,  14,
      4,   2,  -2,  -4,  -4,  -2,   2,   4,
      8,   6,   4,   2,   2,   4,   6,   8,
      0,   0,   0,   0,   0,   0,   0,   0
};

constexpr int KnightMG_Top[64] = {
    -65, -40, -25, -20, -20, -25, -40, -65,
    -35, -10,   5,  12,  12,   5, -10, -35,
    -15,  12,  28,  34,  34,  28,  12, -15,
    -10,  10,  22,  30,  30,  22,  10, -10,
    -12,   4,  16,  22,  22,  16,   4, -12,
    -18,  -2,  10,  14,  14,  10,  -2, -18,
    -30, -15,  -4,   2,   2,  -4, -15, -30,
    -50, -30, -20, -15, -15, -20, -30, -50
};

constexpr int KnightEG_Top[64] = {
    -45, -30, -20, -15, -15, -20, -30, -45,
    -28, -14,  -4,   2,   2,  -4, -14, -28,
    -18,  -2,  12,  18,  18,  12,  -2, -18,
    -14,   2,  16,  24,  24,  16,   2, -14,
    -14,   2,  16,  22,  22,  16,   2, -14,
    -20,  -6,   8,  12,  12,   8,  -6, -20,
    -30, -16,  -8,  -4,  -4,  -8, -16, -30,
    -45, -30, -22, -18, -18, -22, -30, -45
};

constexpr int BishopMG_Top[64] = {
    -24, -12, -14, -16, -16, -14, -12, -24,
    -14,   4,   2,   0,   0,   2,   4, -14,
     -8,  12,  16,  18,  18,  16,  12,  -8,
     -6,   8,  16,  22,  22,  16,   8,  -6,
     -6,  10,  12,  20,  20,  12,  10,  -6,
     -4,  12,  14,  12,  12,  14,  12,  -4,
     -6,  16,   8,   6,   6,   8,  16,  -6,
    -20, -10, -12, -10, -10, -12, -10, -20
};

constexpr int BishopEG_Top[64] = {
    -18, -10,  -8,  -6,  -6,  -8, -10, -18,
    -10,  -2,   2,   4,   4,   2,  -2, -10,
     -6,   2,   8,  10,  10,   8,   2,  -6,
     -4,   4,  10,  14,  14,  10,   4,  -4,
     -4,   4,  10,  14,  14,  10,   4,  -4,
     -6,   2,   8,  10,  10,   8,   2,  -6,
    -10,  -4,   0,   2,   2,   0,  -4, -10,
    -18, -10,  -8,  -6,  -6,  -8, -10, -18
};

constexpr int RookMG_Top[64] = {
      8,  10,  12,  14,  14,  12,  10,   8,
     18,  22,  26,  28,  28,  26,  22,  18,
     -4,   4,   6,   8,   8,   6,   4,  -4,
    -10,  -4,   0,   4,   4,   0,  -4, -10,
    -14,  -8,  -4,   0,   0,  -4,  -8, -14,
    -14,  -8,  -4,   0,   0,  -4,  -8, -14,
    -12,  -8,  -4,   2,   2,  -4,  -8, -12,
     -8,  -6,   2,  10,  10,   6,  -6,  -8
};

constexpr int RookEG_Top[64] = {
     10,  12,  12,  14,  14,  12,  12,  10,
     10,  12,  14,  14,  14,  14,  12,  10,
      4,   6,   8,   8,   8,   8,   6,   4,
      2,   4,   6,   6,   6,   6,   4,   2,
     -2,   0,   2,   2,   2,   2,   0,  -2,
     -6,  -4,  -2,  -2,  -2,  -2,  -4,  -6,
     -8,  -6,  -4,  -4,  -4,  -4,  -6,  -8,
    -10,  -8,  -6,  -4,  -4,  -6,  -8, -10
};

constexpr int QueenMG_Top[64] = {
    -20, -10,  -6,  -4,  -4,  -6, -10, -20,
    -10,   0,   4,   6,   6,   4,   0, -10,
     -6,   4,   8,  10,  10,   8,   4,  -6,
     -4,   4,   8,  10,  10,   8,   4,  -4,
     -4,   4,   8,  10,  10,   8,   4,  -4,
     -6,   4,   8,   8,   8,   8,   4,  -6,
    -10,   2,   6,   8,   8,   6,   2, -10,
    -16, -10,  -6,  -2,  -2,  -6, -10, -16
};

constexpr int QueenEG_Top[64] = {
    -24, -14, -10,  -8,  -8, -10, -14, -24,
    -12,  -2,   4,   8,   8,   4,  -2, -12,
     -8,   4,  12,  16,  16,  12,   4,  -8,
     -6,   6,  16,  22,  22,  16,   6,  -6,
     -6,   6,  16,  22,  22,  16,   6,  -6,
    -10,   2,  10,  14,  14,  10,   2, -10,
    -16,  -6,   0,   4,   4,   0,  -6, -16,
    -28, -18, -12, -10, -10, -12, -18, -28
};

constexpr int KingMG_Top[64] = {
    -50, -45, -45, -50, -50, -45, -45, -50,
    -45, -40, -40, -45, -45, -40, -40, -45,
    -40, -35, -35, -40, -40, -35, -35, -40,
    -35, -30, -30, -35, -35, -30, -30, -35,
    -28, -24, -24, -30, -30, -24, -24, -28,
    -18, -14, -16, -22, -22, -16, -14, -18,
      4,   8,  -6, -16, -16,  -6,   8,   4,
     14,  26,  10, -14,   0, -10,  28,  16
};

constexpr int KingEG_Top[64] = {
    -45, -30, -20, -15, -15, -20, -30, -45,
    -25,  -8,   4,  10,  10,   4,  -8, -25,
    -15,   4,  18,  24,  24,  18,   4, -15,
    -12,   8,  22,  30,  30,  22,   8, -12,
    -14,   6,  20,  28,  28,  20,   6, -14,
    -18,   0,  12,  18,  18,  12,   0, -18,
    -28, -12,  -2,   4,   4,  -2, -12, -28,
    -45, -30, -20, -15, -15, -20, -30, -45
};

PScore PST[COLOR_NB][PIECE_TYPE_NB][SQ_NB];

// Mobility bonuses (tapered)
constexpr PScore KnightMob[9] = {
    S(-28,-32), S(-14,-16), S(-4,-6), S(4,4), S(10,12), S(16,18), S(22,22), S(26,25), S(30,26)
};
constexpr PScore BishopMob[14] = {
    S(-30,-34), S(-16,-18), S(-6,-6), S(2,4), S(10,12), S(16,20), S(22,26),
    S(27,32), S(31,36), S(35,40), S(38,43), S(41,45), S(44,47), S(46,48)
};
constexpr PScore RookMob[15] = {
    S(-22,-28), S(-12,-14), S(-6,-4), S(-2,4), S(2,12), S(6,18), S(10,24),
    S(14,30), S(18,35), S(22,40), S(25,44), S(28,48), S(30,52), S(32,54), S(34,56)
};
constexpr PScore QueenMob[28] = {
    S(-20,-24), S(-12,-16), S(-6,-8), S(-2,-2), S(1,4), S(4,10), S(7,15),
    S(10,20), S(12,24), S(14,28), S(16,32), S(18,35), S(20,38), S(22,41),
    S(24,44), S(25,46), S(27,48), S(28,50), S(29,52), S(30,54), S(31,55),
    S(32,56), S(33,57), S(34,58), S(35,59), S(36,60), S(37,61), S(38,62)
};

// Passed pawn bonus by relative rank (0..7)
constexpr PScore PassedBonus[8] = {
    S(0, 0), S(4, 8), S(8, 16), S(16, 30), S(34, 58), S(65, 105), S(110, 175), S(0, 0)
};

// Connected / defended pawn bonus by relative rank
constexpr PScore ConnectedBonus[8] = {
    S(0, 0), S(4, 2), S(8, 5), S(12, 8), S(18, 14), S(28, 24), S(40, 36), S(0, 0)
};

// Pawn masks
Bitboard AdjacentFiles[8];
Bitboard PassedMask[COLOR_NB][64];
Bitboard ForwardFileMask[COLOR_NB][64];
Bitboard KingZone[COLOR_NB][64];

// Attack weights for king safety
constexpr int AttackWeight[PIECE_TYPE_NB] = { 0, 14, 14, 24, 42, 0 };
constexpr int SafeCheckBonus[PIECE_TYPE_NB] = { 0, 18, 14, 26, 36, 0 };

// Pawn hash table (per-thread / single-threaded fast cache)
struct PawnHashEntry {
    U64 key = ~0ULL;
    PScore score = 0;
    Bitboard passed[2] = {0, 0};
};
constexpr size_t PAWN_HASH_SIZE = 16384;
thread_local PawnHashEntry PawnTable[PAWN_HASH_SIZE];

inline int chebyshev_dist(Square a, Square b) {
    return std::max(std::abs(file_of(a) - file_of(b)), std::abs(rank_of(a) - rank_of(b)));
}

void evaluate_pawns_uncached(const Position& pos, PScore& outScore, Bitboard passedOut[2]) {
    PScore score = 0;
    passedOut[WHITE] = 0;
    passedOut[BLACK] = 0;

    Bitboard wp = pos.pieces(WHITE, PAWN);
    Bitboard bp = pos.pieces(BLACK, PAWN);
    Bitboard wAtt = pawn_attacks_bb<WHITE>(wp);
    Bitboard bAtt = pawn_attacks_bb<BLACK>(bp);

    // White pawns
    Bitboard b = wp;
    while (b) {
        Square s = pop_lsb(b);
        int f = file_of(s), r = rank_of(s);
        Bitboard phalanx = wp & rank_bb(r) & AdjacentFiles[f];
        bool supported = (wAtt & square_bb(s)) != 0;
        bool isolated = !(wp & AdjacentFiles[f]);
        bool doubled = (wp & ForwardFileMask[WHITE][s]) != 0;
        bool opposed = (bp & ForwardFileMask[WHITE][s]) != 0;

        if (isolated) score -= opposed ? S(10, 14) : S(16, 18);
        if (doubled)  score -= S(12, 22);
        if (supported || phalanx) {
            score += ConnectedBonus[r];
            if (supported) score += S(4, 4);
        } else if (!isolated && !opposed) {
            Bitboard adjSpan = AdjacentFiles[f] & ~PassedMask[WHITE][s];
            if (!(wp & adjSpan) && (bAtt & square_bb(Square(s + 8))))
                score -= S(10, 12);
        }

        if (!(bp & PassedMask[WHITE][s])) {
            passedOut[WHITE] |= square_bb(s);
            score += PassedBonus[r];
            if (supported || phalanx) score += S(r * 3, r * 5);
        }
    }

    // Black pawns
    b = bp;
    while (b) {
        Square s = pop_lsb(b);
        int f = file_of(s), r = 7 - rank_of(s);
        Bitboard phalanx = bp & rank_bb(rank_of(s)) & AdjacentFiles[f];
        bool supported = (bAtt & square_bb(s)) != 0;
        bool isolated = !(bp & AdjacentFiles[f]);
        bool doubled = (bp & ForwardFileMask[BLACK][s]) != 0;
        bool opposed = (wp & ForwardFileMask[BLACK][s]) != 0;

        if (isolated) score += opposed ? S(10, 14) : S(16, 18);
        if (doubled)  score += S(12, 22);
        if (supported || phalanx) {
            score -= ConnectedBonus[r];
            if (supported) score -= S(4, 4);
        } else if (!isolated && !opposed) {
            Bitboard adjSpan = AdjacentFiles[f] & ~PassedMask[BLACK][s];
            if (!(bp & adjSpan) && (wAtt & square_bb(Square(s - 8))))
                score += S(10, 12);
        }

        if (!(wp & PassedMask[BLACK][s])) {
            passedOut[BLACK] |= square_bb(s);
            score -= PassedBonus[r];
            if (supported || phalanx) score -= S(r * 3, r * 5);
        }
    }

    outScore = score;
}

template<Color Us>
PScore eval_king_shield(const Position& pos) {
    Square ksq = pos.king_square(Us);
    int kf = std::clamp(file_of(ksq), 1, 6);
    int kr = relative_rank(Us, ksq);
    if (kr > 2) return 0;

    Bitboard ourPawns = pos.pieces(Us, PAWN);
    Bitboard theirPawns = pos.pieces(~Us, PAWN);
    int mg = 0;

    for (int f = kf - 1; f <= kf + 1; ++f) {
        Bitboard fbb = file_bb(f);
        Bitboard ours = ourPawns & fbb;
        int ourDist = 7;
        if (ours) {
            Square ps = (Us == WHITE) ? lsb(ours) : msb(ours);
            int pr = relative_rank(Us, ps);
            if (pr >= kr) ourDist = pr - kr;
        }
        if (ourDist == 1)      mg += (f == kf ? 18 : 13);
        else if (ourDist == 2) mg += (f == kf ? 8 : 6);
        else if (ourDist >= 4) mg -= (f == kf ? 18 : 11);

        Bitboard theirs = theirPawns & fbb;
        if (theirs) {
            Square ps = (Us == WHITE) ? lsb(theirs) : msb(theirs);
            int pr = relative_rank(Us, ps);
            if (pr == 3)      mg -= 12;
            else if (pr == 2) mg -= 22;
        } else if (!ours) {
            mg -= (f == kf ? 14 : 9); // open file near king
        }
    }
    return S(mg, 0);
}

} // namespace

void init() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    const int* mgTables[6] = { PawnMG_Top, KnightMG_Top, BishopMG_Top, RookMG_Top, QueenMG_Top, KingMG_Top };
    const int* egTables[6] = { PawnEG_Top, KnightEG_Top, BishopEG_Top, RookEG_Top, QueenEG_Top, KingEG_Top };

    for (int pt = 0; pt < 6; ++pt) {
        for (int s = 0; s < 64; ++s) {
            int topIdx = flip_rank(Square(s)); // s=SQ_A1 -> rank 0 -> bottom row (topIdx=56)
            int mg = mg_val(PieceValue[pt]) + mgTables[pt][topIdx];
            int eg = eg_val(PieceValue[pt]) + egTables[pt][topIdx];
            PST[WHITE][pt][s] = S(mg, eg);
            PST[BLACK][pt][flip_rank(Square(s))] = S(mg, eg);
        }
    }

    for (int f = 0; f < 8; ++f) {
        AdjacentFiles[f] = 0;
        if (f > 0) AdjacentFiles[f] |= file_bb(f - 1);
        if (f < 7) AdjacentFiles[f] |= file_bb(f + 1);
    }

    for (int s = 0; s < 64; ++s) {
        int r = rank_of(Square(s)), f = file_of(Square(s));
        Bitboard wAhead = 0, bAhead = 0;
        for (int rr = r + 1; rr < 8; ++rr) wAhead |= rank_bb(rr);
        for (int rr = r - 1; rr >= 0; --rr) bAhead |= rank_bb(rr);

        ForwardFileMask[WHITE][s] = wAhead & file_bb(f);
        ForwardFileMask[BLACK][s] = bAhead & file_bb(f);
        PassedMask[WHITE][s] = wAhead & (file_bb(f) | AdjacentFiles[f]);
        PassedMask[BLACK][s] = bAhead & (file_bb(f) | AdjacentFiles[f]);

        Bitboard kz = Bitboards::king_attacks(Square(s)) | square_bb(Square(s));
        KingZone[WHITE][s] = kz | shift_north(kz);
        KingZone[BLACK][s] = kz | shift_south(kz);
    }
}

int evaluate(const Position& pos) {
    PScore score = 0;
    int phase = 0;

    Bitboard occ = pos.pieces();
    Bitboard wp = pos.pieces(WHITE, PAWN);
    Bitboard bp = pos.pieces(BLACK, PAWN);
    Bitboard wPawnAtt = pawn_attacks_bb<WHITE>(wp);
    Bitboard bPawnAtt = pawn_attacks_bb<BLACK>(bp);

    Square wKing = pos.king_square(WHITE);
    Square bKing = pos.king_square(BLACK);

    // 1. Material + PSTs + Phase
    for (int pt = PAWN; pt <= KING; ++pt) {
        Bitboard wb = pos.pieces(WHITE, PieceType(pt));
        Bitboard bb = pos.pieces(BLACK, PieceType(pt));
        if (PhaseWeight[pt])
            phase += PhaseWeight[pt] * (popcount(wb) + popcount(bb));
        while (wb) score += PST[WHITE][pt][pop_lsb(wb)];
        while (bb) score -= PST[BLACK][pt][pop_lsb(bb)];
    }

    // Bishop pair
    if (popcount(pos.pieces(WHITE, BISHOP)) >= 2) score += S(32, 54);
    if (popcount(pos.pieces(BLACK, BISHOP)) >= 2) score -= S(32, 54);

    // 2. Pawn structure (cached)
    U64 pKey = pos.pawn_key();
    PawnHashEntry& pe = PawnTable[pKey & (PAWN_HASH_SIZE - 1)];
    if (pe.key != pKey) {
        evaluate_pawns_uncached(pos, pe.score, pe.passed);
        pe.key = pKey;
    }
    score += pe.score;

    // Dynamic passed pawn evaluation (blockade + king proximity)
    for (int c = 0; c < 2; ++c) {
        Color us = Color(c);
        Square ourK = (us == WHITE) ? wKing : bKing;
        Square theirK = (us == WHITE) ? bKing : wKing;
        Bitboard passed = pe.passed[us];
        int sign = (us == WHITE) ? 1 : -1;
        while (passed) {
            Square s = pop_lsb(passed);
            int r = relative_rank(us, s);
            if (r >= 3) {
                Square stopSq = Square(s + (us == WHITE ? 8 : -8));
                int w = (r - 2) * (r - 2);
                int egBonus = (chebyshev_dist(stopSq, theirK) * 5 - chebyshev_dist(stopSq, ourK) * 2) * w;
                if (!(occ & square_bb(stopSq))) egBonus += w * 4;
                score += sign * S(0, egBonus);
            }
        }
    }

    // 3. Mobility, Outposts, Rooks on files, King Safety, Threats
    Bitboard wMobilityArea = ~(wp | square_bb(wKing) | bPawnAtt);
    Bitboard bMobilityArea = ~(bp | square_bb(bKing) | wPawnAtt);

    int wKingDanger = 0, bKingDanger = 0;
    int wAttackers = 0, bAttackers = 0;
    Bitboard bKingZone = KingZone[BLACK][bKing];
    Bitboard wKingZone = KingZone[WHITE][wKing];

    Bitboard wAttacks[PIECE_TYPE_NB] = {};
    Bitboard bAttacks[PIECE_TYPE_NB] = {};
    wAttacks[PAWN] = wPawnAtt;
    bAttacks[PAWN] = bPawnAtt;

    Bitboard bKingRookChecks = Bitboards::rook_attacks(bKing, occ);
    Bitboard bKingBishopChecks = Bitboards::bishop_attacks(bKing, occ);
    Bitboard wKingRookChecks = Bitboards::rook_attacks(wKing, occ);
    Bitboard wKingBishopChecks = Bitboards::bishop_attacks(wKing, occ);

    // White pieces
    {
        Bitboard knights = pos.pieces(WHITE, KNIGHT);
        while (knights) {
            Square s = pop_lsb(knights);
            Bitboard att = Bitboards::knight_attacks(s);
            wAttacks[KNIGHT] |= att;
            score += KnightMob[popcount(att & wMobilityArea)];
            if (att & bKingZone) { bKingDanger += AttackWeight[KNIGHT]; bAttackers++; }
            // Outpost
            int r = rank_of(s);
            if (r >= 3 && r <= 5 && (wPawnAtt & square_bb(s)) && !(bp & PassedMask[WHITE][s] & AdjacentFiles[file_of(s)]))
                score += S(24, 14);
        }

        Bitboard bishops = pos.pieces(WHITE, BISHOP);
        Bitboard occNoBQ = occ ^ pos.pieces(WHITE, QUEEN);
        while (bishops) {
            Square s = pop_lsb(bishops);
            Bitboard att = Bitboards::bishop_attacks(s, occNoBQ);
            wAttacks[BISHOP] |= att;
            score += BishopMob[popcount(att & wMobilityArea)];
            if (att & bKingZone) { bKingDanger += AttackWeight[BISHOP]; bAttackers++; }
            int r = rank_of(s);
            if (r >= 3 && r <= 5 && (wPawnAtt & square_bb(s)) && !(bp & PassedMask[WHITE][s] & AdjacentFiles[file_of(s)]))
                score += S(14, 8);
        }

        Bitboard rooks = pos.pieces(WHITE, ROOK);
        Bitboard occNoRQ = occ ^ pos.pieces(WHITE, ROOK, QUEEN);
        while (rooks) {
            Square s = pop_lsb(rooks);
            Bitboard att = Bitboards::rook_attacks(s, occNoRQ);
            wAttacks[ROOK] |= att;
            score += RookMob[popcount(att & wMobilityArea)];
            if (att & bKingZone) { bKingDanger += AttackWeight[ROOK]; bAttackers++; }
            Bitboard fbb = file_bb(file_of(s));
            if (!(wp & fbb)) {
                score += (!(bp & fbb)) ? S(26, 12) : S(13, 6);
            }
            if (rank_of(s) == 6 && (rank_of(bKing) == 7 || (bp & RANK_7_BB)))
                score += S(18, 28);
        }

        Bitboard queens = pos.pieces(WHITE, QUEEN);
        while (queens) {
            Square s = pop_lsb(queens);
            Bitboard att = Bitboards::queen_attacks(s, occ);
            wAttacks[QUEEN] |= att;
            score += QueenMob[std::min(27, popcount(att & wMobilityArea))];
            if (att & bKingZone) { bKingDanger += AttackWeight[QUEEN]; bAttackers++; }
        }
    }

    // Black pieces
    {
        Bitboard knights = pos.pieces(BLACK, KNIGHT);
        while (knights) {
            Square s = pop_lsb(knights);
            Bitboard att = Bitboards::knight_attacks(s);
            bAttacks[KNIGHT] |= att;
            score -= KnightMob[popcount(att & bMobilityArea)];
            if (att & wKingZone) { wKingDanger += AttackWeight[KNIGHT]; wAttackers++; }
            int r = 7 - rank_of(s);
            if (r >= 3 && r <= 5 && (bPawnAtt & square_bb(s)) && !(wp & PassedMask[BLACK][s] & AdjacentFiles[file_of(s)]))
                score -= S(24, 14);
        }

        Bitboard bishops = pos.pieces(BLACK, BISHOP);
        Bitboard occNoBQ = occ ^ pos.pieces(BLACK, QUEEN);
        while (bishops) {
            Square s = pop_lsb(bishops);
            Bitboard att = Bitboards::bishop_attacks(s, occNoBQ);
            bAttacks[BISHOP] |= att;
            score -= BishopMob[popcount(att & bMobilityArea)];
            if (att & wKingZone) { wKingDanger += AttackWeight[BISHOP]; wAttackers++; }
            int r = 7 - rank_of(s);
            if (r >= 3 && r <= 5 && (bPawnAtt & square_bb(s)) && !(wp & PassedMask[BLACK][s] & AdjacentFiles[file_of(s)]))
                score -= S(14, 8);
        }

        Bitboard rooks = pos.pieces(BLACK, ROOK);
        Bitboard occNoRQ = occ ^ pos.pieces(BLACK, ROOK, QUEEN);
        while (rooks) {
            Square s = pop_lsb(rooks);
            Bitboard att = Bitboards::rook_attacks(s, occNoRQ);
            bAttacks[ROOK] |= att;
            score -= RookMob[popcount(att & bMobilityArea)];
            if (att & wKingZone) { wKingDanger += AttackWeight[ROOK]; wAttackers++; }
            Bitboard fbb = file_bb(file_of(s));
            if (!(bp & fbb)) {
                score -= (!(wp & fbb)) ? S(26, 12) : S(13, 6);
            }
            if (rank_of(s) == 1 && (rank_of(wKing) == 0 || (wp & RANK_2_BB)))
                score -= S(18, 28);
        }

        Bitboard queens = pos.pieces(BLACK, QUEEN);
        while (queens) {
            Square s = pop_lsb(queens);
            Bitboard att = Bitboards::queen_attacks(s, occ);
            bAttacks[QUEEN] |= att;
            score -= QueenMob[std::min(27, popcount(att & bMobilityArea))];
            if (att & wKingZone) { wKingDanger += AttackWeight[QUEEN]; wAttackers++; }
        }
    }

    // 4. King safety (shield + quadratic attack danger + safe checks)
    score += eval_king_shield<WHITE>(pos);
    score -= eval_king_shield<BLACK>(pos);

    Bitboard wAllAtt = wAttacks[PAWN] | wAttacks[KNIGHT] | wAttacks[BISHOP] | wAttacks[ROOK] | wAttacks[QUEEN] | Bitboards::king_attacks(wKing);
    Bitboard bAllAtt = bAttacks[PAWN] | bAttacks[KNIGHT] | bAttacks[BISHOP] | bAttacks[ROOK] | bAttacks[QUEEN] | Bitboards::king_attacks(bKing);

    if (bAttackers >= 2) {
        Bitboard safeForW = ~(pos.pieces(WHITE) | bAllAtt);
        if (wAttacks[KNIGHT] & Bitboards::knight_attacks(bKing) & safeForW) bKingDanger += SafeCheckBonus[KNIGHT];
        if (wAttacks[BISHOP] & bKingBishopChecks & safeForW) bKingDanger += SafeCheckBonus[BISHOP];
        if (wAttacks[ROOK]   & bKingRookChecks   & safeForW) bKingDanger += SafeCheckBonus[ROOK];
        if (wAttacks[QUEEN]  & (bKingRookChecks | bKingBishopChecks) & safeForW) bKingDanger += SafeCheckBonus[QUEEN];
        if (!pos.pieces(WHITE, QUEEN)) bKingDanger = bKingDanger * 2 / 5;
        int d = std::min(bKingDanger, 240);
        score += S((d * d) / 280, d / 6);
    }

    if (wAttackers >= 2) {
        Bitboard safeForB = ~(pos.pieces(BLACK) | wAllAtt);
        if (bAttacks[KNIGHT] & Bitboards::knight_attacks(wKing) & safeForB) wKingDanger += SafeCheckBonus[KNIGHT];
        if (bAttacks[BISHOP] & wKingBishopChecks & safeForB) wKingDanger += SafeCheckBonus[BISHOP];
        if (bAttacks[ROOK]   & wKingRookChecks   & safeForB) wKingDanger += SafeCheckBonus[ROOK];
        if (bAttacks[QUEEN]  & (wKingRookChecks | wKingBishopChecks) & safeForB) wKingDanger += SafeCheckBonus[QUEEN];
        if (!pos.pieces(BLACK, QUEEN)) wKingDanger = wKingDanger * 2 / 5;
        int d = std::min(wKingDanger, 240);
        score -= S((d * d) / 280, d / 6);
    }

    // 5. Threats
    score += S(38, 26) * popcount(wPawnAtt & pos.pieces(BLACK, KNIGHT, BISHOP));
    score += S(52, 34) * popcount(wPawnAtt & pos.pieces(BLACK, ROOK));
    score += S(60, 40) * popcount(wPawnAtt & pos.pieces(BLACK, QUEEN));
    score += S(28, 20) * popcount((wAttacks[KNIGHT] | wAttacks[BISHOP]) & pos.pieces(BLACK, ROOK, QUEEN));
    score += S(24, 16) * popcount(wAttacks[ROOK] & pos.pieces(BLACK, QUEEN));

    score -= S(38, 26) * popcount(bPawnAtt & pos.pieces(WHITE, KNIGHT, BISHOP));
    score -= S(52, 34) * popcount(bPawnAtt & pos.pieces(WHITE, ROOK));
    score -= S(60, 40) * popcount(bPawnAtt & pos.pieces(WHITE, QUEEN));
    score -= S(28, 20) * popcount((bAttacks[KNIGHT] | bAttacks[BISHOP]) & pos.pieces(WHITE, ROOK, QUEEN));
    score -= S(24, 16) * popcount(bAttacks[ROOK] & pos.pieces(WHITE, QUEEN));

    // 6. Tapered interpolation
    phase = std::clamp(phase, 0, MAX_PHASE);
    int mg = mg_val(score);
    int eg = eg_val(score);

    // Endgame scaling: opposite-colored bishops
    if (phase <= 4 && popcount(pos.pieces(WHITE, BISHOP)) == 1 && popcount(pos.pieces(BLACK, BISHOP)) == 1
        && !pos.pieces(KNIGHT) && !pos.pieces(ROOK) && !pos.pieces(QUEEN)) {
        Square wb = lsb(pos.pieces(WHITE, BISHOP));
        Square bb = lsb(pos.pieces(BLACK, BISHOP));
        if (((rank_of(wb) + file_of(wb)) & 1) != ((rank_of(bb) + file_of(bb)) & 1)) {
            eg = eg * 55 / 100;
        }
    }

    // Winning side has no pawns -> scale down drawish endgames
    int eval = (mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE;
    Color strong = eval >= 0 ? WHITE : BLACK;
    if (pos.pieces(strong, PAWN) == 0) {
        int strongNonPawn = popcount(pos.pieces(strong, KNIGHT)) * 320
                          + popcount(pos.pieces(strong, BISHOP)) * 330
                          + popcount(pos.pieces(strong, ROOK)) * 500
                          + popcount(pos.pieces(strong, QUEEN)) * 950;
        int weakNonPawn   = popcount(pos.pieces(~strong, KNIGHT)) * 320
                          + popcount(pos.pieces(~strong, BISHOP)) * 330
                          + popcount(pos.pieces(~strong, ROOK)) * 500
                          + popcount(pos.pieces(~strong, QUEEN)) * 950;
        if (strongNonPawn - weakNonPawn <= 350 && strongNonPawn < 650) {
            eval = eval / 4;
        }
    }

    // 50-move rule damping
    eval = eval * (200 - pos.rule50_count()) / 200;

    // Return from side-to-move perspective + tempo
    constexpr int TEMPO = 14;
    return (pos.side_to_move() == WHITE ? eval : -eval) + TEMPO;
}

} // namespace Eval
