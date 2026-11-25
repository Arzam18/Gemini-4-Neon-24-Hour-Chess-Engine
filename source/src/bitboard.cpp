#include "bitboard.h"
#include <cstdlib>

namespace Bitboards {

Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard PawnAttacks[2][64];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];
Bitboard RookPseudoAttacks[64];
Bitboard BishopPseudoAttacks[64];

Bitboard RookMasks[64];
Bitboard BishopMasks[64];
U32 RookOffsets[64];
U32 BishopOffsets[64];
Bitboard RookTable[102400];
Bitboard BishopTable[5248];

namespace {

Bitboard sliding_attacks_slow(Square s, Bitboard occ, const int dirs[4][2]) {
    Bitboard attacks = 0;
    int r0 = rank_of(s), f0 = file_of(s);
    for (int d = 0; d < 4; ++d) {
        int r = r0 + dirs[d][0];
        int f = f0 + dirs[d][1];
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            Bitboard sq = square_bb(make_square(f, r));
            attacks |= sq;
            if (occ & sq) break;
            r += dirs[d][0];
            f += dirs[d][1];
        }
    }
    return attacks;
}

Bitboard rook_mask(Square s) {
    Bitboard edges = ((RANK_1_BB | RANK_8_BB) & ~rank_bb(rank_of(s)))
                   | ((FILE_A_BB | FILE_H_BB) & ~file_bb(file_of(s)));
    constexpr int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};
    return sliding_attacks_slow(s, 0, dirs) & ~edges;
}

Bitboard bishop_mask(Square s) {
    Bitboard edges = RANK_1_BB | RANK_8_BB | FILE_A_BB | FILE_H_BB;
    constexpr int dirs[4][2] = {{1,1}, {1,-1}, {-1,1}, {-1,-1}};
    return sliding_attacks_slow(s, 0, dirs) & ~edges;
}

} // namespace

void init() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    constexpr int rookDirs[4][2]   = {{1,0}, {-1,0}, {0,1}, {0,-1}};
    constexpr int bishopDirs[4][2] = {{1,1}, {1,-1}, {-1,1}, {-1,-1}};
    constexpr int knightD[8][2]    = {{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};
    constexpr int kingD[8][2]      = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};

    for (int s = 0; s < 64; ++s) {
        int r = rank_of(Square(s)), f = file_of(Square(s));

        Bitboard kn = 0;
        for (auto& d : knightD) {
            int nr = r + d[0], nf = f + d[1];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                kn |= square_bb(make_square(nf, nr));
        }
        KnightAttacks[s] = kn;

        Bitboard ki = 0;
        for (auto& d : kingD) {
            int nr = r + d[0], nf = f + d[1];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                ki |= square_bb(make_square(nf, nr));
        }
        KingAttacks[s] = ki;

        Bitboard sb = square_bb(Square(s));
        PawnAttacks[WHITE][s] = shift_ne(sb) | shift_nw(sb);
        PawnAttacks[BLACK][s] = shift_se(sb) | shift_sw(sb);

        RookPseudoAttacks[s]   = sliding_attacks_slow(Square(s), 0, rookDirs);
        BishopPseudoAttacks[s] = sliding_attacks_slow(Square(s), 0, bishopDirs);
    }

    // Initialize PEXT tables using Carry-Rippler subset enumeration
    U32 rOff = 0, bOff = 0;
    for (int s = 0; s < 64; ++s) {
        RookMasks[s] = rook_mask(Square(s));
        RookOffsets[s] = rOff;
        Bitboard m = RookMasks[s];
        Bitboard sub = 0;
        do {
            U64 idx = _pext_u64(sub, m);
            RookTable[rOff + idx] = sliding_attacks_slow(Square(s), sub, rookDirs);
            sub = (sub - m) & m;
        } while (sub != 0);
        rOff += (1U << popcount(m));

        BishopMasks[s] = bishop_mask(Square(s));
        BishopOffsets[s] = bOff;
        m = BishopMasks[s];
        sub = 0;
        do {
            U64 idx = _pext_u64(sub, m);
            BishopTable[bOff + idx] = sliding_attacks_slow(Square(s), sub, bishopDirs);
            sub = (sub - m) & m;
        } while (sub != 0);
        bOff += (1U << popcount(m));
    }

    // BetweenBB and LineBB
    for (int s1 = 0; s1 < 64; ++s1) {
        for (int s2 = 0; s2 < 64; ++s2) {
            BetweenBB[s1][s2] = 0;
            LineBB[s1][s2] = 0;
            if (s1 == s2) continue;

            Square sq1 = Square(s1), sq2 = Square(s2);
            if (RookPseudoAttacks[s1] & square_bb(sq2)) {
                LineBB[s1][s2] = (RookPseudoAttacks[s1] & RookPseudoAttacks[s2])
                               | square_bb(sq1) | square_bb(sq2);
                BetweenBB[s1][s2] = rook_attacks(sq1, square_bb(sq2))
                                  & rook_attacks(sq2, square_bb(sq1));
            } else if (BishopPseudoAttacks[s1] & square_bb(sq2)) {
                LineBB[s1][s2] = (BishopPseudoAttacks[s1] & BishopPseudoAttacks[s2])
                               | square_bb(sq1) | square_bb(sq2);
                BetweenBB[s1][s2] = bishop_attacks(sq1, square_bb(sq2))
                                  & bishop_attacks(sq2, square_bb(sq1));
            }
        }
    }
}

} // namespace Bitboards
