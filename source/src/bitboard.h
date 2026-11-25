#pragma once

#include "types.h"
#include <immintrin.h>

namespace Bitboards {

extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];
extern Bitboard PawnAttacks[2][64];
extern Bitboard BetweenBB[64][64];
extern Bitboard LineBB[64][64];
extern Bitboard RookPseudoAttacks[64];
extern Bitboard BishopPseudoAttacks[64];

extern Bitboard RookMasks[64];
extern Bitboard BishopMasks[64];
extern U32 RookOffsets[64];
extern U32 BishopOffsets[64];
extern Bitboard RookTable[102400];
extern Bitboard BishopTable[5248];

void init();

inline Bitboard knight_attacks(Square s) { return KnightAttacks[s]; }
inline Bitboard king_attacks(Square s)   { return KingAttacks[s]; }
inline Bitboard pawn_attacks(Color c, Square s) { return PawnAttacks[c][s]; }

inline Bitboard rook_attacks(Square s, Bitboard occ) {
    return RookTable[RookOffsets[s] + _pext_u64(occ, RookMasks[s])];
}

inline Bitboard bishop_attacks(Square s, Bitboard occ) {
    return BishopTable[BishopOffsets[s] + _pext_u64(occ, BishopMasks[s])];
}

inline Bitboard queen_attacks(Square s, Bitboard occ) {
    return rook_attacks(s, occ) | bishop_attacks(s, occ);
}

inline Bitboard between_bb(Square s1, Square s2) { return BetweenBB[s1][s2]; }
inline Bitboard line_bb(Square s1, Square s2)    { return LineBB[s1][s2]; }

inline bool aligned(Square s1, Square s2, Square s3) {
    return (LineBB[s1][s2] & square_bb(s3)) != 0;
}

inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occ) {
    switch (pt) {
        case KNIGHT: return knight_attacks(s);
        case BISHOP: return bishop_attacks(s, occ);
        case ROOK:   return rook_attacks(s, occ);
        case QUEEN:  return queen_attacks(s, occ);
        case KING:   return king_attacks(s);
        default:     return 0;
    }
}

} // namespace Bitboards
