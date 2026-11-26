#pragma once

#include <cstdint>
#include <cassert>
#include <string>
#include <algorithm>

using U64 = uint64_t;
using U32 = uint32_t;
using U16 = uint16_t;
using U8  = uint8_t;
using Bitboard = uint64_t;

constexpr int MAX_PLY   = 128;
constexpr int MAX_MOVES = 256;

constexpr int SCORE_ZERO      = 0;
constexpr int SCORE_DRAW      = 0;
constexpr int SCORE_NONE      = 32002;
constexpr int SCORE_INFINITE  = 32001;
constexpr int SCORE_MATE      = 31000;
constexpr int SCORE_MATE_IN_MAX  = SCORE_MATE - MAX_PLY;
constexpr int SCORE_MATED_IN_MAX = -SCORE_MATE + MAX_PLY;
constexpr int SCORE_TB_WIN    = 29000;

enum Color : int {
    WHITE = 0,
    BLACK = 1,
    COLOR_NB = 2
};

constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    PIECE_TYPE_NB = 6,
    NO_PIECE_TYPE = 6
};

enum Piece : int {
    W_PAWN = 0, W_KNIGHT = 1, W_BISHOP = 2, W_ROOK = 3, W_QUEEN = 4, W_KING = 5,
    B_PAWN = 6, B_KNIGHT = 7, B_BISHOP = 8, B_ROOK = 9, B_QUEEN = 10, B_KING = 11,
    PIECE_NB = 12,
    NO_PIECE = 12
};

constexpr Piece make_piece(Color c, PieceType pt) {
    return Piece(c * 6 + pt);
}

constexpr Color color_of(Piece p) {
    return Color(p >= 6);
}

constexpr PieceType type_of(Piece p) {
    return p == NO_PIECE ? NO_PIECE_TYPE : PieceType(p % 6);
}

enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE = 64,
    SQ_NB = 64
};

constexpr int file_of(Square s) { return s & 7; }
constexpr int rank_of(Square s) { return s >> 3; }
constexpr Square make_square(int f, int r) { return Square((r << 3) | f); }
constexpr Square flip_rank(Square s) { return Square(s ^ 56); }
constexpr Square flip_file(Square s) { return Square(s ^ 7); }
constexpr Square relative_square(Color c, Square s) { return Square(s ^ (c * 56)); }
constexpr int relative_rank(Color c, int r) { return r ^ (c * 7); }
constexpr int relative_rank(Color c, Square s) { return relative_rank(c, rank_of(s)); }
constexpr Bitboard square_bb(Square s) { return 1ULL << s; }

enum CastlingRights : U8 {
    NO_CASTLING = 0,
    WHITE_OO    = 1,
    WHITE_OOO   = 2,
    BLACK_OO    = 4,
    BLACK_OOO   = 8,
    WHITE_CASTLING = WHITE_OO | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING
};

// Move encoding (16 bits):
// bits 0..5:   from square
// bits 6..11:  to square
// bits 12..15: move flag
enum MoveFlag : U16 {
    MF_QUIET        = 0,
    MF_DOUBLE_PUSH  = 1,
    MF_CASTLE_KS    = 2,
    MF_CASTLE_QS    = 3,
    MF_CAPTURE      = 4,
    MF_EP_CAPTURE   = 5,
    MF_PROMO_N      = 8,
    MF_PROMO_B      = 9,
    MF_PROMO_R      = 10,
    MF_PROMO_Q      = 11,
    MF_PROMO_CAP_N  = 12,
    MF_PROMO_CAP_B  = 13,
    MF_PROMO_CAP_R  = 14,
    MF_PROMO_CAP_Q  = 15
};

using Move = U16;
constexpr Move MOVE_NONE = 0;

constexpr Move make_move(Square from, Square to, MoveFlag flag = MF_QUIET) {
    return Move(from | (to << 6) | (flag << 12));
}

constexpr Square from_sq(Move m) { return Square(m & 63); }
constexpr Square to_sq(Move m)   { return Square((m >> 6) & 63); }
constexpr MoveFlag move_flag(Move m) { return MoveFlag(m >> 12); }
constexpr bool is_capture(Move m)   { return (m & (4 << 12)) != 0; }
constexpr bool is_promotion(Move m) { return (m & (8 << 12)) != 0; }
constexpr bool is_tactical(Move m)  { return (m & (12 << 12)) != 0; } // capture or promotion
constexpr bool is_enpassant(Move m) { return move_flag(m) == MF_EP_CAPTURE; }
constexpr bool is_castle(Move m)    { MoveFlag f = move_flag(m); return f == MF_CASTLE_KS || f == MF_CASTLE_QS; }
constexpr PieceType promo_type(Move m) { return PieceType(((m >> 12) & 3) + KNIGHT); }

inline std::string move_to_uci(Move m) {
    if (m == MOVE_NONE) return "0000";
    Square f = from_sq(m);
    Square t = to_sq(m);
    std::string s;
    s.reserve(5);
    s.push_back(char('a' + file_of(f)));
    s.push_back(char('1' + rank_of(f)));
    s.push_back(char('a' + file_of(t)));
    s.push_back(char('1' + rank_of(t)));
    if (is_promotion(m)) {
        constexpr char promoChars[4] = {'n', 'b', 'r', 'q'};
        s.push_back(promoChars[(m >> 12) & 3]);
    }
    return s;
}

inline int popcount(Bitboard b) {
    return __builtin_popcountll(b);
}

inline Square lsb(Bitboard b) {
    assert(b != 0);
    return Square(__builtin_ctzll(b));
}

inline Square msb(Bitboard b) {
    assert(b != 0);
    return Square(63 ^ __builtin_clzll(b));
}

inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

constexpr bool more_than_one(Bitboard b) {
    return (b & (b - 1)) != 0;
}

constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_B_BB = FILE_A_BB << 1;
constexpr Bitboard FILE_C_BB = FILE_A_BB << 2;
constexpr Bitboard FILE_D_BB = FILE_A_BB << 3;
constexpr Bitboard FILE_E_BB = FILE_A_BB << 4;
constexpr Bitboard FILE_F_BB = FILE_A_BB << 5;
constexpr Bitboard FILE_G_BB = FILE_A_BB << 6;
constexpr Bitboard FILE_H_BB = FILE_A_BB << 7;

constexpr Bitboard RANK_1_BB = 0x00000000000000FFULL;
constexpr Bitboard RANK_2_BB = RANK_1_BB << 8;
constexpr Bitboard RANK_3_BB = RANK_1_BB << 16;
constexpr Bitboard RANK_4_BB = RANK_1_BB << 24;
constexpr Bitboard RANK_5_BB = RANK_1_BB << 32;
constexpr Bitboard RANK_6_BB = RANK_1_BB << 40;
constexpr Bitboard RANK_7_BB = RANK_1_BB << 48;
constexpr Bitboard RANK_8_BB = RANK_1_BB << 56;

constexpr Bitboard file_bb(int f) { return FILE_A_BB << f; }
constexpr Bitboard rank_bb(int r) { return RANK_1_BB << (8 * r); }

inline Bitboard shift_north(Bitboard b) { return b << 8; }
inline Bitboard shift_south(Bitboard b) { return b >> 8; }
inline Bitboard shift_east(Bitboard b)  { return (b & ~FILE_H_BB) << 1; }
inline Bitboard shift_west(Bitboard b)  { return (b & ~FILE_A_BB) >> 1; }
inline Bitboard shift_ne(Bitboard b)    { return (b & ~FILE_H_BB) << 9; }
inline Bitboard shift_nw(Bitboard b)    { return (b & ~FILE_A_BB) << 7; }
inline Bitboard shift_se(Bitboard b)    { return (b & ~FILE_H_BB) >> 7; }
inline Bitboard shift_sw(Bitboard b)    { return (b & ~FILE_A_BB) >> 9; }

template<Color C>
inline Bitboard pawn_push_bb(Bitboard b) {
    return C == WHITE ? (b << 8) : (b >> 8);
}

template<Color C>
inline Bitboard pawn_attacks_bb(Bitboard b) {
    return C == WHITE ? (shift_ne(b) | shift_nw(b)) : (shift_se(b) | shift_sw(b));
}

struct ExtMove {
    Move move;
    int score;
};

struct MoveList {
    ExtMove moves[MAX_MOVES];
    int count = 0;

    void push(Move m) { moves[count++].move = m; }
    int size() const { return count; }
    ExtMove* begin() { return moves; }
    ExtMove* end() { return moves + count; }
    const ExtMove* begin() const { return moves; }
    const ExtMove* end() const { return moves + count; }
};
