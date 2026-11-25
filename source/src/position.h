#pragma once

#include "types.h"
#include "bitboard.h"
#include "zobrist.h"
#include <string>

struct StateInfo {
    U64 key = 0;
    U64 pawnKey = 0;
    U64 nonPawnKey[2] = {0, 0};
    U64 minorKey = 0;
    U64 majorKey = 0;

    Bitboard checkersBB = 0;
    mutable Bitboard blockersForKing[2] = {0, 0};
    mutable Bitboard checkSquares[PIECE_TYPE_NB] = {};
    mutable U8 pinReady = 0;

    U8 castlingRights = 0;
    Square epSquare = SQ_NONE;
    int rule50 = 0;
    int pliesFromNull = 0;
    Piece captured = NO_PIECE;

    StateInfo* prev = nullptr;
};

class Position {
public:
    Position() { clear(); }

    void clear();
    void set_fen(const std::string& fen, StateInfo* si);
    std::string to_fen() const;

    // Accessors
    Bitboard pieces() const { return byColorBB[WHITE] | byColorBB[BLACK]; }
    Bitboard pieces(Color c) const { return byColorBB[c]; }
    Bitboard pieces(PieceType pt) const { return byTypeBB[pt]; }
    Bitboard pieces(PieceType pt1, PieceType pt2) const { return byTypeBB[pt1] | byTypeBB[pt2]; }
    Bitboard pieces(Color c, PieceType pt) const { return byColorBB[c] & byTypeBB[pt]; }
    Bitboard pieces(Color c, PieceType pt1, PieceType pt2) const {
        return byColorBB[c] & (byTypeBB[pt1] | byTypeBB[pt2]);
    }

    Piece piece_on(Square s) const { return board[s]; }
    Color side_to_move() const { return sideToMove; }
    Square king_square(Color c) const { return lsb(pieces(c, KING)); }
    U8 castling_rights() const { return st->castlingRights; }
    Square ep_square() const { return st->epSquare; }
    int rule50_count() const { return st->rule50; }
    int game_ply() const { return gamePly; }

    U64 key() const { return st->key; }
    U64 pawn_key() const { return st->pawnKey; }
    U64 non_pawn_key(Color c) const { return st->nonPawnKey[c]; }
    U64 minor_key() const { return st->minorKey; }
    U64 major_key() const { return st->majorKey; }

    Bitboard checkers() const { return st->checkersBB; }
    bool in_check() const { return st->checkersBB != 0; }
    Bitboard blockers_for_king(Color c) const { ensure_pin_info(c); return st->blockersForKing[c]; }
    Bitboard check_squares(PieceType pt) const { ensure_check_squares(); return st->checkSquares[pt]; }

    StateInfo* state() const { return st; }

    // Attack & legality helpers
    Bitboard attackers_to(Square s, Bitboard occ) const;
    Bitboard attackers_to(Square s) const { return attackers_to(s, pieces()); }
    bool is_attacked_by(Color c, Square s, Bitboard occ) const;
    bool is_attacked_by(Color c, Square s) const { return is_attacked_by(c, s, pieces()); }

    bool is_legal(Move m) const;
    bool is_pseudo_legal(Move m) const;
    bool gives_check(Move m) const;
    bool see_ge(Move m, int threshold = 0) const;

    bool has_non_pawn_material(Color c) const {
        return (pieces(c) ^ pieces(c, PAWN) ^ pieces(c, KING)) != 0;
    }

    bool is_repetition(int searchPly) const;
    bool is_upcoming_repetition() const { return false; }
    bool is_insufficient_material() const;
    bool is_draw(int searchPly) const;

    // Make / Unmake
    void do_move(Move m, StateInfo& newSt);
    void undo_move(Move m);
    void do_null_move(StateInfo& newSt);
    void undo_null_move();

    Move parse_uci_move(const std::string& str) const;

    static constexpr int SeeValues[PIECE_TYPE_NB + 1] = {
        100, 320, 330, 500, 950, 20000, 0
    };

private:
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    void set_state(StateInfo* si) const;
    void set_check_info(StateInfo* si) const;
    void ensure_pin_info(Color c) const;
    void ensure_check_squares() const;
    Bitboard slider_blockers(Bitboard sliders, Square s) const;

    Bitboard byTypeBB[PIECE_TYPE_NB];
    Bitboard byColorBB[COLOR_NB];
    Piece board[SQ_NB];
    Color sideToMove;
    int gamePly;
    StateInfo* st;
};
