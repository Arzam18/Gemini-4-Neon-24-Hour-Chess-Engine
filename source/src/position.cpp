#include "position.h"
#include "movegen.h"
#include <sstream>
#include <cstring>

namespace {

constexpr U8 CastleMask[64] = {
    13, 15, 15, 15, 12, 15, 15, 14,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
     7, 15, 15, 15,  3, 15, 15, 11
};

inline bool is_minor(PieceType pt) { return pt == KNIGHT || pt == BISHOP || pt == KING; }
inline bool is_major(PieceType pt) { return pt == ROOK || pt == QUEEN || pt == KING; }

} // namespace

void Position::clear() {
    std::memset(byTypeBB, 0, sizeof(byTypeBB));
    std::memset(byColorBB, 0, sizeof(byColorBB));
    for (int s = 0; s < SQ_NB; ++s) board[s] = NO_PIECE;
    sideToMove = WHITE;
    gamePly = 0;
    st = nullptr;
}

inline void Position::put_piece(Piece p, Square s) {
    Bitboard b = square_bb(s);
    board[s] = p;
    byTypeBB[type_of(p)] |= b;
    byColorBB[color_of(p)] |= b;
}

inline void Position::remove_piece(Square s) {
    Piece p = board[s];
    Bitboard b = square_bb(s);
    board[s] = NO_PIECE;
    byTypeBB[type_of(p)] ^= b;
    byColorBB[color_of(p)] ^= b;
}

inline void Position::move_piece(Square from, Square to) {
    Piece p = board[from];
    Bitboard fromTo = square_bb(from) | square_bb(to);
    board[from] = NO_PIECE;
    board[to] = p;
    byTypeBB[type_of(p)] ^= fromTo;
    byColorBB[color_of(p)] ^= fromTo;
}

void Position::set_fen(const std::string& fen, StateInfo* si) {
    clear();
    *si = StateInfo{};
    st = si;

    std::istringstream ss(fen);
    std::string boardStr, sideStr, castleStr, epStr;
    ss >> boardStr >> sideStr >> castleStr >> epStr;

    int rank = 7, file = 0;
    for (char ch : boardStr) {
        if (ch == '/') {
            rank--;
            file = 0;
        } else if (ch >= '1' && ch <= '8') {
            file += (ch - '0');
        } else {
            Piece p = NO_PIECE;
            switch (ch) {
                case 'P': p = W_PAWN;   break;
                case 'N': p = W_KNIGHT; break;
                case 'B': p = W_BISHOP; break;
                case 'R': p = W_ROOK;   break;
                case 'Q': p = W_QUEEN;  break;
                case 'K': p = W_KING;   break;
                case 'p': p = B_PAWN;   break;
                case 'n': p = B_KNIGHT; break;
                case 'b': p = B_BISHOP; break;
                case 'r': p = B_ROOK;   break;
                case 'q': p = B_QUEEN;  break;
                case 'k': p = B_KING;   break;
                default: break;
            }
            if (p != NO_PIECE && rank >= 0 && rank < 8 && file >= 0 && file < 8) {
                put_piece(p, make_square(file, rank));
                file++;
            }
        }
    }

    sideToMove = (sideStr == "b") ? BLACK : WHITE;

    U8 cr = NO_CASTLING;
    for (char ch : castleStr) {
        if (ch == 'K') cr |= WHITE_OO;
        else if (ch == 'Q') cr |= WHITE_OOO;
        else if (ch == 'k') cr |= BLACK_OO;
        else if (ch == 'q') cr |= BLACK_OOO;
    }
    st->castlingRights = cr;

    if (epStr.size() == 2 && epStr[0] >= 'a' && epStr[0] <= 'h' &&
        (epStr[1] == '3' || epStr[1] == '6')) {
        Square ep = make_square(epStr[0] - 'a', epStr[1] - '1');
        // Keep epSquare only if an enemy pawn can actually attack it (standard FEN normalization)
        if (Bitboards::pawn_attacks(~sideToMove, ep) & pieces(sideToMove, PAWN))
            st->epSquare = ep;
        else
            st->epSquare = SQ_NONE;
    } else {
        st->epSquare = SQ_NONE;
    }

    int halfmove = 0, fullmove = 1;
    if (ss >> halfmove) st->rule50 = halfmove;
    if (ss >> fullmove) gamePly = std::max(0, (fullmove - 1) * 2 + (sideToMove == BLACK ? 1 : 0));

    st->pliesFromNull = 0;
    st->captured = NO_PIECE;
    st->prev = nullptr;

    set_state(st);
}

std::string Position::to_fen() const {
    std::ostringstream ss;
    constexpr char pieceChar[12] = {'P','N','B','R','Q','K','p','n','b','r','q','k'};
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board[make_square(f, r)];
            if (p == NO_PIECE) {
                empty++;
            } else {
                if (empty) { ss << empty; empty = 0; }
                ss << pieceChar[p];
            }
        }
        if (empty) ss << empty;
        if (r > 0) ss << '/';
    }
    ss << (sideToMove == WHITE ? " w " : " b ");
    if (!st->castlingRights) ss << '-';
    else {
        if (st->castlingRights & WHITE_OO)  ss << 'K';
        if (st->castlingRights & WHITE_OOO) ss << 'Q';
        if (st->castlingRights & BLACK_OO)  ss << 'k';
        if (st->castlingRights & BLACK_OOO) ss << 'q';
    }
    ss << ' ';
    if (st->epSquare == SQ_NONE) ss << '-';
    else ss << char('a' + file_of(st->epSquare)) << char('1' + rank_of(st->epSquare));
    ss << ' ' << st->rule50 << ' ' << (gamePly / 2 + 1);
    return ss.str();
}

void Position::set_state(StateInfo* si) const {
    si->key = 0;
    si->pawnKey = 0;
    si->nonPawnKey[WHITE] = 0;
    si->nonPawnKey[BLACK] = 0;
    si->minorKey = 0;
    si->majorKey = 0;

    for (int s = 0; s < 64; ++s) {
        Piece p = board[s];
        if (p != NO_PIECE) {
            U64 k = Zobrist::PieceKeys[p][s];
            si->key ^= k;
            PieceType pt = type_of(p);
            if (pt == PAWN) {
                si->pawnKey ^= k;
            } else {
                si->nonPawnKey[color_of(p)] ^= k;
                if (is_minor(pt)) si->minorKey ^= k;
                if (is_major(pt)) si->majorKey ^= k;
            }
        }
    }
    si->key ^= Zobrist::CastleKeys[si->castlingRights];
    if (si->epSquare != SQ_NONE)
        si->key ^= Zobrist::EpKeys[file_of(si->epSquare)];
    if (sideToMove == BLACK)
        si->key ^= Zobrist::SideKey;

    set_check_info(si);
}

Bitboard Position::slider_blockers(Bitboard sliders, Square s) const {
    Bitboard blockers = 0;

    Bitboard snipers = ((Bitboards::RookPseudoAttacks[s] & pieces(ROOK, QUEEN))
                      | (Bitboards::BishopPseudoAttacks[s] & pieces(BISHOP, QUEEN))) & sliders;
    Bitboard occ = pieces() ^ snipers;

    while (snipers) {
        Square sniperSq = pop_lsb(snipers);
        Bitboard b = Bitboards::between_bb(s, sniperSq) & occ;
        if (b && !more_than_one(b)) {
            blockers |= b;
        }
    }
    return blockers;
}

void Position::set_check_info(StateInfo* si) const {
    Color us = sideToMove;
    Square ourKing = king_square(us);
    si->checkersBB = attackers_to(ourKing) & pieces(~us);
    si->pinReady = 0;
}

void Position::ensure_pin_info(Color c) const {
    U8 bit = U8(1u << int(c));
    if (!(st->pinReady & bit)) {
        st->blockersForKing[c] = slider_blockers(pieces(~c), king_square(c));
        st->pinReady |= bit;
    }
}

void Position::ensure_check_squares() const {
    if (!(st->pinReady & 4u)) {
        Color them = ~sideToMove;
        Square theirKing = king_square(them);
        Bitboard occ = pieces();
        st->checkSquares[PAWN]   = Bitboards::pawn_attacks(them, theirKing);
        st->checkSquares[KNIGHT] = Bitboards::knight_attacks(theirKing);
        st->checkSquares[BISHOP] = Bitboards::bishop_attacks(theirKing, occ);
        st->checkSquares[ROOK]   = Bitboards::rook_attacks(theirKing, occ);
        st->checkSquares[QUEEN]  = st->checkSquares[BISHOP] | st->checkSquares[ROOK];
        st->checkSquares[KING]   = 0;
        st->pinReady |= 4u;
    }
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    return (Bitboards::pawn_attacks(BLACK, s) & pieces(WHITE, PAWN))
         | (Bitboards::pawn_attacks(WHITE, s) & pieces(BLACK, PAWN))
         | (Bitboards::knight_attacks(s) & pieces(KNIGHT))
         | (Bitboards::king_attacks(s) & pieces(KING))
         | (Bitboards::rook_attacks(s, occ) & pieces(ROOK, QUEEN))
         | (Bitboards::bishop_attacks(s, occ) & pieces(BISHOP, QUEEN));
}

bool Position::is_attacked_by(Color c, Square s, Bitboard occ) const {
    if (Bitboards::pawn_attacks(~c, s) & pieces(c, PAWN)) return true;
    if (Bitboards::knight_attacks(s) & pieces(c, KNIGHT)) return true;
    if (Bitboards::king_attacks(s) & pieces(c, KING)) return true;
    if (Bitboards::bishop_attacks(s, occ) & pieces(c, BISHOP, QUEEN)) return true;
    if (Bitboards::rook_attacks(s, occ) & pieces(c, ROOK, QUEEN)) return true;
    return false;
}

bool Position::is_legal(Move m) const {
    Color us = sideToMove;
    Color them = ~us;
    Square from = from_sq(m);
    Square to = to_sq(m);
    Square ksq = king_square(us);
    MoveFlag flag = move_flag(m);

    if (flag == MF_EP_CAPTURE) {
        Square capSq = Square(to + (us == WHITE ? -8 : 8));
        Bitboard occ = (pieces() ^ square_bb(from) ^ square_bb(capSq)) | square_bb(to);
        Bitboard enemyPawns = pieces(them, PAWN) ^ square_bb(capSq);
        return !(Bitboards::pawn_attacks(us, ksq) & enemyPawns)
            && !(Bitboards::knight_attacks(ksq) & pieces(them, KNIGHT))
            && !(Bitboards::rook_attacks(ksq, occ) & pieces(them, ROOK, QUEEN))
            && !(Bitboards::bishop_attacks(ksq, occ) & pieces(them, BISHOP, QUEEN));
    }

    if (flag == MF_CASTLE_KS || flag == MF_CASTLE_QS) {
        // Movegen already verifies castling path isn't attacked; verify here too for safety
        if (st->checkersBB) return false;
        Square mid = Square((from + to) / 2);
        return !is_attacked_by(them, mid) && !is_attacked_by(them, to);
    }

    if (from == ksq) {
        return !is_attacked_by(them, to, pieces() ^ square_bb(from));
    }

    // Non-king, non-ep move:
    // If in check, double check cannot be resolved by non-king move; single check must capture or block
    if (st->checkersBB) {
        if (more_than_one(st->checkersBB)) return false;
        Square checkerSq = lsb(st->checkersBB);
        if (!(square_bb(to) & (Bitboards::between_bb(ksq, checkerSq) | st->checkersBB)))
            return false;
    }

    ensure_pin_info(us);
    return !(st->blockersForKing[us] & square_bb(from))
        || Bitboards::aligned(from, to, ksq);
}

bool Position::is_pseudo_legal(Move m) const {
    if (m == MOVE_NONE) return false;
    Color us = sideToMove;
    Color them = ~us;
    Square from = from_sq(m);
    Square to = to_sq(m);
    Piece pc = board[from];
    if (pc == NO_PIECE || color_of(pc) != us) return false;

    PieceType pt = type_of(pc);
    MoveFlag flag = move_flag(m);

    // Destination cannot hold our own piece
    if (pieces(us) & square_bb(to)) return false;

    // Handle special flags first
    if (flag == MF_CASTLE_KS) {
        if (pt != KING || st->checkersBB) return false;
        if (us == WHITE) {
            return (st->castlingRights & WHITE_OO) && from == SQ_E1 && to == SQ_G1
                && !(pieces() & (square_bb(SQ_F1) | square_bb(SQ_G1)))
                && !is_attacked_by(them, SQ_F1) && !is_attacked_by(them, SQ_G1);
        } else {
            return (st->castlingRights & BLACK_OO) && from == SQ_E8 && to == SQ_G8
                && !(pieces() & (square_bb(SQ_F8) | square_bb(SQ_G8)))
                && !is_attacked_by(them, SQ_F8) && !is_attacked_by(them, SQ_G8);
        }
    }
    if (flag == MF_CASTLE_QS) {
        if (pt != KING || st->checkersBB) return false;
        if (us == WHITE) {
            return (st->castlingRights & WHITE_OOO) && from == SQ_E1 && to == SQ_C1
                && !(pieces() & (square_bb(SQ_B1) | square_bb(SQ_C1) | square_bb(SQ_D1)))
                && !is_attacked_by(them, SQ_D1) && !is_attacked_by(them, SQ_C1);
        } else {
            return (st->castlingRights & BLACK_OOO) && from == SQ_E8 && to == SQ_C8
                && !(pieces() & (square_bb(SQ_B8) | square_bb(SQ_C8) | square_bb(SQ_D8)))
                && !is_attacked_by(them, SQ_D8) && !is_attacked_by(them, SQ_C8);
        }
    }
    if (flag == MF_EP_CAPTURE) {
        return pt == PAWN && to == st->epSquare
            && (Bitboards::pawn_attacks(us, from) & square_bb(to));
    }

    bool cap = is_capture(m);
    bool targetOcc = (pieces(them) & square_bb(to)) != 0;
    if (cap != targetOcc) return false;

    if (pt == PAWN) {
        int rFrom = rank_of(from);
        int rTo = rank_of(to);
        bool promoRank = (us == WHITE ? rTo == 7 : rTo == 0);
        if (is_promotion(m) != promoRank) return false;

        if (flag == MF_DOUBLE_PUSH) {
            int expectedFrom = (us == WHITE ? 1 : 6);
            int push = (us == WHITE ? 8 : -8);
            return rFrom == expectedFrom && to == from + 2 * push && board[from + push] == NO_PIECE;
        }
        if (cap) {
            return (Bitboards::pawn_attacks(us, from) & square_bb(to)) != 0;
        } else {
            int push = (us == WHITE ? 8 : -8);
            return to == from + push;
        }
    } else {
        if (is_promotion(m) || flag == MF_DOUBLE_PUSH) return false;
        return (Bitboards::attacks_bb(pt, from, pieces()) & square_bb(to)) != 0;
    }
}

bool Position::gives_check(Move m) const {
    ensure_check_squares();
    Square from = from_sq(m);
    Square to = to_sq(m);
    PieceType pt = type_of(board[from]);
    MoveFlag flag = move_flag(m);

    // Direct check for normal moves
    if (!is_promotion(m) && (st->checkSquares[pt] & square_bb(to)))
        return true;

    // Discovered check
    Color them = ~sideToMove;
    Square ksq = king_square(them);
    ensure_pin_info(them);
    if ((st->blockersForKing[them] & square_bb(from)) && !Bitboards::aligned(from, to, ksq))
        return true;

    // Special moves
    if (is_promotion(m)) {
        PieceType promoPt = promo_type(m);
        return (Bitboards::attacks_bb(promoPt, to, pieces() ^ square_bb(from)) & square_bb(ksq)) != 0;
    }
    if (flag == MF_EP_CAPTURE) {
        Square capSq = Square(to + (sideToMove == WHITE ? -8 : 8));
        Bitboard occ = (pieces() ^ square_bb(from) ^ square_bb(capSq)) | square_bb(to);
        return (Bitboards::rook_attacks(ksq, occ) & pieces(sideToMove, ROOK, QUEEN))
            || (Bitboards::bishop_attacks(ksq, occ) & pieces(sideToMove, BISHOP, QUEEN));
    }
    if (flag == MF_CASTLE_KS || flag == MF_CASTLE_QS) {
        Square rTo = Square((from + to) / 2);
        Bitboard occ = (pieces() ^ square_bb(from)) | square_bb(to) | square_bb(rTo);
        return (Bitboards::rook_attacks(rTo, occ) & square_bb(ksq)) != 0;
    }
    return false;
}

bool Position::see_ge(Move m, int threshold) const {
    MoveFlag flag = move_flag(m);
    if (flag == MF_CASTLE_KS || flag == MF_CASTLE_QS || flag == MF_EP_CAPTURE)
        return 0 >= threshold;

    Square from = from_sq(m);
    Square to = to_sq(m);

    int nextVictim = is_promotion(m) ? promo_type(m) : type_of(board[from]);
    int balance = (board[to] != NO_PIECE ? SeeValues[type_of(board[to])] : 0) - threshold;
    if (is_promotion(m))
        balance += SeeValues[promo_type(m)] - SeeValues[PAWN];

    if (balance < 0) return false;

    balance -= SeeValues[nextVictim];
    if (balance >= 0) return true;

    Bitboard occ = (pieces() ^ square_bb(from)) | square_bb(to);
    Bitboard attackers = attackers_to(to, occ) & occ;
    Bitboard bishops = pieces(BISHOP, QUEEN);
    Bitboard rooks = pieces(ROOK, QUEEN);

    Color stm = ~sideToMove;
    while (true) {
        Bitboard stmAttackers = attackers & pieces(stm);
        if (!stmAttackers) break;

        // Find least valuable attacker
        int pt;
        Bitboard bb = 0;
        for (pt = PAWN; pt <= KING; ++pt) {
            bb = stmAttackers & pieces(PieceType(pt));
            if (bb) break;
        }

        stm = ~stm;
        balance = -balance - 1 - SeeValues[pt];
        if (balance >= 0) {
            // If king captures and opponent still has attackers, king capture is illegal
            if (pt == KING && (attackers & pieces(stm)))
                stm = ~stm;
            break;
        }

        // Remove attacker and reveal x-ray sliders
        occ ^= (bb & -bb);
        if (pt == PAWN || pt == BISHOP || pt == QUEEN)
            attackers |= Bitboards::bishop_attacks(to, occ) & bishops;
        if (pt == ROOK || pt == QUEEN)
            attackers |= Bitboards::rook_attacks(to, occ) & rooks;
        attackers &= occ;
    }

    return stm != sideToMove;
}

void Position::do_move(Move m, StateInfo& newSt) {
    newSt.prev = st;
    newSt.key = st->key;
    newSt.pawnKey = st->pawnKey;
    newSt.nonPawnKey[WHITE] = st->nonPawnKey[WHITE];
    newSt.nonPawnKey[BLACK] = st->nonPawnKey[BLACK];
    newSt.minorKey = st->minorKey;
    newSt.majorKey = st->majorKey;
    newSt.castlingRights = st->castlingRights;
    newSt.epSquare = SQ_NONE;
    newSt.rule50 = st->rule50 + 1;
    newSt.pliesFromNull = st->pliesFromNull + 1;
    newSt.captured = NO_PIECE;

    if (st->epSquare != SQ_NONE)
        newSt.key ^= Zobrist::EpKeys[file_of(st->epSquare)];

    Color us = sideToMove;
    Color them = ~us;
    Square from = from_sq(m);
    Square to = to_sq(m);
    MoveFlag flag = move_flag(m);
    Piece pc = board[from];
    PieceType pt = type_of(pc);

    // Handle capture
    if (flag == MF_EP_CAPTURE) {
        Square capSq = Square(to + (us == WHITE ? -8 : 8));
        Piece capPc = make_piece(them, PAWN);
        newSt.captured = capPc;
        remove_piece(capSq);
        U64 k = Zobrist::PieceKeys[capPc][capSq];
        newSt.key ^= k;
        newSt.pawnKey ^= k;
        newSt.rule50 = 0;
    } else if (is_capture(m)) {
        Piece capPc = board[to];
        PieceType capPt = type_of(capPc);
        newSt.captured = capPc;
        remove_piece(to);
        U64 k = Zobrist::PieceKeys[capPc][to];
        newSt.key ^= k;
        if (capPt == PAWN) {
            newSt.pawnKey ^= k;
        } else {
            newSt.nonPawnKey[them] ^= k;
            if (is_minor(capPt)) newSt.minorKey ^= k;
            if (is_major(capPt)) newSt.majorKey ^= k;
        }
        newSt.rule50 = 0;
    }

    // Move piece
    U64 kFrom = Zobrist::PieceKeys[pc][from];
    if (is_promotion(m)) {
        Piece promoPc = make_piece(us, promo_type(m));
        PieceType promoPt = promo_type(m);
        remove_piece(from);
        put_piece(promoPc, to);
        U64 kTo = Zobrist::PieceKeys[promoPc][to];
        newSt.key ^= kFrom ^ kTo;
        newSt.pawnKey ^= kFrom;
        newSt.nonPawnKey[us] ^= kTo;
        if (is_minor(promoPt)) newSt.minorKey ^= kTo;
        if (is_major(promoPt)) newSt.majorKey ^= kTo;
        newSt.rule50 = 0;
    } else {
        move_piece(from, to);
        U64 kTo = Zobrist::PieceKeys[pc][to];
        U64 kDelta = kFrom ^ kTo;
        newSt.key ^= kDelta;
        if (pt == PAWN) {
            newSt.pawnKey ^= kDelta;
            newSt.rule50 = 0;
            if (flag == MF_DOUBLE_PUSH) {
                Square ep = Square((from + to) / 2);
                if (Bitboards::pawn_attacks(us, ep) & pieces(them, PAWN)) {
                    newSt.epSquare = ep;
                    newSt.key ^= Zobrist::EpKeys[file_of(ep)];
                }
            }
        } else {
            newSt.nonPawnKey[us] ^= kDelta;
            if (is_minor(pt)) newSt.minorKey ^= kDelta;
            if (is_major(pt)) newSt.majorKey ^= kDelta;

            if (flag == MF_CASTLE_KS) {
                Square rFrom = (us == WHITE ? SQ_H1 : SQ_H8);
                Square rTo   = (us == WHITE ? SQ_F1 : SQ_F8);
                Piece rook = make_piece(us, ROOK);
                move_piece(rFrom, rTo);
                U64 rDelta = Zobrist::PieceKeys[rook][rFrom] ^ Zobrist::PieceKeys[rook][rTo];
                newSt.key ^= rDelta;
                newSt.nonPawnKey[us] ^= rDelta;
                newSt.majorKey ^= rDelta;
            } else if (flag == MF_CASTLE_QS) {
                Square rFrom = (us == WHITE ? SQ_A1 : SQ_A8);
                Square rTo   = (us == WHITE ? SQ_D1 : SQ_D8);
                Piece rook = make_piece(us, ROOK);
                move_piece(rFrom, rTo);
                U64 rDelta = Zobrist::PieceKeys[rook][rFrom] ^ Zobrist::PieceKeys[rook][rTo];
                newSt.key ^= rDelta;
                newSt.nonPawnKey[us] ^= rDelta;
                newSt.majorKey ^= rDelta;
            }
        }
    }

    // Update castling rights
    U8 newCR = newSt.castlingRights & CastleMask[from] & CastleMask[to];
    if (newCR != newSt.castlingRights) {
        newSt.key ^= Zobrist::CastleKeys[newSt.castlingRights] ^ Zobrist::CastleKeys[newCR];
        newSt.castlingRights = newCR;
    }

    newSt.key ^= Zobrist::SideKey;
    sideToMove = them;
    gamePly++;
    st = &newSt;

    set_check_info(st);
}

void Position::undo_move(Move m) {
    sideToMove = ~sideToMove;
    gamePly--;

    Color us = sideToMove;
    Square from = from_sq(m);
    Square to = to_sq(m);
    MoveFlag flag = move_flag(m);

    if (is_promotion(m)) {
        remove_piece(to);
        put_piece(make_piece(us, PAWN), from);
    } else {
        move_piece(to, from);
        if (flag == MF_CASTLE_KS) {
            Square rFrom = (us == WHITE ? SQ_H1 : SQ_H8);
            Square rTo   = (us == WHITE ? SQ_F1 : SQ_F8);
            move_piece(rTo, rFrom);
        } else if (flag == MF_CASTLE_QS) {
            Square rFrom = (us == WHITE ? SQ_A1 : SQ_A8);
            Square rTo   = (us == WHITE ? SQ_D1 : SQ_D8);
            move_piece(rTo, rFrom);
        }
    }

    if (st->captured != NO_PIECE) {
        Square capSq = (flag == MF_EP_CAPTURE) ? Square(to + (us == WHITE ? -8 : 8)) : to;
        put_piece(st->captured, capSq);
    }

    st = st->prev;
}

void Position::do_null_move(StateInfo& newSt) {
    newSt.prev = st;
    newSt.key = st->key ^ Zobrist::SideKey;
    if (st->epSquare != SQ_NONE)
        newSt.key ^= Zobrist::EpKeys[file_of(st->epSquare)];
    newSt.pawnKey = st->pawnKey;
    newSt.nonPawnKey[WHITE] = st->nonPawnKey[WHITE];
    newSt.nonPawnKey[BLACK] = st->nonPawnKey[BLACK];
    newSt.minorKey = st->minorKey;
    newSt.majorKey = st->majorKey;
    newSt.castlingRights = st->castlingRights;
    newSt.epSquare = SQ_NONE;
    newSt.rule50 = st->rule50 + 1;
    newSt.pliesFromNull = 0;
    newSt.captured = NO_PIECE;

    sideToMove = ~sideToMove;
    gamePly++;
    st = &newSt;

    newSt.checkersBB = 0;
    newSt.pinReady = 0;
}

void Position::undo_null_move() {
    sideToMove = ~sideToMove;
    gamePly--;
    st = st->prev;
}

bool Position::is_repetition(int searchPly) const {
    int end = std::min(st->rule50, st->pliesFromNull);
    if (end < 4) return false;

    StateInfo* prevSt = st->prev;
    if (!prevSt || !prevSt->prev) return false;
    prevSt = prevSt->prev;
    int reps = 0;
    for (int i = 4; i <= end; i += 2) {
        if (!prevSt->prev || !prevSt->prev->prev) break;
        prevSt = prevSt->prev->prev;
        if (prevSt->key == st->key) {
            if (i < searchPly) return true;
            if (++reps >= 2) return true;
        }
    }
    return false;
}

bool Position::is_insufficient_material() const {
    if (pieces(PAWN) | pieces(ROOK) | pieces(QUEEN)) return false;
    int total = popcount(pieces());
    if (total <= 3) return true; // K vs K, KN vs K, KB vs K
    if (total == 4 && popcount(pieces(WHITE, BISHOP)) == 1 && popcount(pieces(BLACK, BISHOP)) == 1) {
        Square wb = lsb(pieces(WHITE, BISHOP));
        Square bb = lsb(pieces(BLACK, BISHOP));
        bool wLight = ((rank_of(wb) + file_of(wb)) & 1) != 0;
        bool bLight = ((rank_of(bb) + file_of(bb)) & 1) != 0;
        if (wLight == bLight) return true; // KB vs KB same color
    }
    return false;
}

bool Position::is_draw(int searchPly) const {
    if (st->rule50 >= 100) {
        if (!in_check()) return true;
        MoveList ml;
        MoveGen::generate_legal(*this, ml);
        return ml.size() > 0; // Checkmate takes precedence over 50-move draw
    }
    if (is_insufficient_material()) return true;
    return is_repetition(searchPly);
}

Move Position::parse_uci_move(const std::string& str) const {
    MoveList ml;
    MoveGen::generate_legal(*this, ml);
    for (const auto& em : ml) {
        if (move_to_uci(em.move) == str)
            return em.move;
    }
    return MOVE_NONE;
}
