#include "movegen.h"

namespace MoveGen {

namespace {

enum GenType {
    GEN_CAPTURES,
    GEN_QUIETS,
    GEN_ALL
};

template<Color Us, GenType Type>
void generate_pawn_moves(const Position& pos, MoveList& list, Bitboard target) {
    constexpr Color Them = ~Us;
    constexpr Bitboard PromoRank  = (Us == WHITE ? RANK_7_BB : RANK_2_BB);
    constexpr Bitboard Rank3BB    = (Us == WHITE ? RANK_3_BB : RANK_6_BB);
    constexpr int PushDelta       = (Us == WHITE ? 8 : -8);
    constexpr int CapLeftDelta    = (Us == WHITE ? 7 : -9);
    constexpr int CapRightDelta   = (Us == WHITE ? 9 : -7);

    Bitboard pawns = pos.pieces(Us, PAWN);
    Bitboard pawnsOn7 = pawns & PromoRank;
    Bitboard pawnsNot7 = pawns & ~PromoRank;
    Bitboard empty = ~pos.pieces();
    Bitboard enemies = pos.pieces(Them);

    // Captures & Capture-Promotions
    if constexpr (Type == GEN_CAPTURES || Type == GEN_ALL) {
        // Non-promotion captures
        Bitboard capTargets = enemies & target;
        Bitboard capL = (Us == WHITE ? shift_nw(pawnsNot7) : shift_sw(pawnsNot7)) & capTargets;
        Bitboard capR = (Us == WHITE ? shift_ne(pawnsNot7) : shift_se(pawnsNot7)) & capTargets;
        while (capL) {
            Square to = pop_lsb(capL);
            list.push(make_move(Square(to - CapLeftDelta), to, MF_CAPTURE));
        }
        while (capR) {
            Square to = pop_lsb(capR);
            list.push(make_move(Square(to - CapRightDelta), to, MF_CAPTURE));
        }

        // En passant
        if (pos.ep_square() != SQ_NONE) {
            Square ep = pos.ep_square();
            Bitboard epPawns = pawnsNot7 & Bitboards::pawn_attacks(Them, ep);
            while (epPawns) {
                Square from = pop_lsb(epPawns);
                list.push(make_move(from, ep, MF_EP_CAPTURE));
            }
        }

        // Promotions (all capture-promotions + queen quiet-promotion for captures stage)
        if (pawnsOn7) {
            Bitboard promoCapL = (Us == WHITE ? shift_nw(pawnsOn7) : shift_sw(pawnsOn7)) & capTargets;
            Bitboard promoCapR = (Us == WHITE ? shift_ne(pawnsOn7) : shift_se(pawnsOn7)) & capTargets;
            while (promoCapL) {
                Square to = pop_lsb(promoCapL);
                Square from = Square(to - CapLeftDelta);
                list.push(make_move(from, to, MF_PROMO_CAP_Q));
                if constexpr (Type == GEN_ALL) {
                    list.push(make_move(from, to, MF_PROMO_CAP_R));
                    list.push(make_move(from, to, MF_PROMO_CAP_B));
                    list.push(make_move(from, to, MF_PROMO_CAP_N));
                }
            }
            while (promoCapR) {
                Square to = pop_lsb(promoCapR);
                Square from = Square(to - CapRightDelta);
                list.push(make_move(from, to, MF_PROMO_CAP_Q));
                if constexpr (Type == GEN_ALL) {
                    list.push(make_move(from, to, MF_PROMO_CAP_R));
                    list.push(make_move(from, to, MF_PROMO_CAP_B));
                    list.push(make_move(from, to, MF_PROMO_CAP_N));
                }
            }

            Bitboard promoPush = pawn_push_bb<Us>(pawnsOn7) & empty & target;
            while (promoPush) {
                Square to = pop_lsb(promoPush);
                Square from = Square(to - PushDelta);
                list.push(make_move(from, to, MF_PROMO_Q));
                if constexpr (Type == GEN_ALL) {
                    list.push(make_move(from, to, MF_PROMO_R));
                    list.push(make_move(from, to, MF_PROMO_B));
                    list.push(make_move(from, to, MF_PROMO_N));
                }
            }
        }
    }

    // Quiet pushes & underpromotions (when Type == GEN_QUIETS)
    if constexpr (Type == GEN_QUIETS || Type == GEN_ALL) {
        Bitboard push1 = pawn_push_bb<Us>(pawnsNot7) & empty;
        Bitboard push2 = pawn_push_bb<Us>(push1 & Rank3BB) & empty & target;
        push1 &= target;

        while (push1) {
            Square to = pop_lsb(push1);
            list.push(make_move(Square(to - PushDelta), to, MF_QUIET));
        }
        while (push2) {
            Square to = pop_lsb(push2);
            list.push(make_move(Square(to - 2 * PushDelta), to, MF_DOUBLE_PUSH));
        }

        if constexpr (Type == GEN_QUIETS) {
            if (pawnsOn7) {
                Bitboard capTargets = enemies & target;
                Bitboard promoCapL = (Us == WHITE ? shift_nw(pawnsOn7) : shift_sw(pawnsOn7)) & capTargets;
                Bitboard promoCapR = (Us == WHITE ? shift_ne(pawnsOn7) : shift_se(pawnsOn7)) & capTargets;
                while (promoCapL) {
                    Square to = pop_lsb(promoCapL);
                    Square from = Square(to - CapLeftDelta);
                    list.push(make_move(from, to, MF_PROMO_CAP_N));
                    list.push(make_move(from, to, MF_PROMO_CAP_R));
                    list.push(make_move(from, to, MF_PROMO_CAP_B));
                }
                while (promoCapR) {
                    Square to = pop_lsb(promoCapR);
                    Square from = Square(to - CapRightDelta);
                    list.push(make_move(from, to, MF_PROMO_CAP_N));
                    list.push(make_move(from, to, MF_PROMO_CAP_R));
                    list.push(make_move(from, to, MF_PROMO_CAP_B));
                }
                Bitboard promoPush = pawn_push_bb<Us>(pawnsOn7) & empty & target;
                while (promoPush) {
                    Square to = pop_lsb(promoPush);
                    Square from = Square(to - PushDelta);
                    list.push(make_move(from, to, MF_PROMO_N));
                    list.push(make_move(from, to, MF_PROMO_R));
                    list.push(make_move(from, to, MF_PROMO_B));
                }
            }
        }
    }
}

template<Color Us, PieceType Pt, GenType Type>
void generate_piece_moves(const Position& pos, MoveList& list, Bitboard target) {
    Bitboard bb = pos.pieces(Us, Pt);
    Bitboard occ = pos.pieces();
    Bitboard enemies = pos.pieces(~Us);

    while (bb) {
        Square from = pop_lsb(bb);
        Bitboard att = Bitboards::attacks_bb(Pt, from, occ) & target;
        if constexpr (Type == GEN_CAPTURES) {
            Bitboard caps = att & enemies;
            while (caps) {
                list.push(make_move(from, pop_lsb(caps), MF_CAPTURE));
            }
        } else if constexpr (Type == GEN_QUIETS) {
            Bitboard quiets = att & ~enemies;
            while (quiets) {
                list.push(make_move(from, pop_lsb(quiets), MF_QUIET));
            }
        } else {
            Bitboard caps = att & enemies;
            Bitboard quiets = att & ~enemies;
            while (caps) {
                list.push(make_move(from, pop_lsb(caps), MF_CAPTURE));
            }
            while (quiets) {
                list.push(make_move(from, pop_lsb(quiets), MF_QUIET));
            }
        }
    }
}

template<Color Us, GenType Type>
void generate_for_color(const Position& pos, MoveList& list) {
    constexpr Color Them = ~Us;
    Bitboard checkers = pos.checkers();
    Square ksq = pos.king_square(Us);
    Bitboard usBB = pos.pieces(Us);
    Bitboard themBB = pos.pieces(Them);

    // King moves
    Bitboard kAtt = Bitboards::king_attacks(ksq) & ~usBB;
    if constexpr (Type == GEN_CAPTURES) {
        Bitboard caps = kAtt & themBB;
        while (caps) list.push(make_move(ksq, pop_lsb(caps), MF_CAPTURE));
    } else if constexpr (Type == GEN_QUIETS) {
        Bitboard quiets = kAtt & ~themBB;
        while (quiets) list.push(make_move(ksq, pop_lsb(quiets), MF_QUIET));
    } else {
        Bitboard caps = kAtt & themBB;
        Bitboard quiets = kAtt & ~themBB;
        while (caps) list.push(make_move(ksq, pop_lsb(caps), MF_CAPTURE));
        while (quiets) list.push(make_move(ksq, pop_lsb(quiets), MF_QUIET));
    }

    // Double check: only king moves
    if (more_than_one(checkers)) return;

    Bitboard target = checkers
        ? (Bitboards::between_bb(ksq, lsb(checkers)) | checkers)
        : ~usBB;

    generate_pawn_moves<Us, Type>(pos, list, target);
    generate_piece_moves<Us, KNIGHT, Type>(pos, list, target);
    generate_piece_moves<Us, BISHOP, Type>(pos, list, target);
    generate_piece_moves<Us, ROOK, Type>(pos, list, target);
    generate_piece_moves<Us, QUEEN, Type>(pos, list, target);

    // Castling (only when not in check and generating quiets/all)
    if constexpr (Type == GEN_QUIETS || Type == GEN_ALL) {
        if (!checkers) {
            Bitboard occ = pos.pieces();
            U8 cr = pos.castling_rights();
            if constexpr (Us == WHITE) {
                if ((cr & WHITE_OO) && !(occ & (square_bb(SQ_F1) | square_bb(SQ_G1)))) {
                    if (!pos.is_attacked_by(BLACK, SQ_F1) && !pos.is_attacked_by(BLACK, SQ_G1))
                        list.push(make_move(SQ_E1, SQ_G1, MF_CASTLE_KS));
                }
                if ((cr & WHITE_OOO) && !(occ & (square_bb(SQ_B1) | square_bb(SQ_C1) | square_bb(SQ_D1)))) {
                    if (!pos.is_attacked_by(BLACK, SQ_D1) && !pos.is_attacked_by(BLACK, SQ_C1))
                        list.push(make_move(SQ_E1, SQ_C1, MF_CASTLE_QS));
                }
            } else {
                if ((cr & BLACK_OO) && !(occ & (square_bb(SQ_F8) | square_bb(SQ_G8)))) {
                    if (!pos.is_attacked_by(WHITE, SQ_F8) && !pos.is_attacked_by(WHITE, SQ_G8))
                        list.push(make_move(SQ_E8, SQ_G8, MF_CASTLE_KS));
                }
                if ((cr & BLACK_OOO) && !(occ & (square_bb(SQ_B8) | square_bb(SQ_C8) | square_bb(SQ_D8)))) {
                    if (!pos.is_attacked_by(WHITE, SQ_D8) && !pos.is_attacked_by(WHITE, SQ_C8))
                        list.push(make_move(SQ_E8, SQ_C8, MF_CASTLE_QS));
                }
            }
        }
    }
}

} // namespace

void generate_captures(const Position& pos, MoveList& list) {
    if (pos.side_to_move() == WHITE)
        generate_for_color<WHITE, GEN_CAPTURES>(pos, list);
    else
        generate_for_color<BLACK, GEN_CAPTURES>(pos, list);
}

void generate_quiets(const Position& pos, MoveList& list) {
    if (pos.side_to_move() == WHITE)
        generate_for_color<WHITE, GEN_QUIETS>(pos, list);
    else
        generate_for_color<BLACK, GEN_QUIETS>(pos, list);
}

void generate_all(const Position& pos, MoveList& list) {
    if (pos.side_to_move() == WHITE)
        generate_for_color<WHITE, GEN_ALL>(pos, list);
    else
        generate_for_color<BLACK, GEN_ALL>(pos, list);
}

void generate_legal(const Position& pos, MoveList& list) {
    MoveList pseudo;
    generate_all(pos, pseudo);
    for (int i = 0; i < pseudo.count; ++i) {
        if (pos.is_legal(pseudo.moves[i].move))
            list.push(pseudo.moves[i].move);
    }
}

} // namespace MoveGen
