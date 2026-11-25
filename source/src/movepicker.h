#pragma once

#include "position.h"
#include "movegen.h"
#include "tune.h"
#include <algorithm>
#include <cstring>

struct SearchTables {
    static constexpr int MAX_HISTORY = 16384;
    static constexpr int CORR_SIZE = 16384;
    static constexpr int CORR_MASK = CORR_SIZE - 1;
    static constexpr int CORR_LIMIT = 1024;
    static constexpr int PAWN_HIST_SIZE = 512;
    static constexpr int PAWN_HIST_MASK = PAWN_HIST_SIZE - 1;

    int16_t mainHist[COLOR_NB][SQ_NB][SQ_NB][4]; // indexed by [us][from][to][fromThreat*2 + toThreat]
    int16_t pawnHist[PAWN_HIST_SIZE][PIECE_NB][SQ_NB];
    int16_t capHist[PIECE_NB][SQ_NB][PIECE_TYPE_NB];
    int16_t contHist[PIECE_NB][SQ_NB][PIECE_NB][SQ_NB];
    Move counterMoves[PIECE_NB][SQ_NB];

    int16_t pawnCorr[COLOR_NB][CORR_SIZE];
    int16_t nonPawnCorr[COLOR_NB][COLOR_NB][CORR_SIZE];
    int16_t minorCorr[COLOR_NB][CORR_SIZE];
    int16_t majorCorr[COLOR_NB][CORR_SIZE];
    int16_t prevMoveCorr[PIECE_NB][SQ_NB];
    int16_t contCorr[PIECE_NB][SQ_NB][PIECE_NB][SQ_NB];

    void clear() {
        std::memset(this, 0, sizeof(SearchTables));
    }

    static inline void update_stat(int16_t& entry, int bonus) {
        int clamped = std::clamp(bonus, -MAX_HISTORY, MAX_HISTORY);
        entry = int16_t(int(entry) + clamped - (int(entry) * std::abs(clamped)) / MAX_HISTORY);
    }

    static inline void update_corr(int16_t& entry, int bonus) {
        int clamped = std::clamp(bonus, -CORR_LIMIT / 4, CORR_LIMIT / 4);
        entry = int16_t(int(entry) + clamped - (int(entry) * std::abs(clamped)) / CORR_LIMIT);
    }

    int correct_eval(const Position& pos, int rawEval,
                     Piece prevPc1, Square prevTo1,
                     Piece prevPc2, Square prevTo2) const {
        Color us = pos.side_to_move();
        int corr = 0;
        corr += int(pawnCorr[us][pos.pawn_key() & CORR_MASK]) * Tune::CorrPawnW;
        corr += int(nonPawnCorr[us][WHITE][pos.non_pawn_key(WHITE) & CORR_MASK]) * Tune::CorrNonPawnW;
        corr += int(nonPawnCorr[us][BLACK][pos.non_pawn_key(BLACK) & CORR_MASK]) * Tune::CorrNonPawnW;
        corr += int(minorCorr[us][pos.minor_key() & CORR_MASK]) * Tune::CorrMinMajW;
        corr += int(majorCorr[us][pos.major_key() & CORR_MASK]) * Tune::CorrMinMajW;
        if (prevPc1 != NO_PIECE) {
            corr += int(prevMoveCorr[prevPc1][prevTo1]) * ((Tune::CorrContW * 9) / 10);
            if (prevPc2 != NO_PIECE) {
                corr += int(contCorr[prevPc2][prevTo2][prevPc1][prevTo1]) * Tune::CorrContW;
            }
        }
        int delta = std::clamp(corr / 256, -260, 260);
        int adjusted = rawEval + delta;
        return std::clamp(adjusted, SCORE_MATED_IN_MAX + 1, SCORE_MATE_IN_MAX - 1);
    }

    void update_correction(const Position& pos, int depth, int diff,
                           Piece prevPc1, Square prevTo1,
                           Piece prevPc2, Square prevTo2) {
        Color us = pos.side_to_move();
        int bonus = std::clamp(diff * depth / 8, -CORR_LIMIT / 4, CORR_LIMIT / 4);
        update_corr(pawnCorr[us][pos.pawn_key() & CORR_MASK], bonus);
        update_corr(nonPawnCorr[us][WHITE][pos.non_pawn_key(WHITE) & CORR_MASK], bonus);
        update_corr(nonPawnCorr[us][BLACK][pos.non_pawn_key(BLACK) & CORR_MASK], bonus);
        update_corr(minorCorr[us][pos.minor_key() & CORR_MASK], bonus);
        update_corr(majorCorr[us][pos.major_key() & CORR_MASK], bonus);
        if (prevPc1 != NO_PIECE) {
            update_corr(prevMoveCorr[prevPc1][prevTo1], bonus);
            if (prevPc2 != NO_PIECE) {
                update_corr(contCorr[prevPc2][prevTo2][prevPc1][prevTo1], bonus);
            }
        }
    }
};

struct SearchStack {
    Move pv[MAX_PLY + 1];
    int pvLen = 0;
    int staticEval = SCORE_NONE;
    Move move = MOVE_NONE;
    Piece movedPiece = NO_PIECE;
    Move killers[2] = {MOVE_NONE, MOVE_NONE};
    Move excludedMove = MOVE_NONE;
    int doubleExtensions = 0;
    int cutoffCount = 0;
    int reduction = 0;
    bool nmpDisabled = false;
    int16_t (*contHist)[SQ_NB] = nullptr;
};

class MovePicker {
public:
    enum Stage {
        STAGE_TT,
        STAGE_GEN_CAPTURES,
        STAGE_GOOD_CAPTURES,
        STAGE_KILLER1,
        STAGE_KILLER2,
        STAGE_COUNTERMOVE,
        STAGE_GEN_QUIETS,
        STAGE_QUIETS,
        STAGE_BAD_CAPTURES,
        STAGE_Q_TT,
        STAGE_Q_GEN_CAPTURES,
        STAGE_Q_CAPTURES,
        STAGE_DONE
    };

    // Main search constructor
    MovePicker(const Position& p, Move ttM, const SearchTables& tbl,
               const SearchStack* stack, int seeThresh = 0)
        : pos(p), ttMove(ttM), tables(tbl), ss(stack), qsearch(false), seeThreshold(seeThresh) {
        killer1 = ss->killers[0];
        killer2 = ss->killers[1];
        counterMove = ((ss - 1)->movedPiece != NO_PIECE && (ss - 1)->move != MOVE_NONE)
            ? tables.counterMoves[(ss - 1)->movedPiece][to_sq((ss - 1)->move)]
            : MOVE_NONE;
        stage = (ttMove != MOVE_NONE && pos.is_pseudo_legal(ttMove)) ? STAGE_TT : STAGE_GEN_CAPTURES;
    }

    // QSearch constructor
    MovePicker(const Position& p, Move ttM, const SearchTables& tbl, bool inCheck)
        : pos(p), ttMove(ttM), tables(tbl), ss(nullptr), qsearch(true), qInCheck(inCheck) {
        if (ttMove != MOVE_NONE && pos.is_pseudo_legal(ttMove) && (inCheck || is_tactical(ttMove)))
            stage = STAGE_Q_TT;
        else
            stage = STAGE_Q_GEN_CAPTURES;
    }

    void skip_quiets() { skipQuiets = true; }

    Move next(int& outScore) {
        outScore = 0;
        while (true) {
            switch (stage) {
                case STAGE_TT:
                    stage = STAGE_GEN_CAPTURES;
                    return ttMove;

                case STAGE_GEN_CAPTURES: {
                    MoveGen::generate_captures(pos, captures);
                    score_captures();
                    capIdx = 0;
                    badCapCount = 0;
                    stage = STAGE_GOOD_CAPTURES;
                    break;
                }

                case STAGE_GOOD_CAPTURES: {
                    while (capIdx < captures.count) {
                        int best = pick_best(captures, capIdx);
                        Move m = captures.moves[best].move;
                        int sc = captures.moves[best].score;
                        capIdx++;
                        if (m == ttMove) continue;
                        Square from = from_sq(m), to = to_sq(m);
                        PieceType capPt = is_enpassant(m) ? PAWN : type_of(pos.piece_on(to));
                        int ch = (capPt < PIECE_TYPE_NB) ? tables.capHist[pos.piece_on(from)][to][capPt] : 0;
                        if (pos.see_ge(m, -ch / 64)) {
                            outScore = sc;
                            return m;
                        }
                        badCaptures[badCapCount++] = {m, sc};
                    }
                    stage = skipQuiets ? STAGE_BAD_CAPTURES : STAGE_KILLER1;
                    break;
                }

                case STAGE_KILLER1:
                    stage = STAGE_KILLER2;
                    if (!skipQuiets && killer1 != MOVE_NONE && killer1 != ttMove
                        && !is_tactical(killer1) && pos.is_pseudo_legal(killer1)) {
                        return killer1;
                    }
                    break;

                case STAGE_KILLER2:
                    stage = STAGE_COUNTERMOVE;
                    if (!skipQuiets && killer2 != MOVE_NONE && killer2 != ttMove && killer2 != killer1
                        && !is_tactical(killer2) && pos.is_pseudo_legal(killer2)) {
                        return killer2;
                    }
                    break;

                case STAGE_COUNTERMOVE:
                    stage = STAGE_GEN_QUIETS;
                    if (!skipQuiets && counterMove != MOVE_NONE && counterMove != ttMove
                        && counterMove != killer1 && counterMove != killer2
                        && !is_tactical(counterMove) && pos.is_pseudo_legal(counterMove)) {
                        return counterMove;
                    }
                    break;

                case STAGE_GEN_QUIETS:
                    if (skipQuiets) {
                        stage = STAGE_BAD_CAPTURES;
                        break;
                    }
                    MoveGen::generate_quiets(pos, quiets);
                    score_quiets();
                    quietIdx = 0;
                    stage = STAGE_QUIETS;
                    break;

                case STAGE_QUIETS:
                    if (skipQuiets) {
                        stage = STAGE_BAD_CAPTURES;
                        break;
                    }
                    while (quietIdx < quiets.count) {
                        int best = pick_best(quiets, quietIdx);
                        Move m = quiets.moves[best].move;
                        int sc = quiets.moves[best].score;
                        quietIdx++;
                        if (m == ttMove || m == killer1 || m == killer2 || m == counterMove)
                            continue;
                        outScore = sc;
                        return m;
                    }
                    stage = STAGE_BAD_CAPTURES;
                    break;

                case STAGE_BAD_CAPTURES:
                    if (badCapIdx < badCapCount) {
                        outScore = badCaptures[badCapIdx].score;
                        return badCaptures[badCapIdx++].move;
                    }
                    stage = STAGE_DONE;
                    return MOVE_NONE;

                case STAGE_Q_TT:
                    stage = STAGE_Q_GEN_CAPTURES;
                    return ttMove;

                case STAGE_Q_GEN_CAPTURES:
                    if (qInCheck) {
                        MoveGen::generate_all(pos, captures);
                    } else {
                        MoveGen::generate_captures(pos, captures);
                    }
                    score_qmoves();
                    capIdx = 0;
                    stage = STAGE_Q_CAPTURES;
                    break;

                case STAGE_Q_CAPTURES:
                    while (capIdx < captures.count) {
                        int best = pick_best(captures, capIdx);
                        Move m = captures.moves[best].move;
                        int sc = captures.moves[best].score;
                        capIdx++;
                        if (m == ttMove) continue;
                        outScore = sc;
                        return m;
                    }
                    stage = STAGE_DONE;
                    return MOVE_NONE;

                case STAGE_DONE:
                    return MOVE_NONE;
            }
        }
    }

    Stage current_stage() const { return stage; }

private:
    static int pick_best(MoveList& ml, int start) {
        int best = start;
        int bestScore = ml.moves[start].score;
        for (int i = start + 1; i < ml.count; ++i) {
            if (ml.moves[i].score > bestScore) {
                bestScore = ml.moves[i].score;
                best = i;
            }
        }
        std::swap(ml.moves[start], ml.moves[best]);
        return start;
    }

    void score_captures() {
        static constexpr int MvvVal[PIECE_TYPE_NB + 1] = {100, 320, 330, 500, 950, 20000, 0};
        for (int i = 0; i < captures.count; ++i) {
            Move m = captures.moves[i].move;
            Square from = from_sq(m), to = to_sq(m);
            Piece pc = pos.piece_on(from);
            PieceType capPt = is_enpassant(m) ? PAWN : type_of(pos.piece_on(to));
            int mvv = MvvVal[capPt];
            if (is_promotion(m)) mvv += MvvVal[promo_type(m)] - MvvVal[PAWN];
            int ch = (capPt < PIECE_TYPE_NB) ? tables.capHist[pc][to][capPt] : 0;
            captures.moves[i].score = mvv * 32 + ch;
        }
    }

    static inline int threat_idx(const Position& pos, Square from, Square to) {
        Color us = pos.side_to_move();
        Bitboard enemyPawnAtt = (us == WHITE)
            ? pawn_attacks_bb<BLACK>(pos.pieces(BLACK, PAWN))
            : pawn_attacks_bb<WHITE>(pos.pieces(WHITE, PAWN));
        int fT = (enemyPawnAtt & square_bb(from)) != 0;
        int tT = (enemyPawnAtt & square_bb(to)) != 0;
        return (fT << 1) | tT;
    }

    void score_quiets() {
        Color us = pos.side_to_move();
        size_t pIdx = pos.pawn_key() & SearchTables::PAWN_HIST_MASK;
        Bitboard enemyPawnAtt = (us == WHITE)
            ? pawn_attacks_bb<BLACK>(pos.pieces(BLACK, PAWN))
            : pawn_attacks_bb<WHITE>(pos.pieces(WHITE, PAWN));
        const int16_t (*ch1)[SQ_NB] = (ss - 1)->contHist;
        const int16_t (*ch2)[SQ_NB] = (ss - 2)->contHist;
        const int16_t (*ch4)[SQ_NB] = (ss - 4)->contHist;
        const int16_t (*ch6)[SQ_NB] = (ss - 6)->contHist;

        for (int i = 0; i < quiets.count; ++i) {
            Move m = quiets.moves[i].move;
            Square from = from_sq(m), to = to_sq(m);
            Piece pc = pos.piece_on(from);
            int tIdx = (((enemyPawnAtt & square_bb(from)) != 0) << 1) | ((enemyPawnAtt & square_bb(to)) != 0);
            int s = tables.mainHist[us][from][to][tIdx] + tables.pawnHist[pIdx][pc][to];
            if (ch1) s += ch1[pc][to];
            if (ch2) s += ch2[pc][to];
            if (ch4) s += ch4[pc][to] / 2;
            if (ch6) s += ch6[pc][to] / 4;
            quiets.moves[i].score = s;
        }
    }

    void score_qmoves() {
        static constexpr int MvvVal[PIECE_TYPE_NB + 1] = {100, 320, 330, 500, 950, 20000, 0};
        Color us = pos.side_to_move();
        for (int i = 0; i < captures.count; ++i) {
            Move m = captures.moves[i].move;
            Square from = from_sq(m), to = to_sq(m);
            Piece pc = pos.piece_on(from);
            if (is_tactical(m)) {
                PieceType capPt = is_enpassant(m) ? PAWN : type_of(pos.piece_on(to));
                int mvv = MvvVal[capPt];
                if (is_promotion(m)) mvv += MvvVal[promo_type(m)] - MvvVal[PAWN];
                int ch = (capPt < PIECE_TYPE_NB) ? tables.capHist[pc][to][capPt] : 0;
                captures.moves[i].score = 100000 + mvv * 32 + ch;
            } else {
                captures.moves[i].score = tables.mainHist[us][from][to][0];
            }
        }
    }

    const Position& pos;
    Move ttMove;
    Move killer1 = MOVE_NONE;
    Move killer2 = MOVE_NONE;
    Move counterMove = MOVE_NONE;
    const SearchTables& tables;
    const SearchStack* ss;

    Stage stage;
    bool qsearch = false;
    bool qInCheck = false;
    bool skipQuiets = false;
    int seeThreshold = 0;

    MoveList captures;
    MoveList quiets;
    ExtMove badCaptures[MAX_MOVES];
    int capIdx = 0;
    int quietIdx = 0;
    int badCapCount = 0;
    int badCapIdx = 0;
};
