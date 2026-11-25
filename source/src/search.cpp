#include "search.h"
#include "tune.h"
#include "eval.h"
#include "movegen.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace {

inline int history_bonus(int depth) {
    return std::min(280 * depth - 200, 2400);
}

inline int format_mate_moves(int score) {
    if (score > 0) return (SCORE_MATE - score + 1) / 2;
    return -(SCORE_MATE + score) / 2;
}

inline int quiet_history_score(const Position& pos, const SearchTables& tables, const SearchStack* ss, Move m) {
    Square from = from_sq(m), to = to_sq(m);
    Piece pc = pos.piece_on(from);
    Color us = pos.side_to_move();
    Bitboard enemyPawnAtt = (us == WHITE)
        ? pawn_attacks_bb<BLACK>(pos.pieces(BLACK, PAWN))
        : pawn_attacks_bb<WHITE>(pos.pieces(WHITE, PAWN));
    int tIdx = (((enemyPawnAtt & square_bb(from)) != 0) << 1) | ((enemyPawnAtt & square_bb(to)) != 0);
    int s = tables.mainHist[us][from][to][tIdx]
          + tables.pawnHist[pos.pawn_key() & SearchTables::PAWN_HIST_MASK][pc][to];
    if ((ss - 1)->contHist) s += (ss - 1)->contHist[pc][to];
    if ((ss - 2)->contHist) s += (ss - 2)->contHist[pc][to];
    if ((ss - 4)->contHist) s += (ss - 4)->contHist[pc][to] / 2;
    if ((ss - 6)->contHist) s += (ss - 6)->contHist[pc][to] / 4;
    return s;
}

} // namespace

Searcher::Searcher()
    : tablesPtr(std::make_unique<SearchTables>()),
      tables(*tablesPtr),
      accStackPtr(std::make_unique<NNUE::AccumulatorStack>()),
      accStack(*accStackPtr) {
    init_tables();
    clear_state();
}

int Searcher::evaluate_pos(const Position& pos, int accPly) {
    if (useNNUE) return accStack.evaluate(pos, accPly);
    return Eval::evaluate(pos);
}

void Searcher::init_tables() {
    for (int d = 0; d < 64; ++d) {
        for (int m = 0; m < 64; ++m) {
            if (d == 0 || m == 0)
                lmrTable[d][m] = 0;
            else
                lmrTable[d][m] = int((Tune::LmrBase / 100.0 + std::log(double(d)) * std::log(double(m)) / (Tune::LmrDiv / 100.0)) * 1024.0);
        }
    }
}

void Searcher::clear_state() {
    tables.clear();
}

int64_t Searcher::elapsed_ms() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
}

void Searcher::check_time() {
    if (stopFlag.load(std::memory_order_relaxed)) {
        aborted = true;
        return;
    }
    if (nodeLimit > 0 && nodes >= nodeLimit) {
        aborted = true;
        return;
    }
    if (hasTimeLimit && elapsed_ms() >= hardLimitMs) {
        aborted = true;
    }
}

void Searcher::print_info(int depth, int score, const Move* pv, int pvLen, std::mutex* outMtx) {
    if (silentMode) return;
    int64_t ms = std::max<int64_t>(1, elapsed_ms());
    U64 nps = (nodes * 1000ULL) / U64(ms);
    int hf = tt.hashfull();

    std::string out = "info depth " + std::to_string(depth)
        + " seldepth " + std::to_string(std::max(depth, selDepth));

    if (score >= SCORE_MATE_IN_MAX || score <= SCORE_MATED_IN_MAX) {
        out += " score mate " + std::to_string(format_mate_moves(score));
    } else {
        out += " score cp " + std::to_string(score);
    }

    out += " nodes " + std::to_string(nodes)
        + " nps " + std::to_string(nps)
        + " hashfull " + std::to_string(hf)
        + " time " + std::to_string(ms)
        + " pv";

    for (int i = 0; i < pvLen; ++i) {
        out += ' ';
        out += move_to_uci(pv[i]);
    }
    out += '\n';

    if (outMtx) {
        std::lock_guard<std::mutex> lock(*outMtx);
        std::cout << out << std::flush;
    } else {
        std::cout << out << std::flush;
    }
}

void Searcher::update_quiet_stats(const Position& pos, SearchStack* ss, Move bestMove,
                                  const Move* quietsTried, int quietCount, int depth) {
    int bonus = history_bonus(depth);
    Color us = pos.side_to_move();
    size_t pIdx = pos.pawn_key() & SearchTables::PAWN_HIST_MASK;

    if (bestMove != MOVE_NONE) {
        if (ss->killers[0] != bestMove) {
            ss->killers[1] = ss->killers[0];
            ss->killers[0] = bestMove;
        }
        if ((ss - 1)->movedPiece != NO_PIECE && (ss - 1)->move != MOVE_NONE) {
            tables.counterMoves[(ss - 1)->movedPiece][to_sq((ss - 1)->move)] = bestMove;
        }
    }

    Bitboard enemyPawnAtt = (us == WHITE)
        ? pawn_attacks_bb<BLACK>(pos.pieces(BLACK, PAWN))
        : pawn_attacks_bb<WHITE>(pos.pieces(WHITE, PAWN));

    auto update_one = [&](Move m, int b) {
        Square from = from_sq(m), to = to_sq(m);
        Piece pc = pos.piece_on(from);
        int tIdx = (((enemyPawnAtt & square_bb(from)) != 0) << 1) | ((enemyPawnAtt & square_bb(to)) != 0);
        SearchTables::update_stat(tables.mainHist[us][from][to][tIdx], b);
        SearchTables::update_stat(tables.pawnHist[pIdx][pc][to], b);
        if ((ss - 1)->contHist) SearchTables::update_stat((ss - 1)->contHist[pc][to], b);
        if ((ss - 2)->contHist) SearchTables::update_stat((ss - 2)->contHist[pc][to], b);
        if ((ss - 4)->contHist) SearchTables::update_stat((ss - 4)->contHist[pc][to], b / 2);
        if ((ss - 6)->contHist) SearchTables::update_stat((ss - 6)->contHist[pc][to], b / 4);
    };

    if (bestMove != MOVE_NONE) {
        update_one(bestMove, bonus);
    }
    int malus = (bestMove != MOVE_NONE) ? -bonus : -(bonus / 2);
    for (int i = 0; i < quietCount; ++i) {
        if (quietsTried[i] != bestMove)
            update_one(quietsTried[i], malus);
    }
}

void Searcher::update_cap_stats(const Position& pos, Move bestMove,
                                const Move* capsTried, int capCount, int depth) {
    int bonus = history_bonus(depth);
    auto update_one = [&](Move m, int b) {
        Square from = from_sq(m), to = to_sq(m);
        Piece pc = pos.piece_on(from);
        PieceType capPt = is_enpassant(m) ? PAWN : type_of(pos.piece_on(to));
        if (capPt < PIECE_TYPE_NB)
            SearchTables::update_stat(tables.capHist[pc][to][capPt], b);
    };
    if (bestMove != MOVE_NONE && is_tactical(bestMove))
        update_one(bestMove, bonus);
    for (int i = 0; i < capCount; ++i) {
        if (capsTried[i] != bestMove)
            update_one(capsTried[i], -bonus);
    }
}

int Searcher::qsearch(Position& pos, SearchStack* ss, int alpha, int beta, int ply, int accPly) {
    if ((nodes & 511) == 0) check_time();
    if (aborted) return 0;

    if (ply >= MAX_PLY - 1) return evaluate_pos(pos, accPly);
    if (ply > selDepth) selDepth = ply;

    if (pos.is_draw(ply)) return 1 - int(nodes & 2);

    bool isPv = (beta - alpha > 1);
    bool inCheck = pos.in_check();

    bool ttHit = false;
    TTEntry* tte = tt.probe(pos.key(), ttHit);
    Move ttMove = ttHit ? tte->move : MOVE_NONE;
    int ttScore = ttHit ? TranspositionTable::score_from_tt(tte->score, ply) : SCORE_NONE;
    Bound ttBound = ttHit ? tte->bound() : BOUND_NONE;

    if (!isPv && ttHit) {
        if (ttBound == BOUND_EXACT
            || (ttBound == BOUND_LOWER && ttScore >= beta)
            || (ttBound == BOUND_UPPER && ttScore <= alpha)) {
            return ttScore;
        }
    }

    int rawEval = SCORE_NONE;
    int standPat = -SCORE_INFINITE;

    if (!inCheck) {
        rawEval = (ttHit && tte->eval != SCORE_NONE) ? int(tte->eval) : evaluate_pos(pos, accPly);
        Piece prevPc1 = (ss - 1)->movedPiece;
        Square prevTo1 = (ss - 1)->move != MOVE_NONE ? to_sq((ss - 1)->move) : SQ_A1;
        Piece prevPc2 = (ss - 2)->movedPiece;
        Square prevTo2 = (ss - 2)->move != MOVE_NONE ? to_sq((ss - 2)->move) : SQ_A1;
        standPat = tables.correct_eval(pos, rawEval, prevPc1, prevTo1, prevPc2, prevTo2);

        if (ttHit && ((ttBound == BOUND_LOWER && ttScore > standPat)
                   || (ttBound == BOUND_UPPER && ttScore < standPat)
                   || ttBound == BOUND_EXACT)) {
            standPat = ttScore;
        }

        if (standPat >= beta) return standPat;
        if (standPat > alpha) alpha = standPat;
    }

    int bestScore = inCheck ? -SCORE_MATE + ply : standPat;
    Move bestMove = MOVE_NONE;
    int origAlpha = alpha;
    int movesSearched = 0;
    int futilityBase = standPat + (useNNUE ? Tune::QsFutMargin : 200);

    MovePicker mp(pos, ttMove, tables, inCheck);
    int moveScore = 0;
    Move m;
    while ((m = mp.next(moveScore)) != MOVE_NONE) {
        if (!pos.is_legal(m)) continue;

        if (!inCheck) {
            if (!is_promotion(m)) {
                int capVal = is_enpassant(m) ? 100 : Position::SeeValues[type_of(pos.piece_on(to_sq(m)))];
                if (futilityBase + capVal <= alpha && !pos.gives_check(m)) {
                    bestScore = std::max(bestScore, futilityBase + capVal);
                    continue;
                }
            }
            if (!pos.see_ge(m, 0))
                continue;
        } else if (bestScore > SCORE_MATED_IN_MAX && movesSearched > 0 && !is_tactical(m) && !pos.see_ge(m, -100)) {
            continue;
        }

        movesSearched++;
        nodes++;

        ss->move = m;
        ss->movedPiece = pos.piece_on(from_sq(m));

        if (useNNUE) accStack.push_move(pos, m, accPly);
        StateInfo st;
        pos.do_move(m, st);
        tt.prefetch(pos.key());
        int score = -qsearch(pos, ss + 1, -beta, -alpha, ply + 1, accPly + 1);
        pos.undo_move(m);

        if (aborted) return 0;

        if (score > bestScore) {
            bestScore = score;
            if (score > alpha) {
                bestMove = m;
                alpha = score;
                if (score >= beta) break;
            }
        }
    }

    if (inCheck && movesSearched == 0)
        return -SCORE_MATE + ply;

    Bound bound = bestScore >= beta ? BOUND_LOWER : (bestScore > origAlpha ? BOUND_EXACT : BOUND_UPPER);
    tt.store(pos.key(), bestMove, TranspositionTable::score_to_tt(bestScore, ply),
             rawEval, 0, bound, isPv);
    return bestScore;
}

int Searcher::negamax(Position& pos, SearchStack* ss, int alpha, int beta, int depth, int ply, int accPly, bool cutNode) {
    bool isRoot = (ply == 0);
    bool isPv = (beta - alpha > 1);

    ss->pvLen = 0;

    if ((nodes & 511) == 0) check_time();
    if (aborted) return 0;

    if (ply > selDepth) selDepth = ply;
    if (ply >= MAX_PLY - 1) return pos.in_check() ? 0 : evaluate_pos(pos, accPly);

    Move excludedMove = ss->excludedMove;
    bool inCheck = pos.in_check();
    if (inCheck && excludedMove == MOVE_NONE && depth < MAX_PLY - 1) depth++;

    if (depth <= 0) {
        return qsearch(pos, ss, alpha, beta, ply, accPly);
    }

    if (!isRoot) {
        if (pos.is_draw(ply)) return 1 - int(nodes & 2);

        alpha = std::max(alpha, -SCORE_MATE + ply);
        beta  = std::min(beta,   SCORE_MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    // TT probe
    bool ttHit = false;
    TTEntry* tte = excludedMove == MOVE_NONE ? tt.probe(pos.key(), ttHit) : nullptr;
    Move ttMove = ttHit ? tte->move : MOVE_NONE;
    int ttScore = ttHit ? TranspositionTable::score_from_tt(tte->score, ply) : SCORE_NONE;
    int ttDepth = ttHit ? int(tte->depth) : 0;
    Bound ttBound = ttHit ? tte->bound() : BOUND_NONE;
    bool ttPv = isPv || (ttHit && tte->is_pv());

    if (!isPv && !isRoot && ttHit && ttDepth >= depth) {
        if (ttBound == BOUND_EXACT
            || (ttBound == BOUND_LOWER && ttScore >= beta)
            || (ttBound == BOUND_UPPER && ttScore <= alpha)) {
            if (ttMove != MOVE_NONE && ttScore >= beta && !is_tactical(ttMove) && pos.is_pseudo_legal(ttMove)) {
                int b = history_bonus(depth) / 2;
                Square from = from_sq(ttMove), to = to_sq(ttMove);
                Piece pc = pos.piece_on(from);
                Color us = pos.side_to_move();
                Bitboard enemyPawnAtt = (us == WHITE)
                    ? pawn_attacks_bb<BLACK>(pos.pieces(BLACK, PAWN))
                    : pawn_attacks_bb<WHITE>(pos.pieces(WHITE, PAWN));
                int tIdx = (((enemyPawnAtt & square_bb(from)) != 0) << 1) | ((enemyPawnAtt & square_bb(to)) != 0);
                SearchTables::update_stat(tables.mainHist[us][from][to][tIdx], b);
                if ((ss - 1)->contHist) SearchTables::update_stat((ss - 1)->contHist[pc][to], b);
            }
            return ttScore;
        }
    }

    Piece prevPc1 = (ss - 1)->movedPiece;
    Square prevTo1 = (ss - 1)->move != MOVE_NONE ? to_sq((ss - 1)->move) : SQ_A1;
    Piece prevPc2 = (ss - 2)->movedPiece;
    Square prevTo2 = (ss - 2)->move != MOVE_NONE ? to_sq((ss - 2)->move) : SQ_A1;

    // Static evaluation
    int rawEval = SCORE_NONE;
    int eval = SCORE_NONE;
    if (inCheck) {
        ss->staticEval = SCORE_NONE;
    } else if (excludedMove != MOVE_NONE) {
        eval = ss->staticEval;
    } else {
        rawEval = (ttHit && tte->eval != SCORE_NONE) ? int(tte->eval) : evaluate_pos(pos, accPly);
        eval = tables.correct_eval(pos, rawEval, prevPc1, prevTo1, prevPc2, prevTo2);
        ss->staticEval = eval;

        if (ttHit && ((ttBound == BOUND_LOWER && ttScore > eval)
                   || (ttBound == BOUND_UPPER && ttScore < eval)
                   || ttBound == BOUND_EXACT)) {
            eval = ttScore;
        }
    }

    // Improving flag with (ss-4) fallback when (ss-2) was in check
    bool improving = false;
    if (!inCheck) {
        if ((ss - 2)->staticEval != SCORE_NONE)
            improving = ss->staticEval > (ss - 2)->staticEval;
        else if ((ss - 4)->staticEval != SCORE_NONE)
            improving = ss->staticEval > (ss - 4)->staticEval;
    }

    // Hindsight reduction / extension on prior LMR move
    if (!isRoot && !inCheck && excludedMove == MOVE_NONE && (ss - 1)->staticEval != SCORE_NONE) {
        int evalSum = ss->staticEval + (ss - 1)->staticEval;
        if ((ss - 1)->reduction >= 3 && evalSum < -30) {
            depth--;
            if (depth <= 0) return qsearch(pos, ss, alpha, beta, ply, accPly);
        } else if ((ss - 1)->reduction >= 2 && evalSum > 140 && depth < MAX_PLY - 2) {
            depth++;
        }
    }

    // Clear child killers & state
    (ss + 1)->killers[0] = MOVE_NONE;
    (ss + 1)->killers[1] = MOVE_NONE;
    (ss + 1)->cutoffCount = 0;
    (ss + 1)->reduction = 0;
    (ss + 1)->nmpDisabled = ss->nmpDisabled;

    // Pre-move-loop pruning (non-PV, not in check, not singular)
    if (!isPv && !inCheck && excludedMove == MOVE_NONE) {
        // 1. Reverse Futility Pruning (RFP)
        if (depth <= 9 && eval < SCORE_MATE_IN_MAX) {
            int corrDelta = (rawEval != SCORE_NONE) ? std::abs(rawEval - ss->staticEval) : 0;
            int rfpMargin = (improving ? Tune::RfpImp : Tune::RfpNonImp) * depth + (ttPv ? 35 : 0) + corrDelta / 3;
            if (eval - rfpMargin >= beta)
                return (eval + beta) / 2;
        }

        // 2. Razoring
        if (depth <= 3 && eval + Tune::RazorBase + Tune::RazorMult * depth < alpha) {
            int qScore = qsearch(pos, ss, alpha, beta, ply, accPly);
            if (qScore <= alpha) return qScore;
        }

        // 3. Null Move Pruning (NMP) with zero-copy accPly and high-depth verification
        if (!ss->nmpDisabled && depth >= 3 && eval >= beta
            && ss->staticEval >= beta - 20 * depth + 100
            && (ss - 1)->move != MOVE_NONE && pos.has_non_pawn_material(pos.side_to_move())) {
            int R = 3 + depth / 3 + std::min((eval - beta) / Tune::NmpEvalDiv, 3) + int(!improving && eval - beta > 120);
            ss->move = MOVE_NONE;
            ss->movedPiece = NO_PIECE;
            ss->contHist = nullptr;
            ss->reduction = 0;

            StateInfo nullSt;
            pos.do_null_move(nullSt);
            tt.prefetch(pos.key());
            // Notice: accPly unchanged! Zero-copy NNUE across null moves!
            int nullScore = -negamax(pos, ss + 1, -beta, -beta + 1, depth - R, ply + 1, accPly, !cutNode);
            pos.undo_null_move();

            if (aborted) return 0;
            if (nullScore >= beta) {
                if (nullScore >= SCORE_MATE_IN_MAX) nullScore = beta;
                if (depth < 12 || std::abs(beta) >= SCORE_MATE_IN_MAX) {
                    return nullScore;
                }
                ss->nmpDisabled = true;
                int vScore = negamax(pos, ss, beta - 1, beta, depth - R, ply, accPly, false);
                ss->nmpDisabled = false;
                if (aborted) return 0;
                if (vScore >= beta) return nullScore;
            }
        }

        // 4. ProbCut
        if (depth >= 5 && std::abs(beta) < SCORE_MATE_IN_MAX) {
            int pcBeta = beta + Tune::ProbCutMargin;
            int pcDepth = depth - 4;
            if (!(ttHit && ttDepth >= pcDepth && ttScore < pcBeta)) {
                MovePicker pcMp(pos, ttMove, tables, false);
                int ms = 0;
                Move pcm;
                while ((pcm = pcMp.next(ms)) != MOVE_NONE) {
                    if (!pos.is_legal(pcm)) continue;
                    if (!pos.see_ge(pcm, std::max(0, pcBeta - ss->staticEval))) continue;

                    nodes++;
                    ss->move = pcm;
                    ss->movedPiece = pos.piece_on(from_sq(pcm));
                    ss->contHist = tables.contHist[ss->movedPiece][to_sq(pcm)];
                    ss->reduction = 0;

                    if (useNNUE) accStack.push_move(pos, pcm, accPly);
                    StateInfo st;
                    pos.do_move(pcm, st);
                    tt.prefetch(pos.key());
                    int score = -qsearch(pos, ss + 1, -pcBeta, -pcBeta + 1, ply + 1, accPly + 1);
                    if (score >= pcBeta)
                        score = -negamax(pos, ss + 1, -pcBeta, -pcBeta + 1, pcDepth, ply + 1, accPly + 1, !cutNode);
                    pos.undo_move(pcm);

                    if (aborted) return 0;
                    if (score >= pcBeta) {
                        tt.store(pos.key(), pcm, TranspositionTable::score_to_tt(score, ply),
                                 rawEval, pcDepth + 1, BOUND_LOWER, ttPv);
                        return score;
                    }
                }
            }
        }
    }

    // Internal Iterative Reduction (IIR)
    if (depth >= 4 && (ttMove == MOVE_NONE || (depth >= 7 && ttDepth + 4 < depth))
        && excludedMove == MOVE_NONE && (isPv || cutNode)) {
        depth--;
    }

    int origAlpha = alpha;
    int bestScore = -SCORE_INFINITE;
    Move bestMove = MOVE_NONE;
    int movesSearched = 0;

    Move quietsTried[MAX_MOVES];
    int quietCount = 0;
    Move capsTried[MAX_MOVES];
    int capCount = 0;

    bool ttCapture = (ttMove != MOVE_NONE && is_tactical(ttMove));

    MovePicker mp(pos, ttMove, tables, ss);
    int moveScore = 0;
    Move m;
    while ((m = mp.next(moveScore)) != MOVE_NONE) {
        if (m == excludedMove) continue;
        if (!pos.is_legal(m)) continue;

        bool tactical = is_tactical(m);
        bool isQuiet = !tactical;

        // Avoid redundant history lookups: STAGE_QUIETS already returns quiet_history_score in moveScore!
        int histScore = 0;
        if (isQuiet) {
            histScore = (mp.current_stage() == MovePicker::STAGE_QUIETS)
                ? moveScore
                : quiet_history_score(pos, tables, ss, m);
        }
        int capHistScore = 0;
        if (tactical) {
            PieceType capPt = is_enpassant(m) ? PAWN : type_of(pos.piece_on(to_sq(m)));
            if (capPt < PIECE_TYPE_NB)
                capHistScore = tables.capHist[pos.piece_on(from_sq(m))][to_sq(m)][capPt];
        }

        // Move-loop pruning
        if (!isRoot && bestScore > SCORE_MATED_IN_MAX) {
            int lmrRed = lmrTable[std::min(depth, 63)][std::min(movesSearched, 63)] / 1024;
            int lmrDepth = std::max(0, depth - lmrRed);

            if (isQuiet) {
                // Late Move Pruning (LMP)
                int lmpLimit = (3 + depth * depth) / (2 - int(improving));
                if (depth <= 8 && movesSearched >= lmpLimit) {
                    mp.skip_quiets();
                    continue;
                }

                // Futility Pruning (only skip_quiets on base margin; per-move with histScore!)
                if (!inCheck && lmrDepth <= 7) {
                    int baseFut = ss->staticEval + Tune::FutBase + Tune::FutMult * lmrDepth;
                    if (baseFut + 200 <= alpha) {
                        mp.skip_quiets();
                        continue;
                    }
                    if (baseFut + histScore / 64 <= alpha) {
                        continue;
                    }
                }

                // History Pruning
                if (depth <= 5 && histScore < -Tune::HistPruneMult * depth) {
                    continue;
                }

                // SEE Pruning for quiets
                if (depth <= 8 && !pos.see_ge(m, -Tune::SeeQuietMult * depth * depth)) {
                    continue;
                }
            } else {
                // SEE Pruning for bad captures (history-adjusted)
                if (depth <= 6 && !pos.see_ge(m, -Tune::SeeCapMult * depth - capHistScore / 64)) {
                    continue;
                }
            }
        }

        // Singular Extensions
        int extension = 0;
        if (!isRoot && ply < MAX_PLY - 2 && depth >= 7 && m == ttMove && excludedMove == MOVE_NONE
            && ttDepth >= depth - 3 && (ttBound & BOUND_LOWER)
            && std::abs(ttScore) < SCORE_MATE_IN_MAX) {
            int singBeta = ttScore - (depth * Tune::SingMarginNum) / 10;
            int singDepth = (depth - 1) / 2;

            Move savedK0 = ss->killers[0], savedK1 = ss->killers[1];
            ss->excludedMove = m;
            int singScore = negamax(pos, ss, singBeta - 1, singBeta, singDepth, ply, accPly, cutNode);
            ss->excludedMove = MOVE_NONE;
            ss->killers[0] = savedK0;
            ss->killers[1] = savedK1;

            if (aborted) return 0;

            if (singScore < singBeta) {
                extension = 1;
                if (!isPv && singScore < singBeta - 25 && ss->doubleExtensions <= 7) {
                    extension = 2;
                    if (!is_tactical(m) && singScore < singBeta - 150) extension = 3;
                    ss->doubleExtensions++;
                }
            } else if (singBeta >= beta) {
                return singBeta;
            } else if (ttScore >= beta) {
                extension = -1;
            } else if (cutNode) {
                extension = -1;
            }
        }

        U64 nodesBefore = nodes;
        nodes++;
        movesSearched++;

        Piece movedPc = pos.piece_on(from_sq(m));
        ss->move = m;
        ss->movedPiece = movedPc;
        ss->contHist = tables.contHist[movedPc][to_sq(m)];
        ss->reduction = 0;
        (ss + 1)->doubleExtensions = ss->doubleExtensions;

        if (useNNUE) accStack.push_move(pos, m, accPly);
        StateInfo st;
        pos.do_move(m, st);
        tt.prefetch(pos.key());

        bool givesCheck = pos.in_check();
        int newDepth = depth - 1 + extension;
        int score;

        // Fractional Late Move Reductions (LMR) + PVS
        if (depth >= 3 && movesSearched > (1 + int(isRoot)) && (isQuiet || !ttPv)) {
            int r = lmrTable[std::min(depth, 63)][std::min(movesSearched, 63)];

            if (isQuiet) {
                if (ttPv) r -= 1024;
                if (!improving) r += 1024;
                if (cutNode) r += (ttPv ? 1024 : Tune::LmrCutNoPv);
                if (ttCapture) r += Tune::LmrTtCap;
                if (givesCheck) r -= Tune::LmrCheck;
                if (m == ss->killers[0] || m == ss->killers[1]) r -= Tune::LmrKiller;
                if (ss->cutoffCount > 3) r += 800;
                r -= std::clamp(histScore / Tune::LmrHistDiv, -2400, 2400);
            } else {
                r = r / 2;
                if (givesCheck) r -= Tune::LmrCheck;
                if (cutNode) r += 700;
                r -= std::clamp(capHistScore / Tune::LmrCapHistDiv, -1400, 1400);
            }

            int R = std::clamp(r / 1024, 0, newDepth - 1);
            int reducedDepth = newDepth - R;

            ss->reduction = R;
            score = -negamax(pos, ss + 1, -alpha - 1, -alpha, reducedDepth, ply + 1, accPly + 1, true);
            ss->reduction = 0;

            if (score > alpha && R > 0) {
                bool doDeeper = score > bestScore + 50 + 2 * newDepth;
                bool doShallower = score < bestScore + newDepth;
                int researchDepth = newDepth + int(doDeeper) - int(doShallower);
                if (researchDepth > reducedDepth) {
                    score = -negamax(pos, ss + 1, -alpha - 1, -alpha, researchDepth, ply + 1, accPly + 1, !cutNode);
                }
                // Continuation history feedback for LMR re-searches
                if (isQuiet && (ss - 1)->contHist) {
                    int b = (score <= alpha) ? -history_bonus(depth) / 2 : history_bonus(depth) / 2;
                    SearchTables::update_stat((ss - 1)->contHist[movedPc][to_sq(m)], b);
                }
            }
        } else if (!isPv || movesSearched > 1) {
            score = -negamax(pos, ss + 1, -alpha - 1, -alpha, newDepth, ply + 1, accPly + 1, !cutNode);
        } else {
            score = alpha + 1;
        }

        if (isPv && (movesSearched == 1 || score > alpha)) {
            score = -negamax(pos, ss + 1, -beta, -alpha, newDepth, ply + 1, accPly + 1, false);
        }

        pos.undo_move(m);

        if (aborted) return 0;

        if (isRoot) {
            rootMoveNodes[from_sq(m)][to_sq(m)] += (nodes - nodesBefore);
        }

        if (score > bestScore) {
            bestScore = score;
            if (score > alpha) {
                bestMove = m;
                alpha = score;

                ss->pv[0] = m;
                int childLen = (ss + 1)->pvLen;
                if (childLen > 0) {
                    std::memcpy(ss->pv + 1, (ss + 1)->pv, childLen * sizeof(Move));
                }
                ss->pvLen = childLen + 1;

                if (score >= beta) {
                    ss->cutoffCount++;
                    int statDepth = depth + int(score > beta + 80) + int(!inCheck && ss->staticEval <= origAlpha);
                    if (isQuiet) {
                        update_quiet_stats(pos, ss, m, quietsTried, quietCount, statDepth);
                    } else if (quietCount > 0) {
                        update_quiet_stats(pos, ss, MOVE_NONE, quietsTried, quietCount, statDepth);
                    }
                    update_cap_stats(pos, m, capsTried, capCount, statDepth);
                    // Penalize opponent's prior quiet move if we instantly refuted it on move 1
                    if (!isRoot && !isPv && movesSearched == 1 && depth >= 3
                        && (ss - 1)->move != MOVE_NONE && !is_tactical((ss - 1)->move)) {
                        Move pm = (ss - 1)->move;
                        Color them = ~pos.side_to_move();
                        SearchTables::update_stat(tables.mainHist[them][from_sq(pm)][to_sq(pm)][0], -history_bonus(depth) / 2);
                    }
                    break;
                }
            }
        }

        if (isQuiet) {
            if (quietCount < MAX_MOVES) quietsTried[quietCount++] = m;
        } else {
            if (capCount < MAX_MOVES) capsTried[capCount++] = m;
        }
    }

    if (movesSearched == 0) {
        if (excludedMove != MOVE_NONE) return alpha;
        return inCheck ? (-SCORE_MATE + ply) : SCORE_DRAW;
    }

    if (excludedMove == MOVE_NONE) {
        Bound bound = bestScore >= beta ? BOUND_LOWER : (bestScore > origAlpha ? BOUND_EXACT : BOUND_UPPER);
        tt.store(pos.key(), bestMove, TranspositionTable::score_to_tt(bestScore, ply),
                 rawEval, depth, bound, ttPv);

        if (!inCheck && (bestMove == MOVE_NONE || !is_tactical(bestMove))
            && !(bound == BOUND_LOWER && bestScore <= ss->staticEval)
            && !(bound == BOUND_UPPER && bestScore >= ss->staticEval)
            && std::abs(bestScore) < SCORE_MATE_IN_MAX) {
            tables.update_correction(pos, depth, bestScore - ss->staticEval,
                                     prevPc1, prevTo1, prevPc2, prevTo2);
        }
    }

    return bestScore;
}

SearchResult Searcher::search(Position& pos, const SearchLimits& limits, bool silent, std::mutex* outMtx) {
    startTime = std::chrono::steady_clock::now();
    stopFlag.store(false, std::memory_order_relaxed);
    aborted = false;
    silentMode = silent;
    nodes = 0;
    selDepth = 0;
    nodeLimit = limits.nodes;
    isInfinite = limits.infinite;
    std::memset(rootMoveNodes, 0, sizeof(rootMoveNodes));

    useNNUE = NNUE::is_loaded();
    if (useNNUE) accStack.reset_root(pos);

    tt.new_search();

    MoveList rootMoves;
    MoveGen::generate_legal(pos, rootMoves);
    SearchResult res;
    if (rootMoves.size() == 0) {
        return res;
    }
    res.bestMove = rootMoves.moves[0].move;

    hasTimeLimit = false;
    softLimitMs = 0;
    hardLimitMs = 0;
    int maxDepth = limits.depth > 0 ? std::min(limits.depth, MAX_PLY - 4) : (MAX_PLY - 4);

    if (limits.movetime > 0) {
        hasTimeLimit = true;
        int64_t safe = std::max<int64_t>(1, limits.movetime - moveOverhead);
        softLimitMs = safe;
        hardLimitMs = safe;
    } else if (limits.useTime) {
        Color us = pos.side_to_move();
        int64_t myTime = std::max<int64_t>(1, limits.time[us]);
        int64_t myInc  = std::max<int64_t>(0, limits.inc[us]);
        int64_t safeTime = myTime - moveOverhead;

        if (safeTime <= 10) {
            hasTimeLimit = true;
            softLimitMs = 1;
            hardLimitMs = 1;
        } else {
            hasTimeLimit = true;
            double base;
            if (limits.movestogo > 0) {
                double mtg = std::max(1, limits.movestogo);
                base = safeTime / (mtg + 1.5) + myInc * 0.78;
            } else {
                base = safeTime / double(Tune::TmDiv) + myInc * 0.80;
            }
            softLimitMs = std::max<int64_t>(1, std::min<int64_t>(int64_t(base * (Tune::TmSoftPct / 100.0)), int64_t(safeTime * 0.48)));
            hardLimitMs = std::max<int64_t>(softLimitMs, std::min<int64_t>(int64_t(base * (Tune::TmHardMult / 100.0)), int64_t(safeTime * 0.62)));
            hardLimitMs = std::min<int64_t>(hardLimitMs, safeTime);
        }
    }

    // Allocate padded SearchStack (6 entries before root so ss-6 is always valid!)
    SearchStack stackArr[MAX_PLY + 10];
    SearchStack* ss = stackArr + 6;

    int prevScore = 0;
    Move prevBestMove = MOVE_NONE;
    int bestMoveStability = 0;
    Move bestPv[MAX_PLY + 1];
    int bestPvLen = 1;
    bestPv[0] = res.bestMove;

    for (int depth = 1; depth <= maxDepth; ++depth) {
        selDepth = 0;
        int alpha = -SCORE_INFINITE;
        int beta  =  SCORE_INFINITE;
        int delta = 12;

        if (depth >= 4 && std::abs(prevScore) < SCORE_MATE_IN_MAX - 100) {
            alpha = std::max(-SCORE_INFINITE, prevScore - delta);
            beta  = std::min( SCORE_INFINITE, prevScore + delta);
        }

        int score = 0;
        int searchDepth = depth;
        while (true) {
            score = negamax(pos, ss, alpha, beta, searchDepth, 0, 0, false);

            if (aborted) break;

            if (score <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = (delta > 500) ? -SCORE_INFINITE : std::max(-SCORE_INFINITE, alpha - delta);
                searchDepth = depth;
                delta += delta / 2 + 4;
            } else if (score >= beta) {
                beta = (delta > 500) ? SCORE_INFINITE : std::min(SCORE_INFINITE, beta + delta);
                if (searchDepth > 1 && std::abs(score) < SCORE_MATE_IN_MAX) searchDepth--;
                delta += delta / 2 + 4;
            } else {
                break;
            }
        }

        if (aborted) {
            if (depth == 1) {
                print_info(1, 0, bestPv, bestPvLen, outMtx);
            }
            break;
        }

        if (ss->pvLen > 0) {
            res.bestMove = ss->pv[0];
            res.ponderMove = (ss->pvLen > 1) ? ss->pv[1] : MOVE_NONE;
            bestPvLen = ss->pvLen;
            std::memcpy(bestPv, ss->pv, bestPvLen * sizeof(Move));
        }
        res.score = score;
        res.depth = depth;
        res.nodes = nodes;

        print_info(depth, score, bestPv, bestPvLen, outMtx);

        if (limits.softNodes > 0 && nodes >= limits.softNodes) {
            break;
        }

        if (rootMoves.size() == 1 && limits.useTime && depth >= 3) {
            break;
        }

        if (hasTimeLimit && limits.useTime && depth >= 4) {
            if (res.bestMove == prevBestMove) {
                bestMoveStability = std::min(bestMoveStability + 1, 10);
            } else {
                bestMoveStability = 0;
            }

            double scale = 1.0;
            if (nodes > 0) {
                double frac = double(rootMoveNodes[from_sq(res.bestMove)][to_sq(res.bestMove)]) / double(nodes);
                scale *= std::clamp((1.50 - frac) * (Tune::TmNodeScale / 100.0), 0.55, 1.65);
            }
            static constexpr double stabScale[11] = {1.25, 1.12, 1.04, 0.98, 0.93, 0.89, 0.86, 0.83, 0.81, 0.79, 0.78};
            scale *= stabScale[bestMoveStability];

            if (depth >= 5 && score < prevScore - 15) {
                double drop = std::clamp(double(prevScore - score) / 80.0, 0.0, 0.60);
                scale *= (1.0 + drop);
            }

            int64_t adjSoft = std::min<int64_t>(hardLimitMs, int64_t(softLimitMs * scale));
            if (elapsed_ms() >= adjSoft) {
                break;
            }
        }

        prevScore = score;
        prevBestMove = res.bestMove;

        if (limits.useTime && depth >= 6 && std::abs(score) >= SCORE_MATE_IN_MAX) {
            int matePlies = SCORE_MATE - std::abs(score);
            if (depth >= matePlies + 2) break;
        }
    }

    res.nodes = nodes;
    return res;
}
