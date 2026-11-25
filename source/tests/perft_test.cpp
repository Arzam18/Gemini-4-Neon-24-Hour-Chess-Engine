#include "bitboard.h"
#include "zobrist.h"
#include "position.h"
#include "movegen.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

U64 perft(Position& pos, int depth) {
    MoveList ml;
    MoveGen::generate_legal(pos, ml);
    if (depth == 1) return ml.size();
    if (depth == 0) return 1;

    U64 nodes = 0;
    for (int i = 0; i < ml.size(); ++i) {
        StateInfo st;
        pos.do_move(ml.moves[i].move, st);
        nodes += perft(pos, depth - 1);
        pos.undo_move(ml.moves[i].move);
    }
    return nodes;
}

int main(int argc, char** argv) {
    Bitboards::init();
    Zobrist::init();

    std::string epdPath = (argc > 1) ? argv[1] : "resources/perft/perft.epd";
    int maxDepth = (argc > 2) ? std::stoi(argv[2]) : 5;

    std::ifstream in(epdPath);
    if (!in) {
        std::cerr << "Cannot open " << epdPath << "\n";
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    std::string line;
    int lineNum = 0, passed = 0, failed = 0;
    U64 totalNodes = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        lineNum++;

        size_t semi = line.find(';');
        std::string fen = line.substr(0, semi);
        while (!fen.empty() && fen.back() == ' ') fen.pop_back();

        Position pos;
        StateInfo rootSt;
        pos.set_fen(fen, &rootSt);

        std::string rest = line.substr(semi);
        std::istringstream ss(rest);
        std::string token;
        bool ok = true;

        while (ss >> token) {
            if (token.size() >= 3 && token[0] == ';' && token[1] == 'D') {
                int d = std::stoi(token.substr(2));
                U64 expected = 0;
                ss >> expected;
                if (d <= maxDepth) {
                    U64 got = perft(pos, d);
                    totalNodes += got;
                    if (got != expected) {
                        std::cerr << "FAIL line " << lineNum << " D" << d
                                  << " FEN: " << fen
                                  << " expected=" << expected << " got=" << got << "\n";
                        ok = false;
                        break;
                    }
                }
            }
        }
        if (ok) passed++;
        else failed++;
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Perft suite (up to D" << maxDepth << "): " << passed << "/" << lineNum
              << " passed, " << failed << " failed. Total nodes=" << totalNodes
              << " in " << secs << "s (" << U64(totalNodes / std::max(secs, 1e-6)) << " NPS)\n";
    return failed == 0 ? 0 : 1;
}
