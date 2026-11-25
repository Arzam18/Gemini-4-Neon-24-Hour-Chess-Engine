#include "zobrist.h"

namespace Zobrist {

U64 PieceKeys[PIECE_NB][SQ_NB];
U64 CastleKeys[16];
U64 EpKeys[8];
U64 SideKey;

namespace {
U64 splitmix64(U64& state) {
    U64 z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
} // namespace

void init() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    U64 seed = 0x123456789ABCDEF0ULL;
    for (int p = 0; p < PIECE_NB; ++p)
        for (int s = 0; s < SQ_NB; ++s)
            PieceKeys[p][s] = splitmix64(seed);

    for (int i = 0; i < 16; ++i)
        CastleKeys[i] = splitmix64(seed);

    for (int i = 0; i < 8; ++i)
        EpKeys[i] = splitmix64(seed);

    SideKey = splitmix64(seed);
}

} // namespace Zobrist
