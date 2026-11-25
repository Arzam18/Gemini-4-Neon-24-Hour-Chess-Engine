#pragma once

#include "types.h"

namespace Zobrist {

extern U64 PieceKeys[PIECE_NB][SQ_NB];
extern U64 CastleKeys[16];
extern U64 EpKeys[8];
extern U64 SideKey;

void init();

} // namespace Zobrist
