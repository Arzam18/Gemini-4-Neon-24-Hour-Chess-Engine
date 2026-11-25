#pragma once

#include <cstdint>

// Compact 32-byte training position (Marlinformat-style)
struct PackedPos {
    uint64_t occ;        // bitboard of all occupied squares (<= 32 bits set)
    uint8_t  pieces[16]; // 4 bits per piece in LSB order of occ (0..11 = W_PAWN..B_KING)
    uint8_t  stm;        // 0 = WHITE, 1 = BLACK
    uint8_t  wdl;        // 0 = Black win, 1 = Draw, 2 = White win (White POV)
    int16_t  score;      // Search score in centipawns (White POV)
    uint16_t ply;        // Game ply
    uint16_t reserved;   // Padding to 32 bytes
};

static_assert(sizeof(PackedPos) == 32, "PackedPos must be 32 bytes");
