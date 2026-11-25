#pragma once

#include "position.h"

namespace MoveGen {

void generate_captures(const Position& pos, MoveList& list);
void generate_quiets(const Position& pos, MoveList& list);
void generate_all(const Position& pos, MoveList& list);
void generate_legal(const Position& pos, MoveList& list);

} // namespace MoveGen
