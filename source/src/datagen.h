#pragma once

#include <string>
#include <cstdint>

namespace Datagen {

void run(const std::string& outPath, uint64_t targetPositions,
         int numThreads, uint64_t softNodes, uint64_t seed);

} // namespace Datagen
