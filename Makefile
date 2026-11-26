# Android ARM64 build for Gemini 4 Neon.
# The normal Windows x86-64-v3 build documented in README.md is unchanged.

ANDROID_API ?= 24
ANDROID_NDK_HOME ?= $(ANDROID_NDK_LATEST_HOME)
ANDROID_NDK_BIN := $(ANDROID_NDK_HOME)/toolchains/llvm/prebuilt/linux-x86_64/bin
ANDROID_CXX ?= $(ANDROID_NDK_BIN)/aarch64-linux-android$(ANDROID_API)-clang++

CXXFLAGS_COMMON := -std=c++20 -O3 -march=armv8-a+simd -DNDEBUG -pthread \
	-DUSE_EMBEDDED_NNUE -DNNUE_KING_BUCKETS=4 -DNNUE_HL=768 -DNNUE_PAIRWISE=0 \
	-Isource/src
SOURCES := source/src/bitboard.cpp source/src/zobrist.cpp source/src/position.cpp \
	source/src/movegen.cpp source/src/tt.cpp source/src/eval.cpp \
	source/src/nnue.cpp source/src/search.cpp source/src/datagen.cpp \
	source/src/uci.cpp source/src/main.cpp source/src/nnue_embed.S
NETWORK := source/data/gen5_4x768_65m.nnue
OUTPUT := release/Gemini4Neonchess24hrs-arm64-android-neon

.PHONY: android-arm64
android-arm64:
	@test -n "$(ANDROID_NDK_HOME)" || (echo "ANDROID_NDK_HOME is not set" >&2; exit 1)
	@test -x "$(ANDROID_CXX)" || (echo "Android compiler not found: $(ANDROID_CXX)" >&2; exit 1)
	@test -s "$(NETWORK)" || (echo "Missing champion NNUE network: $(NETWORK)" >&2; exit 1)
	@mkdir -p source/data release
	@cp -f "$(NETWORK)" source/data/current.nnue
	$(ANDROID_CXX) $(CXXFLAGS_COMMON) $(SOURCES) -fPIE -pie -o $(OUTPUT)
