#pragma once

namespace Tune {

#ifdef SPSA_TUNE
#define TUNE_VAR inline int
#else
#define TUNE_VAR inline constexpr int
#endif

TUNE_VAR EvalMatBase   = 22709;
TUNE_VAR LmrBase       = 72;
TUNE_VAR LmrDiv        = 228;
TUNE_VAR LmrHistDiv    = 6;
TUNE_VAR RfpImp        = 57;
TUNE_VAR RfpNonImp     = 80;
TUNE_VAR RazorBase     = 243;
TUNE_VAR RazorMult     = 211;
TUNE_VAR NmpEvalDiv    = 174;
TUNE_VAR ProbCutMargin = 186;
TUNE_VAR FutBase       = 108;
TUNE_VAR FutMult       = 87;
TUNE_VAR HistPruneMult = 2363;
TUNE_VAR SeeQuietMult  = 20;
TUNE_VAR SeeCapMult    = 85;
TUNE_VAR SingMarginNum = 16;
TUNE_VAR TmDiv         = 22;
TUNE_VAR TmSoftPct     = 73;

// Round 2 SPSA parameters (kept at proven baseline after R2 verification was neutral)
TUNE_VAR LmrCutNoPv    = 1800;
TUNE_VAR LmrTtCap      = 900;
TUNE_VAR LmrCheck      = 900;
TUNE_VAR LmrKiller     = 900;
TUNE_VAR LmrCapHistDiv = 10;
TUNE_VAR QsFutMargin   = 140;
TUNE_VAR CorrPawnW     = 24;
TUNE_VAR CorrNonPawnW  = 20;
TUNE_VAR CorrMinMajW   = 16;
TUNE_VAR CorrContW     = 20;
TUNE_VAR TmHardMult    = 240;
TUNE_VAR TmNodeScale   = 135;

#undef TUNE_VAR

} // namespace Tune
