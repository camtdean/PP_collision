#!/usr/bin/env python3
"""
patch_macro.py: sPHENIX's standard detectors/sPHENIX/Fun4All_G4_sPHENIX.C (macros commit 661f781, as TPCpp-10M) to
Fun4All_TPCpp.C: Pythia8 minimum-bias pp at 200 GeV with the Detroit tune and sPHENIX's pp beam parameters instead of
the single-particle gun, with SpacepointDumper after the reconstruction (its output file: the macro's outputFile).
The random seed is $TPCPP_SEED when set, else sPHENIX's default (/dev/urandom).

    python3 patch_macro.py <Fun4All_G4_sPHENIX.C> <Fun4All_TPCpp.C>

Every edit must match the source exactly once, else nothing is written.
"""
import sys

EDITS = [
    ("int Fun4All_G4_sPHENIX(", "int Fun4All_TPCpp("),
    ("recoConsts *rc = recoConsts::instance();",
     "recoConsts *rc = recoConsts::instance();\n"
     "  if (getenv(\"TPCPP_SEED\")) rc->set_IntFlag(\"RANDOMSEED\", std::stoi(getenv(\"TPCPP_SEED\")));"),
    ("Input::SIMPLE = true;", "Input::SIMPLE = false;"),
    ("// Input::PYTHIA8 = true;",
     "Input::PYTHIA8 = true;\n"
     "  PYTHIA8::config_file = std::string(getenv(\"CALIBRATIONROOT\")) + \"/Generators/phpythia8_detroit_minBias.cfg\";"),
    ("// Input::BEAM_CONFIGURATION = Input::AA_COLLISION;", "Input::BEAM_CONFIGURATION = Input::pp_COLLISION;"),
    ("#include <fun4all/Fun4AllDstOutputManager.h>",
     "#include <fun4all/Fun4AllDstOutputManager.h>\n"
     "#include <spacepointdumper/SpacepointDumper.h>\n"
     "R__LOAD_LIBRARY(libspacepointdumper.so)"),
    ("se->run(nEvents);",
     "se->registerSubsystem(new SpacepointDumper(\"SpacepointDumper\", outputFile));\n  se->run(nEvents);"),
]


def main():
    source, target = sys.argv[1:3]
    text = open(source).read()
    missing = [old for old, _ in EDITS if text.count(old) != 1]
    if missing:
        sys.exit(f"{source}: expected exactly one of each: {missing}")
    for old, new in EDITS:
        text = text.replace(old, new)
    open(target, "w").write(text)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
