#!/usr/bin/env python3
# usage: cd examples/game_2048 && uv run python run_arm.py <A|B|C> <rep> [num_generations]
# Arms (only reflection flags change; everything else fixed):
#   A: use_reflection=False                       (baseline)
#   B: use_reflection=True, grounding=False       (ungrounded reflection)
#   C: use_reflection=True, grounding=True        (grounded reflection)
import sys
import os
import copy
import importlib
from shinka.core import ShinkaEvolveRunner

# --- provider/model override (user has OPENAI_API_KEY [+ optional ANTHROPIC_API_KEY]) ---
# Single editor model -> reflector reuses it (capability held constant, D8).
# These require ANTHROPIC_API_KEY (opus) and OPENAI_API_KEY (gpt-5.5).
EDITOR_MODELS = ["claude-opus-4-8", "gpt-5.5"]


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_arm.py <A|B|C> <rep> [num_generations]")
    arm = sys.argv[1].upper()
    rep = sys.argv[2]
    ngen = int(sys.argv[3]) if len(sys.argv) > 3 else None

    base = importlib.import_module("run_evo")  # the task's existing config module
    evo = copy.deepcopy(base.evo_config)
    db = copy.deepcopy(base.db_config)
    job = copy.deepcopy(base.job_config)

    # Hold everything fixed; flip ONLY the reflection flags.
    evo.use_text_feedback = True
    evo.reflection_llm_models = None  # None -> reuse editor llm_models (capability held constant)
    if arm == "A":
        evo.use_reflection = False
    elif arm == "B":
        evo.use_reflection = True
        evo.reflection_grounding = False
    elif arm == "C":
        evo.use_reflection = True
        evo.reflection_grounding = True
    else:
        raise SystemExit(f"unknown arm {arm!r} (use A, B, or C)")

    # Restrict to available providers (overrides base run_evo.py's Gemini/Bedrock models).
    evo.llm_models = list(EDITOR_MODELS)
    evo.meta_llm_models = ["gpt-5-nano"]      # OpenAI
    evo.novelty_llm_models = ["gpt-5-nano"]   # OpenAI
    evo.embedding_model = "text-embedding-3-small"  # OpenAI
    # reflection_llm_models stays None -> reuses evo.llm_models (the editor model)

    if ngen is not None:
        evo.num_generations = ngen

    run_dir = os.path.abspath(os.path.join("results_ablation", f"{arm}_rep{rep}"))
    os.makedirs(run_dir, exist_ok=True)
    evo.results_dir = run_dir
    db.db_path = os.path.join(run_dir, "evolution_db.sqlite")

    print(
        f"[{arm} rep{rep}] gens={evo.num_generations} "
        f"use_reflection={evo.use_reflection} "
        f"grounding={getattr(evo, 'reflection_grounding', None)} -> {run_dir}"
    )

    ShinkaEvolveRunner(
        evo_config=evo,
        job_config=job,
        db_config=db,
        max_evaluation_jobs=base.MAX_EVALUATION_JOBS,
        max_proposal_jobs=base.MAX_PROPOSAL_JOBS,
        max_db_workers=base.MAX_DB_WORKERS,
        verbose=True,
    ).run()


if __name__ == "__main__":
    main()
