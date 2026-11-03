"""Generate evals/adversarial_variants.yaml from the adversarial.yaml seeds.

Offline, one-shot generation: CI never runs this — the output is a
checked-in YAML file, diff-reviewed like a lockfile before committing.
The deterministic mutations come from mutations.MUTATIONS; each attack
seed also gets a split_payload case (half the payload in the user task,
half in a poisoned tool observation). Benign near-miss controls are
hand-seeded in near_miss_controls.yaml and copied in verbatim.

Every variant carries seed_id + mutation, so failures group by mutation
class, and expect_patterns computed from the detector itself at
generation time (the patterns the mutated text actually trips — empty
when it trips none, which is exactly the detector-gap signal).

Dedup: exact hash on whitespace-collapsed text across all variants
and against the seeds, then a token-set Jaccard >= 0.9 near-dup filter
per seed. The hash keeps case (case is the entire mutation for
all_caps, so lowercasing would dedup it away against its own seed)
and split cases are keyed with their field placement (placement is
the entire mutation there — the rejoined text equals the seed's).

    python3 evals/generate_variants.py [--llm]

--llm additionally asks the free-tier LLM (LLM_API_KEY, see
llm_client.py) for 3 paraphrases and translations into
TRANSLATE_LANGS for every attack seed. LLM output is tagged
real_only: true and the mock runner skips it — those cases exist for
the real-backend runs in a later phase.
"""
import argparse
import hashlib
import os
import sys
from datetime import date

import yaml

sys.path.insert(0, ".")
sys.path.insert(0, "evals")
import mutations
from guardrails import detect_injection

GENERATOR_VERSION = 1
SEEDS_PATH = "evals/adversarial.yaml"
CONTROLS_PATH = "evals/near_miss_controls.yaml"
OUT_PATH = "evals/adversarial_variants.yaml"
TRANSLATE_LANGS = ["Hindi", "Bengali", "Swahili"]
# benign carrier for split variants of task seeds: an ordinary question
# that makes the agent call the poisoned tool, the way a real smuggled
# instruction would need it to
SPLIT_CARRIER_QUESTION = "What is the refund window?"
SPLIT_CARRIER_TOOL = "search_docs"
NEAR_DUP_JACCARD = 0.9


def _pattern_names(text):
    return [name for name, _ in detect_injection(text)]


def _collapse(text):
    return " ".join((text or "").split())


def _tokens(text):
    return set(_collapse(text).split())


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _mutated_cases(seed):
    """All deterministic variants of one attack seed (pre-dedup)."""
    out = []
    if seed["type"] == "task":
        payload = seed["attack"]
    else:  # observation — the payload is the poison string
        payload = seed["poison"]
    for name, fn in mutations.MUTATIONS.items():
        mutated = fn(payload)
        case = {"id": f"{seed['id']}__{name}", "seed_id": seed["id"],
                "mutation": name,
                "expect_patterns": _pattern_names(mutated)}
        if seed["type"] == "task":
            case.update({"type": "task", "attack": mutated})
        else:
            case.update({"type": "observation",
                         "question": seed["question"],
                         "poison_tool": seed["poison_tool"],
                         "poison": mutated})
        out.append(case)
    first, second = mutations.split_payload(payload)
    case = {"id": f"{seed['id']}__split_payload", "seed_id": seed["id"],
            "mutation": "split_payload", "type": "split",
            "fragment_task": first, "poison": second,
            "expect_patterns": sorted(set(_pattern_names(first))
                                      | set(_pattern_names(second)))}
    if seed["type"] == "task":
        case.update({"question": SPLIT_CARRIER_QUESTION,
                     "poison_tool": SPLIT_CARRIER_TOOL})
    else:
        case.update({"question": seed["question"],
                     "poison_tool": seed["poison_tool"]})
    out.append(case)
    return out


def _llm_cases(seed, complete):
    """Paraphrase + translation variants of one attack seed (real_only)."""
    out = []
    payload = seed["attack"] if seed["type"] == "task" else seed["poison"]

    def _wrap(case_id, mutation, text):
        case = {"id": case_id, "seed_id": seed["id"],
                "mutation": mutation, "real_only": True,
                "expect_patterns": _pattern_names(text)}
        if seed["type"] == "task":
            case.update({"type": "task", "attack": text})
        else:
            case.update({"type": "observation",
                         "question": seed["question"],
                         "poison_tool": seed["poison_tool"],
                         "poison": text})
        return case

    for i, text in enumerate(mutations.paraphrase_with(payload, complete),
                             start=1):
        out.append(_wrap(f"{seed['id']}__paraphrase_{i}",
                         f"paraphrase_{i}", text))
    for lang, text in mutations.translate_with(
            payload, complete, TRANSLATE_LANGS).items():
        tag = lang.lower()
        out.append(_wrap(f"{seed['id']}__translate_{tag}",
                         f"translate_{tag}", text))
    return out


def _case_text(case):
    """The payload text a case is deduped on."""
    if case["type"] == "task":
        return case["attack"]
    if case["type"] == "observation":
        return case["poison"]
    if case["type"] == "split":
        return case["fragment_task"] + " " + case["poison"]
    return case.get("question", "")


def _dedup_key(case):
    # whitespace-collapsed payload text, case preserved (see module
    # docstring); split cases carry their placement in the key
    key = _collapse(_case_text(case))
    if case["type"] == "split":
        key += " #split"
    return key


def _dedup(candidates, seed_texts):
    """Exact-hash dedup (vs seeds and kept variants), then a per-seed
    Jaccard near-dup filter. Returns (kept, dropped_count)."""
    seen = {_collapse(t) for t in seed_texts}
    kept_by_seed = {}
    kept, dropped = [], 0
    for case in candidates:
        text = _case_text(case)
        if _dedup_key(case) in seen:
            dropped += 1
            continue
        toks = _tokens(text)
        siblings = kept_by_seed.setdefault(case["seed_id"], [])
        if any(_jaccard(toks, s) >= NEAR_DUP_JACCARD for s in siblings):
            dropped += 1
            continue
        seen.add(_dedup_key(case))
        siblings.append(toks)
        kept.append(case)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true",
                    help="also generate paraphrase/translation variants "
                         "via the free-tier LLM (needs LLM_API_KEY); "
                         "output is tagged real_only")
    args = ap.parse_args()

    seed_bytes = open(SEEDS_PATH, "rb").read()
    seeds = yaml.safe_load(seed_bytes)["cases"]
    attacks = [s for s in seeds if s["type"] in ("task", "observation")]
    seed_texts = [s["attack"] if s["type"] == "task" else s["poison"]
                  for s in attacks]

    candidates = []
    for seed in attacks:
        candidates.extend(_mutated_cases(seed))

    if args.llm:
        from llm_client import FreeLLMClient
        client = FreeLLMClient.from_env()
        if client is None:
            print("--llm needs LLM_API_KEY in the environment")
            sys.exit(1)
        print(f"generating LLM variants via {client.label} "
              "(tagged real_only)")
        complete = lambda p: client.generate(p, temperature=0.7)
        for seed in attacks:
            candidates.extend(_llm_cases(seed, complete))

    variants, dropped = _dedup(candidates, seed_texts)

    controls = yaml.safe_load(open(CONTROLS_PATH))["cases"]
    for case in controls:
        case = dict(case, seed_id=case["id"], mutation="hand_seeded")
        variants.append(case)

    header = {"generator": "generate_variants.py",
              "generator_version": GENERATOR_VERSION,
              "generated": date.today().isoformat(),
              "seed_file": os.path.basename(SEEDS_PATH),
              "seed_sha256": hashlib.sha256(seed_bytes).hexdigest(),
              "llm_variants": bool(args.llm)}
    with open(OUT_PATH, "w") as f:
        f.write("# generated by evals/generate_variants.py — do not "
                "hand-edit;\n# regenerate from adversarial.yaml + "
                "near_miss_controls.yaml instead.\n")
        yaml.safe_dump({"header": header, "cases": variants}, f,
                       allow_unicode=True, sort_keys=False, width=4096)

    by_mutation = {}
    for case in variants:
        by_mutation[case["mutation"]] = by_mutation.get(
            case["mutation"], 0) + 1
    print(f"wrote {OUT_PATH}: {len(variants)} cases "
          f"({dropped} duplicates dropped)")
    for name in sorted(by_mutation):
        print(f"  {name:18s} {by_mutation[name]}")
    real_only = sum(1 for c in variants if c.get("real_only"))
    if real_only:
        print(f"  ({real_only} tagged real_only — skipped on mocks)")


if __name__ == "__main__":
    main()
