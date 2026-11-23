"""LLM as communicator: word the facts from 04 in plain English. The LLM adds no facts."""
import sys, json, os, re
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(ROOT))

from src.config import load_config

cfg = load_config()
reports = cfg["paths"]["reports"]
SEED, N_REPEATS = 0, 5                                     # must match 04_analysis_interp.py

# load what 04 saved
need = ["permutation_importance.csv", "permutation_importance_grouped.csv", "descriptor_clusters.csv"]
missing = [f for f in need if not (reports / f).exists()]
if missing:
    sys.exit(f"missing {missing}: run 04_analysis_interp first")
summary = pd.read_csv(reports / need[0], index_col=0).sort_values("RMSE_increase", ascending=False)
summary_grouped = pd.read_csv(reports / need[1], index_col=0).sort_values("RMSE_increase", ascending=False)
clusters = pd.read_csv(reports / need[2], index_col=0)
cluster, rep_of = clusters["cluster"], clusters["representative"]

imp_all = summary.RMSE_increase
sd_all = summary["fold_SD"] if "fold_SD" in summary.columns else None   # CSVs from the crashed run lack it

# hypotheses (ML only): statement, rule fixed in advance, numbers, verdict 
pos = imp_all.clip(lower=0)
share = lambda k: pos.nlargest(k).sum() / pos.sum()
top2, top10 = imp_all.index[:2].tolist(), imp_all.index[:10]
n_desc_top10 = int(top10.str.startswith("d_").sum())
sd_text = f" ± {sd_all.iloc[0]:.3f} over folds" if sd_all is not None else " (fold SD not saved)"
h4 = []
for c in top2:
    c_rep = rep_of.get(c, c)                                   # fingerprint bits have no cluster
    n_copies = int((cluster == cluster[c]).sum() - 1) if c in cluster.index else 0
    h4.append((c, n_copies, imp_all[c], summary_grouped. RMSE_increase[c_rep]))

hyp = [
    ("H1", "The model relies on very few columns", "top-2 columns hold at least 50% of the total importance",
     f"top-1 {share(1):.0%}, top-2 {share(2):.0%}, top-10 {share(10):.0%}; top column {top2[0]} "
     f"{imp_all.iloc[0]:.3f}{sd_text}", share(2) >= 0.5),
    ("H2", "Most columns are not used", "at least 50% of columns have importance <= 0",
     f"{(imp_all <= 0).mean():.0%} of {len(imp_all)} columns are <= 0; {(imp_all > 0.001).sum()} are above 0.001",
     (imp_all <= 0).mean() >= 0.5),
    ("H3", "The most important columns are mainly descriptors, not fingerprint bits",
     "at least 6 of the top-10 columns are descriptors",
     f"{n_desc_top10} of the top-10 are descriptors, {10 - n_desc_top10} are fingerprint bits", n_desc_top10 >= 6),
    ("H4", "Near-copies were hiding some importance of the top columns",
     "a top column has near-copies and its importance rises when they are collapsed",
     "; ".join(f"{c}: {n} near-copies, {a:.3f} alone -> {g:.3f} collapsed" for c, n, a, g in h4),
     any(n > 0 and g > a for _, n, a, g in h4)),
    ("H5", "The top columns stay on top after grouping", "the top-2 columns (or their representatives) are still the top-2",
     f"before {top2}, after {summary_grouped.index[:2].tolist()}",
     {rep_of.get(c, c) for c in top2} <= set(summary_grouped.index[:2])),
]
print()
for hid, text, rule, evidence, ok in hyp:
    print(f"{hid} {text}: {'SUPPORTED' if ok else 'NOT SUPPORTED'}\n   {evidence}")

# LLM as communicator: every fact comes from the code above; the LLM only words it 
facts = {
    "setup": {"model": "HGB_both", "columns": str(len(imp_all)),
              "evaluation": "scaffold folds; importance measured on test molecules only",
              "folds": str(cfg["cv"]["n_splits"]), "repeats_per_column": str(N_REPEATS), "seed": str(SEED),
              "importance_meaning": "increase in RMSE when one column is shuffled"},
    "hypotheses": [{"id": hid, "statement": text, "rule": rule, "numbers": evidence,
                    "verdict": "supported" if ok else "not supported"} for hid, text, rule, evidence, ok in hyp],
    "limits": ["Importance shows what the model uses, not what causes the target.",
               "Correlated columns share credit, so a single column's number understates it.",
               "H3 counts the top-10 columns; it does not compare the whole descriptor and fingerprint blocks "
               "(that comparison is the cross-validation ablation).",
               "One seed and five scaffold folds; spread over folds is reported, spread over seeds is not.",
               "Verdict rules (such as 50% or 6 of 10) were fixed by the author in advance and are not statistical tests."]}
facts_file = reports / "xai_facts.json"
facts_file.write_text(json.dumps(facts, indent=1))
print(f"\nfacts written to {facts_file}")

SYSTEM = ("You are a communicator, not an analyst. Turn the JSON into a short plain-English summary for a reader who "
          "knows basic machine learning. Use ONLY facts in the JSON and copy numbers exactly as written. Column names "
          "are labels only: do not say what they mean or why they matter, and make no causal claims. Format: a "
          "2-sentence overview; then one line per hypothesis (id, verdict, the key number); then one sentence on "
          "limits taken from the JSON. If a verdict is 'not supported', say so plainly.")


def ask_llm(facts):
    """Gemini via the google-genai SDK. Key and model come from .env; returns None if unavailable."""
    try:
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types
        load_dotenv(ROOT / ".env")
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        r = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
            contents=json.dumps(facts, indent=1),
            config=types.GenerateContentConfig(system_instruction=SYSTEM, max_output_tokens=1000, temperature=0))
        return r.text
    except Exception as e:
        print(f"\n(LLM call skipped: {type(e).__name__}: {e}. Paste this instruction plus "
              f"{facts_file} into any chat.)\n{SYSTEM}")


def numbers_not_in_facts(text, facts):                   # LLM will only quote numbers that are in the facts
    nums = lambda t: set(re.findall(r"\d+\.?\d*", t))
    return sorted(nums(text) - nums(json.dumps(facts)))


text = ask_llm(facts)
if text:
    print("\nLLM wording:\n" + text)
    extra = numbers_not_in_facts(text, facts)
    print("number check:", "OK" if not extra else f"WARNING, numbers not in the facts: {extra}")
    (reports / "xai_summary.txt").write_text(text)
