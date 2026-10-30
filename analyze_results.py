"""
Is any difference between your 9 DBs real, or just noise from only ~95 questions?

Reads retrieval_results_per_question.csv (made by the evaluator) and prints:
  1. each DB with a 95% bootstrap confidence interval
  2. chunking strategy vs embedding model (averaged over the other factor) with paired significance
  3. how many questions NO DB found (a ceiling caused by labels/corpus, not by retrieval)
  4. recall by the question's labelled source document
and writes never_found_questions.csv so you can check those labels by hand.

Run:  python analyze_results.py --csv retrieval_results_per_question.csv
"""
import argparse
import itertools

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="retrieval_results_per_question.csv")
ap.add_argument("--metric", default="recall@5", help="per-question metric used for the tests (recall@5, recall@10, mrr@10)")
ap.add_argument("--boot", type=int, default=5000)
args = ap.parse_args()

df = pd.read_csv(args.csv)
parts = df["collection"].str.split(r"[\\/]", regex=True)
df["embedding"], df["chunking"] = parts.str[0], parts.str[1]
rng = np.random.default_rng(0)


def ci(v, B=args.boot):
    v = np.asarray(v, float)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def matrix(metric):  # rows = questions, columns = DBs
    return df.pivot(index="question", columns="collection", values=metric)


M, R = matrix(args.metric), matrix("mrr@10")
n = len(M)
print(f"{n} questions, {M.shape[1]} DBs.  1 question = {1 / n:.3f} of recall.\n")

# 1. per-DB confidence intervals --------------------------------------------------
print(f"1) Each DB, {args.metric} and MRR@10 with 95% bootstrap CI")
rows = []
for c in M.columns:
    lo, hi = ci(M[c]); mlo, mhi = ci(R[c])
    rows.append((c, M[c].mean(), f"[{lo:.2f}, {hi:.2f}]", R[c].mean(), f"[{mlo:.2f}, {mhi:.2f}]"))
t = pd.DataFrame(rows, columns=["db", args.metric, "CI", "mrr@10", "mrr CI"]).sort_values(args.metric, ascending=False)
print(t.round(3).to_string(index=False))
print("   -> If the intervals of two DBs overlap heavily, you cannot say one is better.\n")


# 2. factor effects with paired tests ----------------------------------------------
def factor(name):
    groups = {g: [c for c in M.columns if df.loc[df.collection == c, name].iloc[0] == g] for g in df[name].unique()}
    per_q = {g: M[cols].mean(axis=1) for g, cols in groups.items()}
    mrr_q = {g: R[cols].mean(axis=1) for g, cols in groups.items()}
    print(f"2) Effect of {name} (each value averaged over the 3 DBs of the other factor)")
    for g in per_q:
        lo, hi = ci(per_q[g])
        print(f"   {g:35s} {args.metric}={per_q[g].mean():.3f} [{lo:.2f}, {hi:.2f}]  mrr@10={mrr_q[g].mean():.3f}")
    for a, b in itertools.combinations(per_q, 2):
        d = per_q[a] - per_q[b]
        lo, hi = ci(d)
        verdict = "DIFFERENT" if (lo > 0 or hi < 0) else "not significant"
        print(f"   {a} - {b}: {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
    print()


factor("chunking")
factor("embedding")

# 3. ceiling -------------------------------------------------------------------------
F = matrix("first_rel_rank")
never = (F == 0).all(axis=1)
any_found = (F > 0).any(axis=1)
print("3) Ceiling check")
print(f"   found by at least one DB : {int(any_found.sum())} / {n}  ({any_found.mean():.0%})")
print(f"   found by ALL 9 DBs        : {int((F > 0).all(axis=1).sum())} / {n}")
print(f"   found by NO DB            : {int(never.sum())} / {n}  ({never.mean():.0%})")
print("   -> Questions that no embedding model and no chunking strategy can retrieve are usually a label,")
print("      document-version or judge problem. Check never_found_questions.csv before trusting the scores.\n")

first_db = M.columns[0]
ref = df[df.collection == first_db].set_index("question")
cols = [c for c in ["labelled_source", "expected_answer", "top1_source", "top1_section", "top1_text"] if c in ref.columns]
ref.loc[never[never].index, cols].reset_index().to_csv("never_found_questions.csv", index=False)
print(f"   wrote never_found_questions.csv ({int(never.sum())} rows; top-1 chunk is from {first_db})\n")

# 4. by labelled source ------------------------------------------------------------------
if "labelled_source" in df.columns:
    print(f"4) {args.metric} by the question's labelled source document (mean over the 9 DBs)")
    g = df.groupby("labelled_source")[args.metric].agg(["mean", "count"])
    g["questions"] = (g["count"] / M.shape[1]).round(0).astype(int)
    print(g[["mean", "questions"]].round(3).to_string())
