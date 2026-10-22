#!/bin/bash
#SBATCH -J pc-eval  # name of job
#SBATCH -p gpu-ms,gpu-troja        # name of partition or queue (if not specified default partition is used)
#SBATCH --cpus-per-task=16                 # number of cores/threads per task (default 1)
#SBATCH --gres=gpu:1                       # number of GPUs to request (default 0)
#SBATCH --mem=50G                         # request 50 gigabytes memory (per node, default depends on node)
#SBATCH --time 2-00:00:00 # time (D-HH:MM:SS)
#SBATCH -o pc-eval.%A_%a.%N.out  # name of output file for this submission script
#SBATCH -e pc-eval.%A_%a.%N.err
#
# The vX.4 models on the test pages of the LINDAT record (atrium-project#15), one run after another.
#
# Each model is scored on the test partition of the fold it was trained on
# (model_registry.REVISION_BEST_FOLDS), its pages extracted from the LINDAT zips; then on the 5,449-page
# set and on small_data_samples; then the --best ensemble on small_data_samples. The summary gives top-1
# and top-3 accuracy, and top-1 for pages in training, of seen documents and of unseen documents
# (data_scripts/eval_by_document_overlap.py).
#
#   bash supplementary/scripts/eval_vX4_job.sh prefetch   # login node, online: weights, Hub commits, checks
#   sbatch supplementary/scripts/eval_vX4_job.sh          # GPU node, offline
#
# Paths can be set by environment variable: ATRIUM, REPO, FOLDS_CSV, CUT_CSV, ZIPS, SET_5449, STAGE.
# KEEP_STAGED=1 keeps the extracted test pages (one fold, ~8 GB, is on disk at a time otherwise).

hostname
date

ATRIUM=${ATRIUM:-/lnet/work/projects/atrium}
REPO=${REPO:-$ATRIUM/GITHUB/atrium-page-classification}
PAGES=$ATRIUM/page_util/pages
FOLDS_CSV=${FOLDS_CSV:-$PAGES/licensed_archives/licensed_crossval_folds.csv}
CUT_CSV=${CUT_CSV:-$PAGES/licensed_crossval_folds_CUT.csv}
ZIPS=${ZIPS:-$PAGES/licensed_archives}
SET_5449=${SET_5449:-$PAGES/test_F12345}
STAGE=${STAGE:-$PAGES/stage_vX4}

export HF_HOME=$ATRIUM/cache/
export TORCH_HOME=$ATRIUM/cache/
export PYTHONUNBUFFERED=1      # progress shows up in the .out file as it happens

# shellcheck source=/dev/null
source "$REPO/venv/bin/activate"
cd "$REPO" || exit 1

STAMP=$(date +%Y%m%d-%H%M)
RESULTS=$(python3 -c 'import configparser; c = configparser.ConfigParser(); c.read("setup/config.txt"); print(c.get("OUTPUT", "FOLDER_RESULTS"))')
STATS=$RESULTS/stats
mkdir -p "$STATS"
# revision:fold pairs, from the one place they are declared
mapfile -t PAIRS < <(python3 -c 'from model_registry import REVISION_BEST_FOLDS as F; print("\n".join(f"{r}:{f}" for r, f in F.items()))')
mapfile -t FOLDS < <(printf '%s\n' "${PAIRS[@]}" | cut -d: -f2 | sort -u)

# ── prefetch: on the login node, with network ──────────────────────────────────────────────────────
if [[ "${1:-}" == "prefetch" ]]; then
    # run.py --hf saves each Hub revision to model/model_vN4 and caches its base model and processor
    for pair in "${PAIRS[@]}"; do
        python3 run.py --hf --no-train --no-eval -rev "${pair%%:*}" || exit 1
    done

    if ! python3 - "$STATS/${STAMP}_vX4_hub_revisions.csv" <<'PY'
import configparser, csv, sys
from huggingface_hub import HfApi
from model_registry import REVISION_BEST_FOLDS, REVISION_TO_BASE_MODEL

config = configparser.ConfigParser()
config.read("setup/config.txt")
repo = config.get("HF", "repo_name")
branches = {ref.name: ref.target_commit for ref in HfApi().list_repo_refs(repo).branches}
with open(sys.argv[1], "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["revision", "base_model", "fold", "hub_repo", "hub_commit"])
    for revision, fold in REVISION_BEST_FOLDS.items():
        writer.writerow([revision, REVISION_TO_BASE_MODEL[revision], fold, repo, branches.get(revision, "")])
print(open(sys.argv[1], encoding="utf-8").read())
PY
    then
        echo "WARNING: the Hub commits of the revisions were not recorded (no network?)" >&2
    fi

    # the published folds against the ones the vX.4 models were trained with
    python3 - "$FOLDS_CSV" "$CUT_CSV" "${FOLDS[@]}" <<'PY'
import csv, sys
from pathlib import Path

published, cut, folds = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
if not cut.is_file():
    print(f"{cut}: not found, folds not compared")
    sys.exit(0)

def tests(path, fold):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["PNG"] for row in csv.DictReader(handle) if (row.get(fold) or "").strip().lower() == "test"}

for fold in folds:
    a, b = tests(published, fold), tests(cut, fold)
    same = "the same pages" if a == b else f"{len(a - b)} only in the first, {len(b - a)} only in the second"
    print(f"{fold}: {len(a)} test pages in {published.name}, {len(b)} in {cut.name}: {same}")
PY

    # every test page is in a zip of its category (nothing is written)
    for fold in "${FOLDS[@]}"; do
        python3 data_scripts/stage_fold_pages.py --folds_csv "$FOLDS_CSV" --fold "$fold" --zips "$ZIPS" \
            --out "$STAGE/$fold" --dry-run
    done
    date
    exit 0
fi

# ── the job: on a GPU node, offline ────────────────────────────────────────────────────────────────
export HF_HUB_OFFLINE=1        # weights downloaded beforehand by the prefetch above

for pair in "${PAIRS[@]}"; do
    rev=${pair%%:*}
    [[ -f "model/model_${rev//./}/config.json" ]] || { echo "model/model_${rev//./} missing: run '$0 prefetch' first" >&2; exit 1; }
done
for path in "$FOLDS_CSV" "$ZIPS"; do
    [[ -e "$path" ]] || { echo "$path not found" >&2; exit 1; }
done

LOG=$STATS/${STAMP}_vX4_steps.txt
MANIFEST=$STATS/${STAMP}_vX4_tables.tsv
SUMMARY=$STATS/${STAMP}_vX4_summary.md
START=$(mktemp)
{
    echo "vX.4 evaluation, job ${SLURM_JOB_ID:-none} on $(hostname), started $(date)"
    echo "commit $(git rev-parse HEAD) ($(git describe --tags --always --dirty 2>/dev/null))"
    python3 check_version.py --para-config setup/para_config.txt 2>&1 | tail -1
    echo "folds CSV $FOLDS_CSV"
    echo "zips $ZIPS"
    echo
} > "$LOG"
printf 'set\trevision\tfold\ttop_n\ttable\n' > "$MANIFEST"

FAILED=0
step() {  # step <name> <command...>: run it, log OK/FAILED, go on either way
    local name=$1 status
    shift
    echo "=== $name: $*"
    "$@"
    status=$?
    if [[ $status -eq 0 ]]; then
        echo "OK      $name" >> "$LOG"
    else
        echo "FAILED  $name (exit $status)" >> "$LOG"
        FAILED=$((FAILED + 1))
    fi
    return $status
}

evaluate() {  # evaluate <set> <revision> <fold> <top-N> <run.py args...>: one --eval run, its table recorded
    local set=$1 rev=$2 fold=$3 topn=$4 marker table
    shift 4
    marker=$(mktemp)
    step "$rev $set top-$topn" python3 run.py --eval --no-train --no-hf -rev "$rev" -tn "$topn" "$@"
    table=$(find "$RESULTS/tables" -name "*_model_${rev//./}_TOP-${topn}_EVAL.csv" -newer "$marker" | sort | tail -1)
    [[ -n $table ]] && printf '%s\t%s\t%s\t%s\t%s\n' "$set" "$rev" "$fold" "$topn" "$table" >> "$MANIFEST"
    rm -f "$marker"
}

# 1. each model on its fold's test pages, as published in the LINDAT record
for fold in "${FOLDS[@]}"; do
    if step "stage $fold" python3 data_scripts/stage_fold_pages.py --folds_csv "$FOLDS_CSV" --fold "$fold" \
        --zips "$ZIPS" --out "$STAGE/$fold"; then
        for pair in "${PAIRS[@]}"; do
            [[ ${pair#*:} == "$fold" ]] || continue
            rev=${pair%%:*}
            evaluate "$fold test" "$rev" "$fold" 1 --folds_csv "$FOLDS_CSV" --eval_dir "$STAGE/$fold" --no-raw
            evaluate "$fold test" "$rev" "$fold" 3 --folds_csv "$FOLDS_CSV" --eval_dir "$STAGE/$fold" --raw
        done
    fi
    [[ -n "${KEEP_STAGED:-}" ]] || rm -rf "${STAGE:?}/$fold"
done

# 2. the 5,449-page set (most of it is training pages; reported by group); 3. the demo pages
for pair in "${PAIRS[@]}"; do
    rev=${pair%%:*}
    if [[ -d "$SET_5449" ]]; then
        evaluate "5,449-page set" "$rev" "${pair#*:}" 1 --folds_csv "" --eval_dir "$SET_5449" --no-raw
    fi
    evaluate "small_data_samples" "$rev" "${pair#*:}" 1 --folds_csv "" --eval_dir small_data_samples --no-raw
done
[[ -d "$SET_5449" ]] || echo "SKIPPED 5,449-page set: $SET_5449 not found" >> "$LOG"

# 4. the --best ensemble demo
step "BEST_5 small_data_samples" python3 run.py --best -d small_data_samples -ff png --inner --no-chunk --no-train \
    --no-eval --no-hf --save-intermediates -tn 3

# 5. the summary
find "$RESULTS" -type f -newer "$START" ! -name "$(basename "$SUMMARY")" | sort > "$STATS/${STAMP}_vX4_outputs.txt"
step "summary" python3 - "$SUMMARY" "$MANIFEST" "$LOG" "$FOLDS_CSV" "$STATS" <<'PY'
import csv, sys
from pathlib import Path

sys.path.insert(0, "data_scripts")
import eval_by_document_overlap as ev

summary, manifest, log, folds_csv, stats = (Path(arg) for arg in sys.argv[1:])
rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
hub = sorted(stats.glob("*_vX4_hub_revisions.csv"))
splits = {}


def accuracy(table, top_n):
    results = ev.read_results(Path(table))
    hits = sum(r["TRUE"] in [r.get(f"CLASS-{n}") or r.get("CATEGORY") for n in range(1, top_n + 1)] for r in results)
    return len(results), (100 * hits / len(results) if results else 0.0)


lines = [f"# vX.4 evaluation, {summary.name.split('_')[0]}", "", "```", log.read_text(encoding="utf-8").strip(), "```", ""]
if hub:
    lines += [f"## Hub revisions ({hub[-1].name})", "", "| " + " | ".join(next(csv.reader(hub[-1].open()))) + " |"]
    lines += ["|" + "---|" * 5] + ["| " + " | ".join(r) + " |" for r in list(csv.reader(hub[-1].open()))[1:]] + [""]
lines += ["## Accuracy", "", "| Set | Revision | Fold | Pages | Top-N | Accuracy | Table |", "|---|---|---|---:|---:|---:|---|"]
for r in rows:
    pages, acc = accuracy(r["table"], int(r["top_n"]))
    lines.append(f"| {r['set']} | {r['revision']} | {r['fold']} | {pages} | {r['top_n']} | {acc:.2f} % | `{Path(r['table']).name}` |")
lines += ["", f"## Top-1 by document overlap with the model's fold ({folds_csv.name})", ""]
for r in rows:
    if r["top_n"] != "1":
        continue
    split = splits.setdefault(r["fold"], ev.read_split(folds_csv, r["fold"]))
    lines += [f"### {r['revision']} ({r['fold']}), {r['set']}", "", ev.markdown(Path(r["table"]).name, ev.evaluate(split, ev.read_results(Path(r["table"]))))]
summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary.read_text(encoding="utf-8"))
PY

rm -f "$START"
echo "$FAILED step(s) failed; summary in $SUMMARY"
date
exit "$FAILED"
