#!/bin/bash
#SBATCH -J pc-decide-brief  # name of job
#SBATCH -p gpu-ms,gpu-troja        # name of partition or queue
#SBATCH --cpus-per-task=16                 # number of cores/threads per task
#SBATCH --gres=gpu:1                       # Clef in bf16 needs ~54 GB of GPU memory; Clef-Flash ~18 GB; Strands none
#SBATCH --mem=80G
#SBATCH --time 0-06:00:00
#SBATCH -o pc-decide-brief.%j.%N.out
#SBATCH -e pc-decide-brief.%j.%N.err
#
# A brief zero-shot pass of a decision model on the 11-category LINDAT pages (#58; ufal/atrium-project#83).
# Nothing is trained or fine-tuned: the CNN is a published checkpoint and the decision model reads the committed
# question sets of decisions/ as they are. TEMPLATE: written 2026-11-08, not yet run.
#
# One job, on the cluster only (the pages never leave it):
#   1. a sample of at most PER_LABEL pages per category from one partition of the folds CSV (fold1 dev by default:
#      held out from v1.4's training, and not the test pages, which wait until wording and tau are frozen);
#   2. those pages out of the LINDAT zips (data_scripts/stage_fold_pages.py);
#   3. the CNN baseline on them (run.py --eval --raw, CNN_REV, no training), for the cascade;
#   4. the decision-model server in this job, as in decide_server_job.sh, and its probe;
#   5. decide_pages.py, one table per task: b1 (the 11 labels), b2 (the cascade behind the CNN, below TAU, among its
#      top 3), triage (content, processors and condition in one request); with DEGRADE=1 also b4 on graded copies of
#      some of the sampled pages (data_scripts/degrade_pages.py);
#   6. BRIEF.md: the headline figures of every task, the step log and the provenance.
#
# Output: result/decide/<stamp>_<model>_brief_<job id>/.
# The sample is small: the result shows whether the pipeline works and what it costs, not how good the model is.
#
#   bash supplementary/scripts/decide_brief_job.sh prefetch            # login node, online: weights, image, CNN, checks
#   MODEL=clef        sbatch supplementary/scripts/decide_brief_job.sh  # SGLang, lmsysorg/sglang:dev-clef
#   MODEL=clef-flash  sbatch supplementary/scripts/decide_brief_job.sh
#   MODEL=strands     sbatch --gres=gpu:0 supplementary/scripts/decide_brief_job.sh   # the CNN then runs on the CPU
#   PER_LABEL=5 TASKS="b1" sbatch supplementary/scripts/decide_brief_job.sh          # an even briefer one
#
# Paths and choices by environment variable: ATRIUM, REPO, FOLDS_CSV, ZIPS, STAGE, MODEL, PORT, SIF, STRANDS_VENV,
# FOLD, PART, PER_LABEL, SEED, CNN_REV, TAU, MAX_SIDES, TASKS, DEGRADE, DEGRADE_PER_LABEL, PROBE_IMAGES, ALLOW_TEST.
# STAGE must hold this sample only: its default name carries the fold, the partition, PER_LABEL and SEED.

set -u
hostname
date

ATRIUM=${ATRIUM:-/lnet/work/projects/atrium}
REPO=${REPO:-$ATRIUM/GITHUB/atrium-page-classification}
PAGES=$ATRIUM/page_util/pages
FOLDS_CSV=${FOLDS_CSV:-$PAGES/licensed_archives/licensed_crossval_folds.csv}
ZIPS=${ZIPS:-$PAGES/licensed_archives}
MODEL=${MODEL:-clef}
PORT=${PORT:-30000}
SIF=${SIF:-$ATRIUM/containers/sglang_dev-clef.sif}
STRANDS_VENV=${STRANDS_VENV:-$ATRIUM/venvs/strands-decider}
FOLD=${FOLD:-fold1}
PART=${PART:-dev}
PER_LABEL=${PER_LABEL:-20}
SEED=${SEED:-420}
CNN_REV=${CNN_REV:-v1.4}
TAU=${TAU:-0.9}
MAX_SIDES=${MAX_SIDES:-1024}          # a space-separated list of long sides in px, e.g. "768 1024 1536"
TASKS=${TASKS:-b1 b2 triage}          # any of b1 b2 b3 b4 triage
DEGRADE=${DEGRADE:-0}
DEGRADE_PER_LABEL=${DEGRADE_PER_LABEL:-2}
STAGE=${STAGE:-$PAGES/stage_decide_brief/${FOLD}_${PART}_${PER_LABEL}_s${SEED}}

case "$MODEL" in
    clef)       HF_ID=Cloudflare/clef ;;
    clef-flash) HF_ID=Cloudflare/clef-flash ;;
    strands)    HF_ID=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 ;;
    *) echo "MODEL must be clef, clef-flash or strands, not '$MODEL'" >&2; exit 2 ;;
esac
for task in $TASKS; do
    case "$task" in
        b1|b2|b3|b4|triage) ;;
        *) echo "TASKS: '$task' is not one of b1 b2 b3 b4 triage" >&2; exit 2 ;;
    esac
done
wants() { [[ " $TASKS " == *" $1 "* ]]; }
if [[ "$PART" == test && "${ALLOW_TEST:-0}" != 1 ]]; then
    echo "PART=test: the test pages wait until wording and tau are frozen on dev (#58 plan, Phase C); ALLOW_TEST=1 overrides" >&2
    exit 2
fi

export HF_HOME=$ATRIUM/cache/
export TORCH_HOME=$ATRIUM/cache/
export PYTHONUNBUFFERED=1      # progress shows up in the .out file as it happens

cd "$REPO" || exit 1
# shellcheck source=/dev/null
source "$REPO/venv/bin/activate"

CNN_DIR=model/model_${CNN_REV//./}
CNN_FOLD=$(python3 -c 'import sys; from model_registry import REVISION_BEST_FOLDS as F; print(F.get(sys.argv[1], ""))' "$CNN_REV")

# sample <out.csv>: at most PER_LABEL rows per category of the FOLD == PART rows, the CSV's own columns kept, so
# stage_fold_pages.py reads it as it reads the folds CSV. Name-sorted, then drawn with SEED, label by label in
# model_registry.CATEGORIES order, as decide_pages.sample_per_label draws.
sample() {
    python3 - "$FOLDS_CSV" "$1" "$FOLD" "$PART" "$PER_LABEL" "$SEED" <<'PY'
import csv, random, sys
from model_registry import CATEGORIES

folds_csv, out, fold, part, per_label, seed = sys.argv[1:]
with open(folds_csv, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames or []
    absent = {"PNG", "categ", fold} - set(fields)
    if absent:
        sys.exit(f"{folds_csv}: no column {sorted(absent)}")
    rows = [r for r in reader if (r[fold] or "").strip().lower() == part and (r["PNG"] or "").strip()]
by_label = {}
for row in sorted(rows, key=lambda r: r["PNG"].strip()):
    by_label.setdefault(row["categ"].strip(), []).append(row)
unknown = sorted(set(by_label) - set(CATEGORIES))
if unknown:
    sys.exit(f"{folds_csv}: categories outside model_registry.CATEGORIES: {unknown}")
rng, n, picked = random.Random(int(seed)), int(per_label), []
for label in CATEGORIES:
    group = by_label.get(label, [])
    picked += group if len(group) <= n else sorted(rng.sample(group, n), key=lambda r: r["PNG"].strip())
if not picked:
    sys.exit(f"{folds_csv}: no {fold} {part} rows with a PNG name")
with open(out, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(picked)
counts = {label: min(len(by_label.get(label, [])), n) for label in CATEGORIES}
print(f"sample: {len(picked)} {fold} {part} pages, " + ", ".join(f"{k} {v}" for k, v in counts.items()))
PY
}

# ── prefetch: on a login node with network, before submitting ─────────────────────────────────────
if [[ "${1:-}" == "prefetch" ]]; then
    python3 -c "from huggingface_hub import snapshot_download as s; print(s('$HF_ID'))" || exit 1
    if [[ "$MODEL" != strands && ! -f "$SIF" ]]; then
        mkdir -p "$(dirname "$SIF")"
        apptainer pull "$SIF" docker://lmsysorg/sglang:dev-clef || exit 1
    fi
    if wants b2; then
        # run.py --hf saves the Hub revision to model/model_vN4 and caches its base model and processor
        python3 run.py --hf --no-train --no-eval -rev "$CNN_REV" || exit 1
    fi
    TMP_SAMPLE=$(mktemp --suffix=.csv)
    sample "$TMP_SAMPLE" || exit 1
    python3 data_scripts/stage_fold_pages.py --folds_csv "$TMP_SAMPLE" --fold "$FOLD" --part "$PART" --zips "$ZIPS" \
        --out "$STAGE" --dry-run
    status=$?
    rm -f "$TMP_SAMPLE"
    date
    exit "$status"
fi

# ── the job: offline ──────────────────────────────────────────────────────────────────────────────
export HF_HUB_OFFLINE=1        # weights downloaded beforehand by the prefetch above

for path in "$FOLDS_CSV" "$ZIPS"; do
    [[ -e "$path" ]] || { echo "$path not found" >&2; exit 1; }
done
if wants b2; then
    [[ -f "$CNN_DIR/config.json" ]] || { echo "$CNN_DIR missing: run '$0 prefetch' first" >&2; exit 1; }
    # the CNN must not have trained on the sampled pages: its fold is the sample's
    [[ "$CNN_FOLD" == "$FOLD" ]] || {
        echo "CNN_REV $CNN_REV was trained on ${CNN_FOLD:-an unknown fold}, not $FOLD: its $FOLD $PART pages may be training pages" >&2
        exit 1
    }
fi

STAMP=$(date +%Y%m%d-%H%M)
OUT=result/decide/${STAMP}_${MODEL}_brief_${SLURM_JOB_ID:-$$}   # the job id: two jobs may start in one minute
RESULTS=$(python3 -c 'import configparser; c = configparser.ConfigParser(); c.read("setup/config.txt"); print(c.get("OUTPUT", "FOLDER_RESULTS"))')
mkdir -p "$OUT" "$RESULTS/tables"
LOG=$OUT/steps.txt
: > "$LOG"

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

# ── 1–2. the sample and its pages ─────────────────────────────────────────────────────────────────
step "sample" sample "$OUT/sample.csv" || exit 1
# decide_pages.py --dir reads every page in STAGE, so a page of another sample there would be asked too
python3 - "$OUT/sample.csv" "$STAGE" <<'PY' || exit 1
import csv, sys
from pathlib import Path

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    names = {row["PNG"].strip() for row in csv.DictReader(handle)}
stage = Path(sys.argv[2])
extra = sorted(p.name for p in stage.glob("*/*.*") if p.suffix.lower() == ".png" and p.name not in names)
if extra:
    sys.exit(f"{stage} holds {len(extra)} pages outside this sample (e.g. {extra[0]}): give this sample a STAGE of its own")
PY
step "stage" python3 data_scripts/stage_fold_pages.py --folds_csv "$OUT/sample.csv" --fold "$FOLD" --part "$PART" \
    --zips "$ZIPS" --out "$STAGE" || exit 1
SAMPLED=$(( $(wc -l < "$OUT/sample.csv") - 1 ))

# ── provenance: what was asked, of which pages, by what ───────────────────────────────────────────
{
    echo "# Brief decision-model pass, job ${SLURM_JOB_ID:-local} on $(hostname), $(date -u +%FT%TZ)"
    echo "repo commit: $(git rev-parse HEAD) ($(git describe --tags --always --dirty 2>/dev/null))"
    echo "pages: $SAMPLED, at most $PER_LABEL per category of $FOLD $PART, seed $SEED, from $FOLDS_CSV"
    echo "sample sha256: $(sha256sum "$OUT/sample.csv" | cut -d' ' -f1)"
    echo "tasks: $TASKS$([[ "$DEGRADE" == 1 ]] && echo " + b4 on degraded copies ($DEGRADE_PER_LABEL per category)")"
    echo "max sides: $MAX_SIDES"
    wants b2 && echo "cnn: $CNN_REV ($CNN_FOLD), tau $TAU"
    echo "model: $HF_ID"
    echo "revision: $(python3 -c "from huggingface_hub import snapshot_download as s; import os; print(os.path.basename(s('$HF_ID', local_files_only=True)))" 2>/dev/null)"
    if [[ "$MODEL" == strands ]]; then
        echo "server: strands-decider $("$STRANDS_VENV/bin/pip" show strands-decider 2>/dev/null | sed -n 's/^Version: //p')"
    else
        echo "server: SGLang, $SIF ($(sha256sum "$SIF" | cut -c1-16))"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    fi
} > "$OUT/PROVENANCE.md"
cat "$OUT/PROVENANCE.md"

# ── 3. the CNN baseline, before the server takes the GPU ──────────────────────────────────────────
CNN_RAW=""
if wants b2; then
    MARKER=$(mktemp)
    if step "cnn $CNN_REV" python3 run.py --eval --no-train --no-hf -rev "$CNN_REV" -tn 3 --folds_csv "" \
        --eval_dir "$STAGE" --raw; then
        table=$(find "$RESULTS/tables" -name "*_${CNN_DIR#model/}_EVAL_RAW.csv" -newer "$MARKER" | sort | tail -1)
        if [[ -n $table ]]; then
            cp "$table" "$OUT/cnn_EVAL_RAW.csv" && CNN_RAW=$OUT/cnn_EVAL_RAW.csv
        else
            echo "FAILED  cnn table (no new *_EVAL_RAW.csv in $RESULTS/tables)" >> "$LOG"
            FAILED=$((FAILED + 1))
        fi
    fi
    rm -f "$MARKER"
fi

# ── 4. the server, on 127.0.0.1 only (strands-decider serve has no authentication) ────────────────
if [[ "$MODEL" == strands ]]; then
    "$STRANDS_VENV/bin/strands-decider" serve --model "$HF_ID" --port "$PORT" > "$OUT/server.log" 2>&1 &
else
    apptainer exec --nv "$SIF" sglang serve --model-path "$HF_ID" --host 127.0.0.1 --port "$PORT" \
        > "$OUT/server.log" 2>&1 &
fi
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

T0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/systemone" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$HF_ID\",\"state\":\"ping\",\"questions\":{\"q\":{\"type\":\"noul\"}}}" > /dev/null; do
    kill -0 $SERVER_PID 2>/dev/null || { echo "the server exited; see $OUT/server.log" >&2; exit 1; }
    sleep 10
done
echo "cold load: $(( $(date +%s) - T0 )) s" | tee -a "$OUT/PROVENANCE.md"

export DECISION_MODEL_URL=http://127.0.0.1:$PORT
export DECISION_MODEL_NAME=$HF_ID
REVISION=$(sed -n 's/^revision: //p' "$OUT/PROVENANCE.md")

# Two public pages for the image check: a photograph and a printed text page.
PROBE_IMAGES=${PROBE_IMAGES:-"small_data_samples/PHOTO/caa_conference-01.png $(find small_data_samples/TEXT_P -name '*.png' | sort | head -1)"}
probe_args=()
for image in $PROBE_IMAGES; do  # a space-separated list of paths without spaces
    probe_args+=(--image "$image")
done
step "probe" python3 decision_client.py --probe --no-proxy "${probe_args[@]}" --out "$OUT/probe.json"

# ── 5. the zero-shot runs ─────────────────────────────────────────────────────────────────────────
side_args=()
for side in $MAX_SIDES; do
    side_args+=(--max-side "$side")
done
decide() {  # decide <name> <decide_pages.py args...>: one task, one table in $OUT
    local name=$1
    shift
    step "$name" python3 decide_pages.py --no-proxy --served-on cluster-job --model-revision "$REVISION" \
        "${side_args[@]}" --out "$OUT/${name}_DECIDE_RAW.csv" "$@"
}

for task in $TASKS; do
    if [[ "$task" == b2 ]]; then
        if [[ -z "$CNN_RAW" ]]; then
            echo "SKIPPED b2: no CNN table" >> "$LOG"
            continue
        fi
        decide b2 --task b2 --dir "$STAGE" --cnn-raw "$CNN_RAW" --tau "$TAU"
    else
        decide "$task" --task "$task" --dir "$STAGE"
    fi
done

if [[ "$DEGRADE" == 1 ]]; then
    # the copies are deterministic for this STAGE and SEED, so an earlier run's are reused (degrade_pages.py refuses
    # to write over them)
    DEGRADED=${STAGE}_degraded_${DEGRADE_PER_LABEL}
    if [[ -f "$DEGRADED/degradations.csv" ]]; then
        echo "OK      degrade (reused $DEGRADED)" >> "$LOG"
        decide b4_degraded --task b4 --degradations "$DEGRADED/degradations.csv"
    elif step "degrade" python3 data_scripts/degrade_pages.py --dir "$STAGE" --per-label "$DEGRADE_PER_LABEL" \
        --seed "$SEED" --out "$DEGRADED"; then
        decide b4_degraded --task b4 --degradations "$DEGRADED/degradations.csv"
    fi
fi

# ── 6. the brief ──────────────────────────────────────────────────────────────────────────────────
step "brief" python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])


def pct(value):
    return "–" if value is None else f"{100 * value:.1f} %"


def num(value, digits=0):
    return "–" if value is None else f"{value:.{digits}f}"


def headline(task, entry):
    if task == "b1":
        return f"top-1 {pct(entry.get('top1'))}"
    if task == "b2":
        return (
            f"cascade top-1 {pct(entry.get('top1'))}, CNN alone {pct(entry.get('cnn_top1'))}; "
            f"{entry.get('deferred')} deferred: {entry.get('fixed')} fixed, {entry.get('broken')} broken "
            f"(McNemar p {entry.get('mcnemar_p')})"
        )
    parts = []
    content = entry.get("content") or {}
    if content:
        parts.append(
            f"main processor right {pct(content.get('primary_accuracy'))}, "
            f"processor set exact {pct(content.get('exact_route_set'))}"
        )
    condition = entry.get("condition") or {}
    if condition:
        damage = (condition.get("scores") or {}).get("overall_damage") or {}
        parts.append(f"overall damage mean {num(damage.get('mean'), 2)} of 4")
        for defect, figures in sorted((condition.get("injected") or {}).items()):
            parts.append(
                f"{defect}: detected {pct(figures.get('detected'))} "
                f"(originals {pct(figures.get('false_alarm_on_originals'))}), mean Δp {num(figures.get('mean_delta'), 3)}"
            )
        if condition.get("orientation_pages"):
            parts.append(f"orientation right {pct(condition.get('orientation_accuracy_text_pages'))} of text pages")
    return "; ".join(parts) or "–"


lines = [f"# {out.name}", "", "```", (out / "PROVENANCE.md").read_text(encoding="utf-8").strip(), "```", ""]
lines += ["| Run | Max side | Rows | Statuses, all sides | p50 ms | p95 ms | Input tokens | Headline |", "|---|---:|---:|---|---:|---:|---:|---|"]
for path in sorted(out.glob("*_DECIDE_RAW.summary.json")):
    summary = json.loads(path.read_text(encoding="utf-8"))
    run = path.name.split("_DECIDE_RAW")[0]
    statuses = ", ".join(f"{k} {v}" for k, v in sorted(summary.get("statuses", {}).items()))
    for side, entry in summary.get("by_max_side", {}).items():
        lines.append(
            f"| {run} | {side} | {entry.get('rows')} | {statuses} | {num(entry.get('latency_ms_p50'))} | "
            f"{num(entry.get('latency_ms_p95'))} | {num(entry.get('input_tokens_mean'))} | {headline(summary['task'], entry)} |"
        )
lines += ["", "Every figure in full: the `*_DECIDE_RAW.summary.json` files beside this one.", ""]
lines += ["## Steps", "", "```", (out / "steps.txt").read_text(encoding="utf-8").strip(), "```", ""]
(out / "BRIEF.md").write_text("\n".join(lines), encoding="utf-8")
print((out / "BRIEF.md").read_text(encoding="utf-8"))
PY

echo "$FAILED step(s) failed; the brief is in $OUT/BRIEF.md"
date
exit "$FAILED"
