#!/bin/bash
# Script to analyze model test results and calculate accuracy, precision, and recall.
#
# Scans a directory for EVAL CSV result files and reports per-model accuracy,
# precision, recall, and confusion statistics for TOP-1 and TOP-3 predictions.
#
# Usage:
#   ./result_analysis.sh [TARGET_DIR] [TOP_N] [--pattern GLOB] [-h|--help]
#
# Arguments:
#   TARGET_DIR      Directory to search for EVAL CSV files (default: .)
#   TOP_N           Max top-N rank to report in the summary (default: 5)
#
# Options:
#   --pattern GLOB  Filename glob to restrict which EVAL files are loaded.
#                   Must match against the basename.
#                   Default: *_model_*_TOP-*_EVAL.csv
#   -h, --help      Show this help message and exit
#
# Examples:
#   ./result_analysis.sh result/tables 3
#   ./result_analysis.sh result/tables 5 --pattern "*_v43_*_EVAL.csv"

set -euo pipefail

echo "Model Test Results Analysis"
echo "==========================="
echo

# ── Argument parsing ──────────────────────────────────────────────────────
TARGET_DIR="."
TOP_N=5
# P1 FIX: the original pattern was "2025102*5449_model_*_TOP-*_EVAL.csv" which
# embedded a specific run date and sample count, making the script match nothing
# in any fresh installation.  The default is now the general form; users can
# still narrow it via --pattern.
FILE_PATTERN="*_model_*_TOP-*_EVAL.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        --pattern)
            FILE_PATTERN="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1"; echo "Run with --help for usage."; exit 1 ;;
        *)
            if [[ -z "${_TARGET_SET:-}" ]]; then
                TARGET_DIR="$1"; _TARGET_SET=1
            elif [[ -z "${_TOPN_SET:-}" ]]; then
                TOP_N="$1"; _TOPN_SET=1
            else
                echo "Unexpected positional argument: $1"; exit 1
            fi
            shift ;;
    esac
done

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

# ── Model version prefix to base model mapping ────────────────────────────
declare -A version_to_base_model=(
    ["v1"]="timm/tf_efficientnetv2_s.in21k"
    ["v2"]="google/vit-base-patch16-224"
    ["v3"]="google/vit-base-patch16-384"
    ["v4"]="timm/tf_efficientnetv2_l.in21k_ft_in1k"
    ["v5"]="google/vit-large-patch16-384"
    ["v6"]="timm/regnety_120.sw_in12k_ft_in1k"
    ["v7"]="timm/regnety_160.swag_ft_in1k"
    ["v8"]="timm/regnety_640.seer"
    ["v9"]="microsoft/dit-base-finetuned-rvlcdip"
    ["v10"]="microsoft/dit-large-finetuned-rvlcdip"
    ["v11"]="microsoft/dit-large"
    ["v12"]="timm/tf_efficientnetv2_m.in21k_ft_in1k"

    # Special remappings (different base than their numeric prefix implies)
    ["v13"]="timm/tf_efficientnetv2_m.in21k_ft_in1k"
    ["v43"]="timm/regnety_160.swag_ft_in1k"
    ["v63"]="timm/regnety_640.seer"

    # New v*.3 variants
    ["v23"]="google/vit-base-patch16-224"
    ["v33"]="google/vit-base-patch16-384"
    ["v53"]="google/vit-large-patch16-384"
    ["v73"]="timm/regnety_160.swag_ft_in1k"
    ["v83"]="timm/regnety_640.seer"
    ["v93"]="microsoft/dit-base-finetuned-rvlcdip"
    ["v103"]="microsoft/dit-large-finetuned-rvlcdip"
    ["v113"]="microsoft/dit-large"
    ["v123"]="timm/tf_efficientnetv2_m.in21k_ft_in1k"
)

declare -A suffix_to_samples=(
    ["15k"]="1,500"   ["2k"]="2,000"   ["25k"]="2,500"
    ["3k"]="3,000"    ["35k"]="3,500"  ["4k"]="4,000"
    ["45k"]="4,500"   ["14k"]="14,000"
)

parse_model_info() {
    local version="$1"
    local base_model=""
    local sample_limit="14,000"
    local fold=""
    local version_core=""

    if [[ $version =~ a([0-9])?$ ]]; then
        fold="avg"
        version_core=$(echo "$version" | sed 's/a[0-9]*$//')
    elif [[ -n "${version_to_base_model[$version]:-}" ]]; then
        fold="?"
        version_core=$version
    elif [[ $version =~ ([0-9])$ ]]; then
        local potential_core="${version%?}"
        local potential_fold="${BASH_REMATCH[1]}"
        if [[ -n "${version_to_base_model[$potential_core]:-}" ]]; then
            version_core=$potential_core
            fold=$potential_fold
        else
            fold="?"
            version_core=$version
        fi
    else
        fold="?"
        version_core="$version"
    fi

    if [[ -n "${version_to_base_model[$version_core]:-}" ]]; then
        base_model="${version_to_base_model[$version_core]}"
    else
        base_model="Unknown"
    fi

    for suffix in "${!suffix_to_samples[@]}"; do
        if [[ $version_core =~ $suffix ]]; then
            sample_limit="${suffix_to_samples[$suffix]}"
            break
        fi
    done

    echo "$base_model|$version_core|$sample_limit|$fold"
}

# ── Locate EVAL files ─────────────────────────────────────────────────────
FILES=$(find "$TARGET_DIR" -name "$FILE_PATTERN" | sort)

if [[ -z "$FILES" ]]; then
    echo "No files matching '$FILE_PATTERN' found in: $TARGET_DIR"
    exit 1
fi

echo "Found $(echo "$FILES" | wc -l) result file(s)  [pattern: $FILE_PATTERN]"
echo

# ── Storage ───────────────────────────────────────────────────────────────
declare -A results
declare -A base_models_seen
declare -A version_cores_seen
declare -A sample_limits_seen

for file in $FILES; do
    basename_file=$(basename "$file")

    if [[ $basename_file =~ _model_([^_]+)_TOP-([0-9]+)_EVAL\.csv ]]; then
        version="${BASH_REMATCH[1]}"
        top_n="${BASH_REMATCH[2]}"
    else
        echo "Warning: Cannot parse filename: $basename_file"
        continue
    fi

    model_info=$(parse_model_info "$version")
    IFS='|' read -r base_model version_core sample_limit fold <<< "$model_info"

    if [[ -z "$base_model" || "$base_model" == "Unknown" ]]; then
        echo "Warning: Unknown base model for version '$version' (core: $version_core)"
        continue
    fi

    base_models_seen["$base_model"]=1
    version_cores_seen["$version_core"]=1
    sample_limits_seen["$sample_limit"]=1

    [[ ! -r "$file" ]] && echo "Warning: Cannot read $file" && continue

    total_rows=$(tail -n +2 "$file" | wc -l)
    [[ "$total_rows" -eq 0 ]] && continue

    metrics=$(tail -n +2 "$file" | awk -F',' -v topn="$top_n" '
        BEGIN { correct = 0 }
        {
            true_val = $NF
            gsub(/^[ \t]*"?|"?[ \t]*$/, "", true_val)
            pred_top1 = $3
            gsub(/^[ \t]*"?|"?[ \t]*$/, "", pred_top1)

            found = 0
            for (i = 3; i < 3 + topn && i <= NF; i++) {
                class_val = $i
                gsub(/^[ \t]*"?|"?[ \t]*$/, "", class_val)
                if (class_val == true_val) { found = 1; break }
            }
            if (found) correct++

            true_count[true_val]++
            pred_count[pred_top1]++
            if (pred_top1 == true_val) tp[true_val]++
        }
        END {
            total_precision = 0; total_recall = 0; total_support = 0
            for (class in true_count) {
                support = true_count[class]; total_support += support
                precision = (pred_count[class] > 0) ? (tp[class]+0) / pred_count[class] : 0
                recall    = (tp[class]+0) / support
                total_precision += precision * support
                total_recall    += recall    * support
            }
            wp = (total_support > 0) ? total_precision / total_support : 0
            wr = (total_support > 0) ? total_recall    / total_support : 0
            printf "%d|%.4f|%.4f", correct, wp, wr
        }
    ')

    IFS='|' read -r correct_matches precision recall <<< "$metrics"
    accuracy=$(awk "BEGIN { printf \"%.2f\", $correct_matches * 100 / $total_rows }")
    precision_pct=$(awk "BEGIN { printf \"%.2f\", $precision * 100 }")
    recall_pct=$(awk "BEGIN { printf \"%.2f\", $recall * 100 }")

    key="${base_model}|${version_core}|${sample_limit}|${fold}|${top_n}"
    results["$key"]="$accuracy|$total_rows|$correct_matches|$precision_pct|$recall_pct"
done

# ── Per-base-model breakdown ──────────────────────────────────────────────
echo
echo "Results by Base Model"
echo "====================="

declare -A best_models
declare -A best_precision_models

for base_model in $(printf '%s\n' "${!base_models_seen[@]}" | sort); do
    echo
    echo ">>> Base Model: $base_model"
    echo "-----------------------------------------------------------------------"

    for version_core in $(printf '%s\n' "${!version_cores_seen[@]}" | sort -V); do
        for sample_limit in $(printf '%s\n' "${!sample_limits_seen[@]}" | sort -n); do
            has_data=0
            for key in "${!results[@]}"; do
                [[ $key == "${base_model}|${version_core}|${sample_limit}|"* ]] && has_data=1 && break
            done
            [[ $has_data -eq 0 ]] && continue

            echo
            echo "Version: $version_core | Sample Limit: $sample_limit"
            printf "  %-10s | %-8s | %-10s | %-10s | %-8s | %-12s | %-12s\n" \
                "Fold" "TOP-N" "Acc(%)" "Correct" "Total" "Precision(%)" "Recall(%)"
            echo "  --------------------------------------------------------------------------------------"

            best_acc=-1; best_fold=""; best_prec=-1; best_prec_fold=""

            for fold in 1 2 3 4 5 "avg" "?"; do
                for top_n in 1 3 5; do
                    key="${base_model}|${version_core}|${sample_limit}|${fold}|${top_n}"
                    [[ -z "${results[$key]:-}" ]] && continue
                    IFS='|' read -r acc total correct prec rec <<< "${results[$key]}"
                    printf "  %-10s | %-8s | %-10s | %-10s | %-8s | %-12s | %-12s\n" \
                        "$fold" "$top_n" "$acc" "$correct" "$total" "$prec" "$rec"

                    if [[ "$top_n" == "1" && "$fold" != "avg" && "$fold" != "?" ]]; then
                        acc_num=$(echo "$acc" | tr ',' '.')
                        (( $(awk "BEGIN{print($acc_num > $best_acc)}") )) && best_acc=$acc_num && best_fold=$fold
                        prec_num=$(echo "$prec" | tr ',' '.')
                        (( $(awk "BEGIN{print($prec_num > $best_prec)}") )) && best_prec=$prec_num && best_prec_fold=$fold
                    fi
                done
            done

            [[ -n "$best_fold" ]]      && best_models["${base_model}|${version_core}|${sample_limit}"]="${best_acc}|${best_fold}"
            [[ -n "$best_prec_fold" ]] && best_precision_models["${base_model}|${version_core}|${sample_limit}"]="${best_prec}|${best_prec_fold}"
        done
    done
done

# ── Summary statistics ────────────────────────────────────────────────────
echo
echo "Summary Statistics"
echo "=================="
printf "%-45s | %-6s | %-10s | %-10s | %-10s | %-10s | %-11s\n" \
    "Base Model" "Ver" "Samples" "Mean_Acc1" "Mean_Acc3" "Best_Acc1" "Worst_Acc1"
echo "-----------------------------------------------------------------------------------------------------------------------------"

tmpfile=$(mktemp)
trap 'rm -f "$tmpfile"' EXIT

for base_model in $(printf '%s\n' "${!base_models_seen[@]}" | sort); do
    for version_core in $(printf '%s\n' "${!version_cores_seen[@]}" | sort -V); do
        for sample_limit in $(printf '%s\n' "${!sample_limits_seen[@]}" | sort -n); do
            total_acc1=0; total_acc3=0; count1=0; count3=0; best_acc1=-1; worst_acc1=101

            for fold in 1 2 3 4 5; do
                key1="${base_model}|${version_core}|${sample_limit}|${fold}|1"
                if [[ -n "${results[$key1]:-}" ]]; then
                    IFS='|' read -r acc _ _ _ _ <<< "${results[$key1]}"
                    total_acc1=$(awk "BEGIN{print $total_acc1 + $acc}")
                    count1=$((count1 + 1))
                    (( $(awk "BEGIN{print($acc > $best_acc1)}") )) && best_acc1=$acc
                    (( $(awk "BEGIN{print($acc < $worst_acc1)}") )) && worst_acc1=$acc
                fi
                key3="${base_model}|${version_core}|${sample_limit}|${fold}|3"
                if [[ -n "${results[$key3]:-}" ]]; then
                    IFS='|' read -r acc _ _ _ _ <<< "${results[$key3]}"
                    total_acc3=$(awk "BEGIN{print $total_acc3 + $acc}")
                    count3=$((count3 + 1))
                fi
            done

            [[ $count1 -eq 0 ]] && continue
            mean_acc1=$(awk "BEGIN{printf \"%.2f\", $total_acc1 / $count1}")
            mean_acc3="N/A"; [[ $count3 -gt 0 ]] && mean_acc3=$(awk "BEGIN{printf \"%.2f\", $total_acc3 / $count3}")
            [[ "$best_acc1"  == "-1"  ]] && best_acc1="N/A"
            [[ "$worst_acc1" == "101" ]] && worst_acc1="N/A"

            printf "%-45s | %-6s | %-10s | %-10s | %-10s | %-10s | %-11s\n" \
                "$base_model" "$version_core" "$sample_limit" \
                "${mean_acc1}%" "${mean_acc3}%" "${best_acc1}%" "${worst_acc1}%"

            echo "${mean_acc1}|${base_model}|${version_core}|${sample_limit}" >> "$tmpfile"
        done
    done
done

# ── Top-N models by mean accuracy ────────────────────────────────────────
if [[ -s "$tmpfile" ]]; then
    awk -F'|' 'BEGIN{OFS=FS}{gsub(",",".",$1);print}' "$tmpfile" > "${tmpfile}.norm" && mv "${tmpfile}.norm" "$tmpfile"
    echo
    echo "Top-$TOP_N models by Mean TOP-1 accuracy"
    echo "-----------------------------------------"
    printf "%3s %-45s %-6s %10s\n" "#" "Base Model" "Ver" "Mean_Acc1"
    rank=1
    sort -t'|' -k1,1nr "$tmpfile" | head -n "$TOP_N" | while IFS='|' read -r mean base ver samples; do
        mean_n="${mean//,/.}"
        mean_d=$(awk -v m="$mean_n" 'BEGIN{printf "%.2f", m}')
        printf "%3d %-45s %-6s %8s%%\n" "$rank" "$base" "$ver" "$mean_d"
        rank=$((rank + 1))
    done
fi

# ── Best fold per model ───────────────────────────────────────────────────
echo
echo "Best Performing Fold by Base Model (TOP-1 Accuracy)"
echo "===================================================="
printf "%-45s | %-6s | %-10s | %-10s | %-6s\n" "Base Model" "Ver" "Samples" "Best_Acc1" "Fold"
echo "--------------------------------------------------------------------------------------------"

for best_key in $(printf '%s\n' "${!best_models[@]}" | sort); do
    IFS='|' read -r base_model version_core sample_limit <<< "$best_key"
    IFS='|' read -r best_acc best_fold <<< "${best_models[$best_key]}"
    printf "%-45s | %-6s | %-10s | %-9s%% | %-6s\n" \
        "$base_model" "$version_core" "$sample_limit" "$best_acc" "$best_fold"
done

echo
echo "Best Performing Fold by Base Model (TOP-1 Precision)"
echo "====================================================="
printf "%-45s | %-6s | %-10s | %-13s | %-6s\n" "Base Model" "Ver" "Samples" "Best_Prec" "Fold"
echo "--------------------------------------------------------------------------------------------"

for best_prec_key in $(printf '%s\n' "${!best_precision_models[@]}" | sort); do
    IFS='|' read -r base_model version_core sample_limit <<< "$best_prec_key"
    IFS='|' read -r best_prec best_prec_fold <<< "${best_precision_models[$best_prec_key]}"
    printf "%-45s | %-6s | %-10s | %-12s%% | %-6s\n" \
        "$base_model" "$version_core" "$sample_limit" "$best_prec" "$best_prec_fold"
done

echo
echo "Done!"
