#!/bin/bash
# Script to analyze model test results and calculate accuracy, precision, and recall
# New format: YYYYMMDD-HHMM_model_<version>_TOP-<N>_EVAL.csv
# Groups by base model, showing all folds and TOP-N values
# Expected columns: FILE,PAGE,CLASS-1,CLASS-2,CLASS-3,SCORE-1,SCORE-2,SCORE-3,TRUE
# Usage: ./analyze_models_topn.sh [TARGET_DIR] [TOP_N]
# TARGET_DIR default: .
# TOP_N default: 5

set -euo pipefail

echo "Model Test Results Analysis (New Format)"
echo "========================================"
echo

# Model version prefix to base model mapping
# THIS IS THE FIRST FIX:
# This map is now explicit for all known version cores.
# It correctly maps v13, v43, and v63 to their intended base models
# (from your original script) instead of grouping them with v1, v4, and v6.
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

    # Special cases (from original script, mapping to different bases)
    ["v13"]="timm/tf_efficientnetv2_m.in21k_ft_in1k"  # Maps to v12's model
    ["v43"]="timm/regnety_160.swag_ft_in1k"          # Maps to v7's model
    ["v63"]="timm/regnety_640.seer"                 # Maps to v8's model

    # New variants (grouped with their logical base)
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

# Sample limit mapping based on version suffix
declare -A suffix_to_samples=(
    ["15k"]="1,500"
    ["2k"]="2,000"
    ["25k"]="2,500"
    ["3k"]="3,000"
    ["35k"]="3,500"
    ["4k"]="4,000"
    ["45k"]="4,500"
    ["14k"]="14,000"
)

# Function to parse model info from version string
# THIS IS THE SECOND FIX:
# This logic now correctly handles all filename patterns.
# It checks for "avg" folds, then *exact* version core matches (like "v13"),
# *then* new-style folds (like "v131").
# This prevents "v13" from being split into "v1" and fold "3".
parse_model_info() {
    local version="$1"
    local base_model=""
    local sample_limit="14,000"  # default
    local fold=""
    local version_core=""

    # 1. Check for "avg" fold (e.g., v13a, v23a5)
    if [[ $version =~ a([0-9])?$ ]]; then
        fold="avg"
        version_core=$(echo "$version" | sed 's/a[0-9]*$//')

    # 2. Check for an *exact match* in the map (e.g., v1, v13, v43)
    # This is crucial to prevent splitting "v13" into "v1" + "3"
    elif [[ -n "${version_to_base_model[$version]:-}" ]]; then
        fold="?" # Fold not specified in filename
        version_core=$version

    # 3. Check for new-style fold (e.g., v131, v232, v1235)
    elif [[ $version =~ ([0-9])$ ]]; then
        local potential_core="${version%?}"
        local potential_fold="${BASH_REMATCH[1]}"

        # Check if the core *without* the last digit is a known prefix
        if [[ -n "${version_to_base_model[$potential_core]:-}" ]]; then
            version_core=$potential_core
            fold=$potential_fold
        else
            # Not a known pattern, treat as a core with unknown fold
            fold="?"
            version_core=$version
        fi
    else
        # No 'a' fold, not an exact match, not ending in a digit
        fold="?"
        version_core="$version"
    fi

    # Match the *found* version_core to its base_model
    if [[ -n "${version_to_base_model[$version_core]:-}" ]]; then
        base_model="${version_to_base_model[$version_core]}"
    else
        base_model="Unknown"
    fi

    # Extract sample limit from suffix (from the version_core)
    for suffix in "${!suffix_to_samples[@]}"; do
        if [[ $version_core =~ $suffix ]]; then
            sample_limit="${suffix_to_samples[$suffix]}"
            break
        fi
    done

    echo "$base_model|$version_core|$sample_limit|$fold"
}

# Check if directory is provided as argument
TARGET_DIR="${1:-.}"
TOP_N="${2:-5}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

# Find files matching new format
FILES=$(find "$TARGET_DIR" -name "2025102*5449_model_*_TOP-*_EVAL.csv" | sort)

if [ -z "$FILES" ]; then
    echo "No matching CSV files found in directory: $TARGET_DIR"
    exit 1
fi

echo "Found $(echo "$FILES" | wc -l) test files"
echo

# Storage for results
declare -A results
declare -A base_models_seen
declare -A version_cores_seen
declare -A sample_limits_seen

# Process each file
for file in $FILES; do
    basename_file=$(basename "$file")

    # Parse new filename format: YYYYMMDD-HHMM_model_<version>_TOP-<N>_EVAL.csv
    if [[ $basename_file =~ _model_([^_]+)_TOP-([0-9]+)_EVAL\.csv ]]; then
        version="${BASH_REMATCH[1]}"
        top_n="${BASH_REMATCH[2]}"
    else
        echo "Warning: Cannot parse filename: $basename_file"
        continue
    fi

    # Parse model info
    model_info=$(parse_model_info "$version")
    IFS='|' read -r base_model version_core sample_limit fold <<< "$model_info"

    if [[ -z "$base_model" || "$base_model" == "Unknown" ]]; then
        echo "Warning: Unknown base model for version: $version (Core: $version_core)"
        continue
    fi

    base_models_seen["$base_model"]=1
    version_cores_seen["$version_core"]=1
    sample_limits_seen["$sample_limit"]=1

    # Read and calculate TOP-N accuracy, precision, and recall
    if [ ! -r "$file" ]; then
        echo "Warning: Cannot read file: $file"
        continue
    fi

    total_rows=$(tail -n +2 "$file" | wc -l)
    if [ "$total_rows" -eq 0 ]; then
        continue
    fi

    # Calculate TOP-N accuracy and per-class metrics
    metrics=$(tail -n +2 "$file" | awk -F',' -v topn="$top_n" '
        BEGIN {
            correct = 0
        }
        {
            true_val = $NF
            gsub(/^[ \t]*"?|"?[ \t]*$/, "", true_val)

            # Get the top-1 prediction for precision/recall
            pred_top1 = $3
            gsub(/^[ \t]*"?|"?[ \t]*$/, "", pred_top1)

            # Check if correct (for TOP-N accuracy)
            found = 0
            for (i = 3; i < 3 + topn && i <= NF; i++) {
                class_val = $i
                gsub(/^[ \t]*"?|"?[ \t]*$/, "", class_val)
                if (class_val == true_val) {
                    found = 1
                    break
                }
            }
            if (found) correct++

            # Track per-class counts for precision/recall
            # Always use top-1 prediction for precision/recall regardless of TOP-N
            true_count[true_val]++
            pred_count[pred_top1]++

            # True positive: top-1 prediction matches true class
            if (pred_top1 == true_val) {
                tp[true_val]++
            }
        }
        END {
            # Calculate weighted precision and recall
            total_precision = 0
            total_recall = 0
            total_support = 0

            for (class in true_count) {
                support = true_count[class]
                total_support += support

                # Precision: TP / (TP + FP) = TP / predicted_count
                if (pred_count[class] > 0) {
                    precision = (tp[class] ? tp[class] : 0) / pred_count[class]
                } else {
                    precision = 0
                }

                # Recall: TP / (TP + FN) = TP / true_count
                recall = (tp[class] ? tp[class] : 0) / support

                total_precision += precision * support
                total_recall += recall * support
            }

            weighted_precision = (total_support > 0) ? total_precision / total_support : 0
            weighted_recall = (total_support > 0) ? total_recall / total_support : 0

            printf "%d|%.4f|%.4f", correct, weighted_precision, weighted_recall
        }
    ')

    IFS='|' read -r correct_matches precision recall <<< "$metrics"
    accuracy=$(awk "BEGIN { printf \"%.2f\", $correct_matches * 100 / $total_rows }")
    precision_pct=$(awk "BEGIN { printf \"%.2f\", $precision * 100 }")
    recall_pct=$(awk "BEGIN { printf \"%.2f\", $recall * 100 }")

    # Store result: key = base_model|version_core|sample_limit|fold|top_n
    key="${base_model}|${version_core}|${sample_limit}|${fold}|${top_n}"
    results["$key"]="$accuracy|$total_rows|$correct_matches|$precision_pct|$recall_pct"
done

echo
echo "Results by Base Model"
echo "===================="
echo

# Storage for best models
declare -A best_models
declare -A best_precision_models

# Sort and display by base model
for base_model in $(printf '%s\n' "${!base_models_seen[@]}" | sort); do
    echo
    echo ">>> Base Model: $base_model"
    echo "-------------------------------------------------------------------------"

    # Find all version cores for this base model
    for version_core in $(printf '%s\n' "${!version_cores_seen[@]}" | sort -V); do
        for sample_limit in $(printf '%s\n' "${!sample_limits_seen[@]}" | sort -n); do

            # Check if there's data for this combination
            has_data=0
            for key in "${!results[@]}"; do
                if [[ $key == "${base_model}|${version_core}|${sample_limit}|"* ]]; then
                    has_data=1
                    break
                fi
            done

            if [[ $has_data -eq 1 ]]; then
                echo
                echo "Version: $version_core | Sample Limit: $sample_limit"
                printf "  %-10s | %-8s | %-10s | %-10s | %-8s | %-12s | %-12s\n" \
                    "Fold" "TOP-N" "Acc(%)" "Correct" "Total" "Precision(%)" "Recall(%)"
                echo "  ---------------------------------------------------------------------------------------"

                # Track best fold for TOP-1
                best_acc=-1
                best_fold=""
                best_prec=-1
                best_prec_fold=""

                # Display results for all folds and TOP-N values
                for fold in 1 2 3 4 5 "avg" "?"; do # Added "?" for unknown folds
                    for top_n in 1 3 5; do
                        key="${base_model}|${version_core}|${sample_limit}|${fold}|${top_n}"
                        if [[ -n "${results[$key]:-}" ]]; then
                            IFS='|' read -r acc total correct prec rec <<< "${results[$key]:-}"
                            printf "  %-10s | %-8s | %-10s | %-10s | %-8s | %-12s | %-12s\n" \
                                "$fold" "$top_n" "$acc" "$correct" "$total" "$prec" "$rec"

                            # Track best TOP-1 accuracy for this model
                            if [[ "$top_n" == "1" && "$fold" != "avg" && "$fold" != "?" ]]; then
                                acc_num=$(echo "$acc" | tr ',' '.')
                                if (( $(awk "BEGIN { print ($acc_num > $best_acc) }") )); then
                                    best_acc=$acc_num
                                    best_fold=$fold
                                fi

                                # Track best TOP-1 precision for this model
                                prec_num=$(echo "$prec" | tr ',' '.')
                                if (( $(awk "BEGIN { print ($prec_num > $best_prec) }") )); then
                                    best_prec=$prec_num
                                    best_prec_fold=$fold
                                fi
                            fi
                        fi
                    done
                done

                # Store best model info
                if [[ -n "$best_fold" ]]; then
                    best_key="${base_model}|${version_core}|${sample_limit}"
                    best_models["$best_key"]="${best_acc}|${best_fold}"
                fi

                # Store best precision info
                if [[ -n "$best_prec_fold" ]]; then
                    best_prec_key="${base_model}|${version_core}|${sample_limit}"
                    best_precision_models["$best_prec_key"]="${best_prec}|${best_prec_fold}"
                fi
            fi
        done
    done
done

echo
echo "Summary Statistics"
echo "=================="
printf "%-45s | %-6s | %-10s | %-10s | %-10s | %-10s | %-11s\n" \
    "Base Model" "Ver" "Samples" "Mean_Acc1" "Mean_Acc3" "Best_Acc1" "Worst_Acc1"
echo "-------------------------------------------------------------------------------------------------------------------------------"

# Temporary file to hold mean Acc1 entries for top-N selection
tmpfile=$(mktemp)
trap 'rm -f "$tmpfile"' EXIT

# Calculate statistics per base model and version
for base_model in $(printf '%s\n' "${!base_models_seen[@]}" | sort); do
    for version_core in $(printf '%s\n' "${!version_cores_seen[@]}" | sort -V); do
        for sample_limit in $(printf '%s\n' "${!sample_limits_seen[@]}" | sort -n); do

            # Calculate statistics for TOP-1 and TOP-3
            total_acc1=0
            total_acc3=0
            count1=0
            count3=0
            best_acc1=-1
            worst_acc1=101

            for fold in 1 2 3 4 5; do # Only average known folds
                # TOP-1
                key1="${base_model}|${version_core}|${sample_limit}|${fold}|1"
                if [[ -n "${results[$key1]:-}" ]]; then
                    IFS='|' read -r acc total correct prec rec <<< "${results[$key1]:-}"
                    total_acc1=$(awk "BEGIN { print $total_acc1 + $acc }")
                    count1=$((count1 + 1))

                    # Track best and worst
                    if (( $(awk "BEGIN { print ($acc > $best_acc1) }") )); then
                        best_acc1=$acc
                    fi
                    if (( $(awk "BEGIN { print ($acc < $worst_acc1) }") )); then
                        worst_acc1=$acc
                    fi
                fi

                # TOP-3
                key3="${base_model}|${version_core}|${sample_limit}|${fold}|3"
                if [[ -n "${results[$key3]:-}" ]]; then
                    IFS='|' read -r acc total correct prec rec <<< "${results[$key3]:-}"
                    total_acc3=$(awk "BEGIN { print $total_acc3 + $acc }")
                    count3=$((count3 + 1))
                fi
            done

            # Print summary line if we have data
            if [[ $count1 -gt 0 ]]; then
                mean_acc1=$(awk "BEGIN { printf \"%.2f\", $total_acc1 / $count1 }")
                if [[ $count3 -gt 0 ]]; then
                    mean_acc3=$(awk "BEGIN { printf \"%.2f\", $total_acc3 / $count3 }")
                else
                    mean_acc3="N/A"
                fi

                if [[ "$best_acc1" == "-1" ]]; then best_acc1="N/A"; fi
                if [[ "$worst_acc1" == "101" ]]; then worst_acc1="N/A"; fi

                printf "%-45s | %-6s | %-10s | %-10s | %-10s | %-10s | %-11s\n" \
                    "$base_model" "$version_core" "$sample_limit" \
                    "${mean_acc1}%" "${mean_acc3}%" "${best_acc1}%" "${worst_acc1}%"

                # Save entry for Top-N selection
                echo "${mean_acc1}|${base_model}|${version_core}|${sample_limit}" >> "$tmpfile"
            fi
        done
    done
done

# Print Top-N models by mean TOP-1 accuracy
if [ -s "$tmpfile" ]; then
    awk -F'|' 'BEGIN{OFS=FS} { gsub(",",".",$1); print }' "$tmpfile" > "${tmpfile}.norm" && mv "${tmpfile}.norm" "$tmpfile"

    echo
    echo "Top-$TOP_N models by Mean TOP-1 accuracy (Acc1)"
    echo "-----------------------------------------------"
    printf "%3s %-45s %-6s %10s\n" "#" "Base Model" "Ver" "Mean_Acc1"

    rank=1
    sort -t'|' -k1,1nr "$tmpfile" | head -n "$TOP_N" | while IFS='|' read -r mean base ver samples; do
        mean_norm="${mean//,/.}"
        if ! awk -v m="$mean_norm" 'BEGIN{ if (m+0==m+0) exit 0; else exit 1 }'; then
            mean_norm=0
        fi
        mean_display=$(awk -v m="$mean_norm" 'BEGIN{printf "%.2f", m}')
        printf "%3d %-45s %-6s %8s%%\n" "$rank" "$base" "$ver" "$mean_display"
        rank=$((rank + 1))
    done
else
    echo
    echo "No summary entries found to compute Top-$TOP_N models."
fi

# Print best fold for each base model
echo
echo "Best Performing Fold by Base Model (TOP-1 Accuracy)"
echo "===================================================="
printf "%-45s | %-6s | %-10s | %-10s | %-6s\n" \
    "Base Model" "Ver" "Samples" "Best_Acc1" "Fold"
echo "--------------------------------------------------------------------------------------------"

for best_key in $(printf '%s\n' "${!best_models[@]}" | sort); do
    IFS='|' read -r base_model version_core sample_limit <<< "$best_key"
    IFS='|' read -r best_acc best_fold <<< "${best_models[$best_key]}"

    printf "%-45s | %-6s | %-10s | %-9s%% | %-6s\n" \
        "$base_model" "$version_core" "$sample_limit" "$best_acc" "$best_fold"
done

# Print best precision fold for each base model
echo
echo "Best Performing Fold by Base Model (TOP-1 Precision)"
echo "====================================================="
printf "%-45s | %-6s | %-10s | %-13s | %-6s\n" \
    "Base Model" "Ver" "Samples" "Best_Prec" "Fold"
echo "--------------------------------------------------------------------------------------------"

for best_prec_key in $(printf '%s\n' "${!best_precision_models[@]}" | sort); do
    IFS='|' read -r base_model version_core sample_limit <<< "$best_prec_key"
    IFS='|' read -r best_prec best_prec_fold <<< "${best_precision_models[$best_prec_key]}"

    printf "%-45s | %-6s | %-10s | %-12s%% | %-6s\n" \
        "$base_model" "$version_core" "$sample_limit" "$best_prec" "$best_prec_fold"
done

echo
echo "Done!"
