#!/bin/bash

# Find files not used in specified dataset(s) across all folds
# Usage: ./find_unused_csv.sh [-tvs] input.csv [fold1.txt] [fold2.txt] ...
# Or: ./find_unused_csv.sh -t -v -s input.csv result/stats/*_FOLD_*_DATASETS.txt

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Function to extract files from specified dataset sections
extract_training_files() {
    local file="$1"
    local get_t="$2" # Training set
    local get_v="$3" # Validation set
    local get_s="$4" # Test set

    local start_patterns=()
    local stop_patterns=()

    # Define what to extract
    [[ "$get_t" -eq 1 ]] && start_patterns+=("^Training set")
    [[ "$get_v" -eq 1 ]] && start_patterns+=("^Validation set")
    [[ "$get_s" -eq 1 ]] && start_patterns+=("^Test set")

    # Define what *stops* extraction (any set we are *not* extracting)
    [[ "$get_t" -eq 0 ]] && stop_patterns+=("^Training set")
    [[ "$get_v" -eq 0 ]] && stop_patterns+=("^Validation set")
    [[ "$get_s" -eq 0 ]] && stop_patterns+=("^Test set")

    # Build awk regexes
    local start_regex=$(IFS='|'; echo "${start_patterns[*]}")
    local stop_regex=$(IFS='|'; echo "${stop_patterns[*]}")

    # If no start patterns, print nothing
    if [[ -z "$start_regex" ]]; then
        return
    fi

    # Build awk script
    local awk_script
    # Start pattern sets the flag
    awk_script="/${start_regex}/ {flag=1; next} "

    # Only add stop regex if it's not empty (i.e., we're not extracting all sets)
    if [[ -n "$stop_regex" ]]; then
        awk_script+="/${stop_regex}/ {flag=0} "
    fi

    # If flag is set and line is a png, print it
    awk_script+="flag && /\.png$/ {print}"
    
    awk "$awk_script" "$file"
}

# Function to extract document ID and page from a file path
# E.g., /path/to/CTX192000019-7.png -> CTX192000019,7
extract_doc_page() {
    local filepath="$1"
    local filename=$(basename "$filepath")
    # Extract CTX ID and page number from filename like CTX192000019-7.png
    if [[ $filename =~ (CTX[0-9]+)-([0-9]+)\.png ]]; then
        echo "${BASH_REMATCH[1]},${BASH_REMATCH[2]}"
    fi
}

main() {
    local GET_TRAIN=0
    local GET_VALID=0
    local GET_TEST=0

    # Parse options
    while getopts "vst" opt; do
        case $opt in
            v) GET_VALID=1 ;;
            s) GET_TEST=1 ;;
            t) GET_TRAIN=1 ;;
            \?) echo -e "${RED}Invalid option: -$OPTARG${NC}" >&2; exit 1 ;;
        esac
    done
    shift $((OPTIND - 1))

    # Default to training set if no flags are provided
    if [[ $GET_TRAIN -eq 0 && $GET_VALID -eq 0 && $GET_TEST -eq 0 ]]; then
        GET_TRAIN=1
    fi

    if [[ $# -lt 2 ]]; then
        echo -e "${RED}Error: Not enough arguments${NC}"
        echo "Usage: $0 [-tvs] <input.csv> [fold1.txt] [fold2.txt] ..."
        echo "  Flags (can be combined):"
        echo "    -t: Filter items from Training sets (default if no flags given)"
        echo "    -v: Filter items from Validation sets"
        echo "    -s: Filter items from Test sets"
        echo "Example: $0 -t -v input.csv result/stats/*_FOLD_*_DATASETS.txt"
        exit 1
    fi
    
    local input_csv="$1"
    shift
    local fold_files=("$@")
    
    # Validate input CSV exists
    if [[ ! -f "$input_csv" ]]; then
        echo -e "${RED}Error: Input CSV not found: $input_csv${NC}"
        exit 1
    fi
    
    # Validate all fold files exist
    for file in "${fold_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo -e "${RED}Error: File not found: $file${NC}"
            exit 1
        fi
    done
    
    # Build description of sets being processed
    local sets_desc_parts=()
    [[ $GET_TRAIN -eq 1 ]] && sets_desc_parts+=("Training")
    [[ $GET_VALID -eq 1 ]] && sets_desc_parts+=("Validation")
    [[ $GET_TEST -eq 1 ]] && sets_desc_parts+=("Test")
    local sets_desc=$(IFS=', '; echo "${sets_desc_parts[*]}")

    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Finding CSV entries not used in any ${sets_desc} set(s)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "${YELLOW}Input CSV:${NC} $(basename "$input_csv")"
    echo -e "${YELLOW}Processing ${#fold_files[@]} fold(s):${NC}"
    for file in "${fold_files[@]}"; do
        echo -e "  - $(basename "$file")"
    done
    echo ""
    
    # Create temporary directory
    local tmpdir=$(mktemp -d)
    local training_files="$tmpdir/training_files.txt"
    local training_doc_pages="$tmpdir/training_doc_pages.txt"
    local input_doc_pages="$tmpdir/input_doc_pages.txt"
    local unused_doc_pages="$tmpdir/unused_doc_pages.txt"
    
    # Extract all files from all specified sets in all folds
    echo -e "${BLUE}Step 1: Collecting files from ${sets_desc} set(s)...${NC}"
    for file in "${fold_files[@]}"; do
        extract_training_files "$file" "$GET_TRAIN" "$GET_VALID" "$GET_TEST"
    done | sort -u > "$training_files"
    
    local total_training=$(wc -l < "$training_files")
    echo -e "  Found ${YELLOW}${total_training}${NC} unique files in specified set(s)\n"
    
    # Convert training files to document,page format
    echo -e "${BLUE}Step 2: Extracting document IDs and pages from files...${NC}"
    while IFS= read -r filepath; do
        extract_doc_page "$filepath"
    done < "$training_files" | grep -v '^$' | sort -u > "$training_doc_pages"
    
    local total_training_docs=$(wc -l < "$training_doc_pages")
    echo -e "  Extracted ${YELLOW}${total_training_docs}${NC} unique document-page combinations\n"
    
    # Read input CSV (skip header) and extract file,page combinations
    echo -e "${BLUE}Step 3: Processing input CSV...${NC}"
    tail -n +2 "$input_csv" | awk -F',' '{print $1","$2}' | sort -u > "$input_doc_pages"
    
    local total_input=$(wc -l < "$input_doc_pages")
    echo -e "  Found ${YELLOW}${total_input}${NC} unique document-page combinations in input CSV\n"
    
    # Find document-page combinations not in training sets
    echo -e "${BLUE}Step 4: Finding unused entries...${NC}"
    comm -23 "$input_doc_pages" "$training_doc_pages" > "$unused_doc_pages"
    
    local total_unused=$(wc -l < "$unused_doc_pages")
    echo -e "  Found ${YELLOW}${total_unused}${NC} entries not used in any specified set(s)\n"
    
    # Statistics
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Summary Statistics${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "  Total entries in CSV:              ${YELLOW}${total_input}${NC}"
    echo -e "  Entries in specified set(s):     ${YELLOW}$((total_input - total_unused))${NC}"
    echo -e "  Entries NOT in specified set(s): ${YELLOW}${total_unused}${NC}"
    
    if [[ $total_input -gt 0 ]]; then
        local pct_unused=$(awk "BEGIN {printf \"%.2f\", ($total_unused/$total_input)*100}")
        echo -e "  Percentage unused:                 ${YELLOW}${pct_unused}%${NC}"
    fi
    echo ""
    
    # Generate output CSV
    local output_file="unused_entries_$(date +%Y%m%d-%H%M%S).csv"
    
    if [[ $total_unused -gt 0 ]]; then
        echo -e "${BLUE}Step 5: Generating output CSV...${NC}"
        
        # Write header
        echo "file,page,category" > "$output_file"
        
        # Filter input CSV to only include unused entries
        # Read header from input
        local header=$(head -n 1 "$input_csv")
        
        # For each unused doc-page, find matching rows in input CSV
        while IFS=',' read -r doc page; do
            # Match rows where first column is $doc and second column is $page
            awk -F',' -v doc="$doc" -v page="$page" \
                'NR>1 && $1==doc && $2==page {print}' "$input_csv"
        done < "$unused_doc_pages" >> "$output_file"
        
        local output_rows=$(tail -n +2 "$output_file" | wc -l)
        echo -e "  Written ${YELLOW}${output_rows}${NC} rows to output CSV\n"
        
        # Category breakdown
        if [[ $output_rows -gt 0 ]]; then
            echo -e "${GREEN}━━━ Unused Entries by Category ━━━${NC}"
            tail -n +2 "$output_file" | awk -F',' '{print $3}' | sort | uniq -c | sort -rn | \
                awk '{printf "  %-15s %s entries\n", $2":", $1}'
            echo ""
        fi
        
        # Output information
        echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}Output${NC}"
        echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"
        
        echo -e "${YELLOW}Output CSV saved to: ${output_file}${NC}\n"
        
        echo -e "To view the output:"
        echo -e "  ${BLUE}cat ${output_file}${NC}"
        echo -e "  ${BLUE}column -t -s, ${output_file} | less -S${NC}\n"
        
        echo -e "To filter by category:"
        echo -e "  ${BLUE}awk -F',' '\$3==\"TEXT\"' ${output_file}${NC}"
        echo -e "  ${BLUE}awk -F',' '\$3==\"LINE_HW\"' ${output_file}${NC}\n"
        
        # Show preview
        echo -e "${GREEN}━━━ Preview (first 10 rows) ━━━${NC}"
        head -n 11 "$output_file" | column -t -s,
        echo ""
        
    else
        echo -e "${GREEN}All CSV entries are used in at least one specified set!${NC}\n"
    fi
    
    # Cleanup
    rm -rf "$tmpdir"
    
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Done!${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"
}

# Run main
main "$@"
