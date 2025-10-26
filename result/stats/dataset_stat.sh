#!/bin/bash

# Dataset Statistics Extractor
# Usage: ./extract_stats.sh [dataset_file1.txt] [dataset_file2.txt] ...
# If no arguments provided, processes all matching files in current directory

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to extract statistics from a single file
extract_stats() {
    local input_file="$1"
    
    if [[ ! -f "$input_file" ]]; then
        echo -e "${RED}Error: File '$input_file' not found${NC}"
        return 1
    fi
    
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Processing: ${YELLOW}$input_file${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"
    
    # Extract fold number from filename
    local fold=$(echo "$input_file" | grep -oP 'FOLD_\K[0-9]+' || echo "Unknown")
    echo -e "${BLUE}Fold Number:${NC} $fold"
    echo ""
    
    # Temporary files for processing
    local tmpdir=$(mktemp -d)
    local train_file="$tmpdir/train.txt"
    local val_file="$tmpdir/val.txt"
    local test_file="$tmpdir/test.txt"
    
    # Split file into sections
    awk '/^Training set/ {flag="train"; next} 
         /^Validation set/ {flag="val"; next} 
         /^Test set/ {flag="test"; next}
         flag=="train" && /\.png$/ {print > "'$train_file'"}
         flag=="val" && /\.png$/ {print > "'$val_file'"}
         flag=="test" && /\.png$/ {print > "'$test_file'"}' "$input_file"
    
    # Process each set
    for set_name in "Training" "Validation" "Test"; do
        local set_file=""
        case $set_name in
            "Training") set_file="$train_file" ;;
            "Validation") set_file="$val_file" ;;
            "Test") set_file="$test_file" ;;
        esac
        
        if [[ ! -s "$set_file" ]]; then
            echo -e "${YELLOW}${set_name} set: No data found${NC}\n"
            continue
        fi
        
        echo -e "${GREEN}━━━ ${set_name} Set ━━━${NC}"
        
        # Total files
        local total_files=$(wc -l < "$set_file")
        echo -e "  Total files: ${YELLOW}$total_files${NC}"
        
        # Extract unique documents (CTX... before the dash)
        local unique_docs=$(grep -oP 'CTX[0-9]+-' "$set_file" | sed 's/-$//' | sort -u | wc -l)
        echo -e "  Unique documents: ${YELLOW}$unique_docs${NC}"
        
        # Files per category (parent directory)
        echo -e "  ${BLUE}Files per category:${NC}"
        awk -F'/' '{print $(NF-1)}' "$set_file" | sort | uniq -c | sort -rn | \
            awk '{printf "    %-15s %s\n", $2":", $1}'
        
        # Unique documents per category
        echo -e "  ${BLUE}Unique documents per category:${NC}"
        for category in $(awk -F'/' '{print $(NF-1)}' "$set_file" | sort -u); do
            local cat_docs=$(grep "/$category/" "$set_file" | grep -oP 'CTX[0-9]+-' | sed 's/-$//' | sort -u | wc -l)
            printf "    %-15s %s\n" "$category:" "$cat_docs"
        done
        
        echo ""
    done
    
    # Overall statistics
    echo -e "${GREEN}━━━ Overall Statistics ━━━${NC}"
    
    local total_all=$(cat "$train_file" "$val_file" "$test_file" 2>/dev/null | wc -l)
    echo -e "  Total files (all sets): ${YELLOW}$total_all${NC}"
    
    local unique_docs_all=$(cat "$train_file" "$val_file" "$test_file" 2>/dev/null | \
        grep -oP 'CTX[0-9]+-' | sed 's/-$//' | sort -u | wc -l)
    echo -e "  Unique documents (all sets): ${YELLOW}$unique_docs_all${NC}"
    
    echo -e "  ${BLUE}Files per category (all sets):${NC}"
    cat "$train_file" "$val_file" "$test_file" 2>/dev/null | \
        awk -F'/' '{print $(NF-1)}' | sort | uniq -c | sort -rn | \
        awk '{printf "    %-15s %s\n", $2":", $1}'
    
    echo -e "  ${BLUE}Unique documents per category (all sets):${NC}"
    for category in $(cat "$train_file" "$val_file" "$test_file" 2>/dev/null | \
        awk -F'/' '{print $(NF-1)}' | sort -u); do
        local cat_docs=$(cat "$train_file" "$val_file" "$test_file" 2>/dev/null | \
            grep "/$category/" | grep -oP 'CTX[0-9]+-' | sed 's/-$//' | sort -u | wc -l)
        printf "    %-15s %s\n" "$category:" "$cat_docs"
    done
    
    # Distribution analysis
    echo ""
    echo -e "${GREEN}━━━ Distribution Analysis ━━━${NC}"
    
    local train_total=$(wc -l < "$train_file" 2>/dev/null || echo 0)
    local val_total=$(wc -l < "$val_file" 2>/dev/null || echo 0)
    local test_total=$(wc -l < "$test_file" 2>/dev/null || echo 0)
    
    if [[ $total_all -gt 0 ]]; then
        local train_pct=$(awk "BEGIN {printf \"%.2f\", ($train_total/$total_all)*100}")
        local val_pct=$(awk "BEGIN {printf \"%.2f\", ($val_total/$total_all)*100}")
        local test_pct=$(awk "BEGIN {printf \"%.2f\", ($test_total/$total_all)*100}")
        
        echo -e "  Training:   ${train_pct}%"
        echo -e "  Validation: ${val_pct}%"
        echo -e "  Test:       ${test_pct}%"
    fi
    
    # Cleanup
    rm -rf "$tmpdir"
    
    echo ""
}

# Main execution
main() {
    local files=("$@")
    
    # If no arguments, find all matching files
    if [[ ${#files[@]} -eq 0 ]]; then
        mapfile -t files < <(ls *_FOLD_*_DATASETS.txt 2>/dev/null)
        
        if [[ ${#files[@]} -eq 0 ]]; then
            echo -e "${RED}No dataset files found matching pattern: *_FOLD_*_DATASETS.txt${NC}"
            echo "Usage: $0 [dataset_file1.txt] [dataset_file2.txt] ..."
            exit 1
        fi
        
        echo -e "${CYAN}Found ${#files[@]} dataset file(s)${NC}"
    fi
    
    # Process each file
    for file in "${files[@]}"; do
        extract_stats "$file"
    done
    
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Processing complete!${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"
}

# Run main function
main "$@"
