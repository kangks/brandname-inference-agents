#!/bin/bash

# Title: batch_compare.sh
# Description: This script sends multiple product names to an inference API,
#              parses the JSON response for each, and displays the results
#              in formatted comparison tables.
#
# Dependencies: jq, column

# --- Helper Function: Check for required commands ---
check_dependencies() {
    for cmd in jq column; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "Error: Required command '$cmd' is not found." >&2
            echo "Please install it to run this script." >&2
            exit 1
        fi
    done
}

# --- Helper Function: Process a single JSON response and print a table ---
# Takes the full JSON string as its first argument ($1)
process_api_response() {
    local json_input="$1"

    # Check if the API returned anything.
    if [ -z "$json_input" ]; then
        echo "Error: Received an empty response from the API." >&2
        return 1
    fi

    # Use jq to extract and format the data as Tab-Separated Values (TSV).
    # -r flag outputs raw text; 'select' handles cases where the path doesn't exist.
    local table_content
    table_content=$(echo "$json_input" | jq -r '
        .result.result.all_predictions | select(. != null) | .[] |
        [.[2], .[0], .[1]] |
        @tsv
    ')

    # Check if the 'all_predictions' array was found and parsed.
    if [ -z "$table_content" ]; then
        echo "Warning: Could not find or parse '.result.result.all_predictions' in the API response." >&2
        # Optionally print the raw response for debugging:
        # echo "--- Raw API Response ---"
        # echo "$json_input"
        # echo "------------------------"
        return 1
    fi

    # Print the final formatted table.
    # The 'column' command formats the tab-separated input into aligned columns.
    (echo -e "Method\tPrediction\tConfidence" && echo "$table_content") | column -t -s $'\t'
}

# --- Main Execution ---

# 1. Configuration
# ------------------
API_URL="http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer"

# Define the array of product names to process.
# You can add or remove products from this list.
declare -a PRODUCT_NAMES=(
    "CLEAR NOSE Moist Skin Barrier Moisturizing Gel 120ml เคลียร์โนส มอยส์เจอไรซิ่งเจล เฟเชียล."
    "Dr.Althea Cream ด๊อกเตอร์อัลเทีย ครีมบำรุงผิวหน้า 50ml (345 Relief/147 Barrier)"
    "Eucerin Spotless Brightening Skin Tone Perfecting Body Lotion 250ml ยูเซอริน ผลิตภัณฑ์บำรุงผิวกาย"
    "[โปรแรง]กันแดดเคลียร์โนส Clear Nose UV Sun Serum SPF50+PA++++ 80ml 1ชิ้น(CUV)"
    "ODBO ICONIC EYESHADOW PALETTE - OD2029"
    "Oriental Princess ครีมทาผิว Oriental Beauty Body Lotion 400 ml"
)

# 2. Run Dependency Check
# ------------------
check_dependencies

# 3. Loop and Process
# ------------------
# Iterate safely over the array of product names.
for name in "${PRODUCT_NAMES[@]}"; do
    echo "======================================================================="
    echo "🔎 Querying for: $name"
    echo "======================================================================="

    # Safely create the JSON payload using jq to handle special characters.
    json_payload=$(jq -n --arg pn "$name" '{
        "product_name": $pn,
        "language_hint": "en",
        "method": "orchestrator"
    }')

    # Execute the curl request and store the server's response.
    # The '-s' flag enables silent mode to hide progress meters.
    api_response=$(curl -s -X POST "$API_URL" \
        -H 'Content-Type: application/json' \
        -d "$json_payload")

    # Process the captured response with our helper function.
    process_api_response "$api_response"

    # Add a newline for better readability between entries.
    echo ""
done