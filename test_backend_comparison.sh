#!/bin/bash
#
# Backend Comparison Script: EZKL vs JSTprove
#
# This script runs the full inference pipeline with both backends
# and compares their performance, predictions, and execution methods.
#
# Usage:
#   ./test_backend_comparison.sh
#
# Requirements:
#   - dsperse environment activated
#   - Doom model slices available
#   - Both EZKL and JSTprove backends configured
#

set -e  # Exit on any error

MODEL_DIR="dsperse/models/doom"
SLICES_DIR="$MODEL_DIR/slices"
INPUT_FILE="$MODEL_DIR/input.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "BACKEND COMPARISON: EZKL vs JSTprove"
echo "========================================="

# Check prerequisites
if [ ! -d "$SLICES_DIR" ]; then
    echo -e "${RED}Error: Slices directory not found: $SLICES_DIR${NC}"
    echo "Please ensure the Doom model is properly sliced."
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    echo "Please ensure the Doom model input.json exists."
    exit 1
fi

# Clean previous runs
echo "Cleaning previous runs..."
rm -rf $MODEL_DIR/run/*
rm -f /tmp/ezkl_*.json /tmp/jstprove_*.json

# Activate environment
echo "Activating environment..."
source .env/bin/activate

echo ""
echo "--- STEP 1: COMPILE (EZKL) ---"
if time DSPERSE_BACKEND=ezkl dsperse compile --slices-dir $SLICES_DIR 2>&1; then
    echo -e "${GREEN}EZKL compilation: SUCCESS${NC}"
else
    echo -e "${YELLOW}EZKL compilation: SKIPPED (may not be available)${NC}"
fi

echo ""
echo "--- STEP 2: COMPILE (JSTprove) ---"
# Clean to force recompilation
rm -rf $MODEL_DIR/run/*
if time DSPERSE_BACKEND=jstprove dsperse compile --slices-dir $SLICES_DIR 2>&1; then
    echo -e "${GREEN}JSTprove compilation: SUCCESS${NC}"
else
    echo -e "${YELLOW}JSTprove compilation: SKIPPED (may not be available)${NC}"
fi

echo ""
echo "--- STEP 3: RUN INFERENCE (EZKL) ---"
rm -rf $MODEL_DIR/run/*
if time DSPERSE_BACKEND=ezkl dsperse run --slices-dir $SLICES_DIR --input-file $INPUT_FILE --output-file /tmp/ezkl_run.json 2>&1; then
    echo -e "${GREEN}EZKL inference: SUCCESS${NC}"
else
    echo -e "${RED}EZKL inference: FAILED${NC}"
fi

echo ""
echo "--- STEP 4: RUN INFERENCE (JSTprove) ---"
rm -rf $MODEL_DIR/run/*
if time DSPERSE_BACKEND=jstprove dsperse run --slices-dir $SLICES_DIR --input-file $INPUT_FILE --output-file /tmp/jstprove_run.json 2>&1; then
    echo -e "${GREEN}JSTprove inference: SUCCESS${NC}"
else
    echo -e "${RED}JSTprove inference: FAILED${NC}"
fi

echo ""
echo "--- STEP 5: PROVE (EZKL) ---"
# Note: EZKL prove requires keys to be generated first
# This may fail without proper setup
if time DSPERSE_BACKEND=ezkl dsperse prove --slices-dir $SLICES_DIR --input-file $INPUT_FILE --proof-output /tmp/ezkl_proof.json 2>&1; then
    echo -e "${GREEN}EZKL prove: SUCCESS${NC}"
else
    echo -e "${YELLOW}EZKL prove: SKIPPED (needs keys)${NC}"
fi

echo ""
echo "--- STEP 6: PROVE (JSTprove) ---"
rm -rf $MODEL_DIR/run/*
if time DSPERSE_BACKEND=jstprove dsperse prove --slices-dir $SLICES_DIR --input-file $INPUT_FILE --proof-output /tmp/jstprove_proof.json 2>&1; then
    echo -e "${GREEN}JSTprove prove: SUCCESS${NC}"
else
    echo -e "${YELLOW}JSTprove prove: SKIPPED (needs setup)${NC}"
fi

echo ""
echo "========================================="
echo "COMPARISON SUMMARY"
echo "========================================="

echo ""
echo "EZKL Output:"
cat /tmp/ezkl_run.json | jq '.prediction, .probabilities' 2>/dev/null || echo "No output"

echo ""
echo "JSTprove Output:"
cat /tmp/jstprove_run.json | jq '.prediction, .probabilities' 2>/dev/null || echo "No output"

echo ""
echo "Segment execution methods (EZKL):"
cat /tmp/ezkl_run.json | jq '.slice_results | to_entries | map({segment: .key, method: .value.method})' 2>/dev/null || echo "No data"

echo ""
echo "Segment execution methods (JSTprove):"
cat /tmp/jstprove_run.json | jq '.slice_results | to_entries | map({segment: .key, method: .value.method})' 2>/dev/null || echo "No data"

