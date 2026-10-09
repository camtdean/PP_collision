#!/bin/bash

set -e

echo "=============================================="
echo " Generating 200 vertex test events"
echo "=============================================="

mkdir -p vertex_test_events

for i in {0..199}; do
    echo ""
    echo "========== Event $i / 199 =========="

    python test_vertex_simple.py \
        --use-helix \
        --seed "$i" \
        --output "vertex_test_events/vertex_test_inputs_$(printf "%04d" "$i").pt"
done

echo ""
echo "=============================================="
echo " All 200 events generated."
echo " Running truth-vs-reco..."
echo "=============================================="

python plot_truth_vs_reco.py --use-helix

echo ""
echo "=============================================="
echo " Finished."
echo " Output: truth_vs_reco.png"
echo "=============================================="
