#!/bin/bash

# Full pipeline for vocal biomarker analysis

echo "========================================"
echo " Step 1: Transcribe and align audio"
echo "========================================"
python scripts/1_transcribe_align.py

echo ""
echo "========================================"
echo " Step 2: Extract speech features"
echo "========================================"
python scripts/2_extract_features.py

echo ""
echo "========================================"
echo " Step 3: Prepare classification datasets"
echo "========================================"
python scripts/3_prepare_classifications.py

echo ""
echo "========================================"
echo " Step 4: Train and evaluate models"
echo "========================================"
python scripts/4_run_modeling.py --start 1 --end 10

echo ""
echo "========================================"
echo " Step 5: Generate final report"
echo "========================================"
python utils/report_generator.py

echo ""
echo "Pipeline completed successfully!"