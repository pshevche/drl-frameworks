source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING RAY QOPT EXPERIMENTS ---"
conda activate drl-frameworks-env
mkdir -p results/query_optimizer/runtime
echo
for fullfile in experiments/query_optimizer/ray/*.yml; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/evaluation/clear_caches.sh
    export TUNE_RESULT_DIR=results/query_optimizer/${experiment}
    python src/drl_fw/ray/run_evaluation.py -f="experiments/query_optimizer/ray/${experiment}.yml"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
conda deactivate
echo "--- RAY QOPT EXPERIMENTS COMPLETED ---"
echo