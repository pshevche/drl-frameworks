source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING HORIZON QOPT EXPERIMENTS ---"
conda activate drl-frameworks-env
mkdir -p results/query_optimizer/runtime
echo
for fullfile in experiments/query_optimizer/horizon/cpu/*.json; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/evaluation/clear_caches.sh
    mkdir -p results/query_optimizer/${experiment}
    python drl_fw/horizon/run_evaluation.py -p experiments/query_optimizer/horizon/cpu/${experiment}.json -f results/query_optimizer/${experiment}/checkpoints.json -v results/query_optimizer/
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
for fullfile in experiments/query_optimizer/horizon/gpu/*.json; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/evaluation/clear_caches.sh
    mkdir -p results/query_optimizer/${experiment}
    python drl_fw/horizon/run_evaluation.py -g 0 -p experiments/query_optimizer/horizon/gpu/${experiment}.json -f results/query_optimizer/${experiment}/checkpoints.json -v results/query_optimizer/
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
conda deactivate
echo "--- HORIZON QOPT EXPERIMENTS COMPLETED ---"
echo