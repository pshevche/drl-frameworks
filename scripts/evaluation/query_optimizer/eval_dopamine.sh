source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING DOPAMINE QOPT EXPERIMENTS ---"
conda activate drl-frameworks-env
mkdir -p results/query_optimizer/runtime
echo
for fullfile in experiments/query_optimizer/dopamine/*.gin; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    python src/drl_fw/dopamine/run_evaluation.py --base_dir="results/query_optimizer/" --gin_files="experiments/query_optimizer/dopamine/${experiment}.gin"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
conda deactivate
echo "--- DOPAMINE QOPT EXPERIMENTS COMPLETED ---"
echo
