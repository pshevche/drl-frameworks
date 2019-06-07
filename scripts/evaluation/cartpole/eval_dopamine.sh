source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING DOPAMINE CARTPOLE EXPERIMENTS ---"
conda activate drl-frameworks-env
mkdir -p results/cartpole/runtime
echo
for fullfile in experiments/cartpole/dopamine/*.gin; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/evaluation/clear_caches.sh
    python src/dopamine/run_evaluation.py --base_dir="results/cartpole/" --gin_files="experiments/cartpole/dopamine/${experiment}.gin"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
conda deactivate
echo "--- DOPAMINE CARTPOLE EXPERIMENTS COMPLETED ---"
echo
