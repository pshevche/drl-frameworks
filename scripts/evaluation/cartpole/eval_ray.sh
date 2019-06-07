source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING RAY CARTPOLE EXPERIMENTS ---"
conda activate ray-env
mkdir -p results/cartpole/runtime
echo
for fullfile in experiments/cartpole/ray/*.yml; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/evaluation/clear_caches.sh
    python src/ray/run_evaluation.py -f="experiments/cartpole/ray/${experiment}.yml"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
rm -rf ~/ray_results
conda deactivate
echo "--- RAY CARTPOLE EXPERIMENTS COMPLETED ---"
echo