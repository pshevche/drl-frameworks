source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING RAY EXPERIMENTS ---"
conda activate ray-env
echo
echo "--- STARTING RAY CARTPOLE EXPERIMENTS ---"
mkdir -p results/cartpole/runtime
echo
for fullfile in src/ray/experiments/cartpole/*.yml; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    python src/ray/run_evaluation.py -f="src/ray/experiments/cartpole/${experiment}.yml"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
echo "--- RAY CARTPOLE EXPERIMENTS COMPLETED ---"
echo
rm -rf ~/ray_results
echo "--- RAY EXPERIMENTS COMPLETED ---"
echo