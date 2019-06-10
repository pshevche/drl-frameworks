source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING RAY SPACE INVADERS EXPERIMENTS ---"
conda activate drl-frameworks-env
mkdir -p results/space_invaders/runtime
echo
for fullfile in experiments/space_invaders/ray/*.yml; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/evaluation/clear_caches.sh
    export TUNE_RESULT_DIR=results/cartpole/${experiment}
    python src/ray/run_evaluation.py -f="experiments/space_invaders/ray/${experiment}.yml"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
conda deactivate
echo "--- RAY SPACE INVADERS EXPERIMENTS COMPLETED ---"
echo