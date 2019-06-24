source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING DOPAMINE SPACE INVADERS EXPERIMENTS ---"
conda activate drl-frameworks-env
mkdir -p results/space_invaders/runtime
echo
for fullfile in experiments/space_invaders/dopamine/*.gin; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/evaluation/clear_caches.sh
    python drl_fw/dopamine/run_evaluation.py --base_dir="results/space_invaders/" --gin_files="experiments/space_invaders/dopamine/${experiment}.gin"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
conda deactivate
echo "--- DOPAMINE SPACE INVADERS EXPERIMENTS COMPLETED ---"
echo
