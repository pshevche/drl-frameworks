source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING HORIZON SPACE INVADERS EXPERIMENTS ---"
conda activate horizon-env
mkdir -p results/space_invaders/runtime
echo
for fullfile in experiments/space_invaders/horizon/cpu/*.json; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/evaluation/clear_caches.sh
    mkdir -p results/space_invaders/${experiment}
    python src/horizon/run_evaluation.py -p experiments/space_invaders/horizon/cpu/${experiment}.json -f results/space_invaders/${experiment}/checkpoints.json -v results/space_invaders/
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
for fullfile in experiments/space_invaders/horizon/gpu/*.json; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/evaluation/clear_caches.sh
    mkdir -p results/space_invaders/${experiment}
    python src/horizon/run_evaluation.py -g 0 -p experiments/space_invaders/horizon/gpu/${experiment}.json -f results/space_invaders/${experiment}/checkpoints.json -v results/space_invaders/
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
conda deactivate
echo "--- HORIZON SPACE INVADERS EXPERIMENTS COMPLETED ---"
echo