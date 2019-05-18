source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING DOPAMINE EXPERIMENTS ---"
conda activate dopamine-env
echo

echo "--- STARTING DOPAMINE CARTPOLE EXPERIMENTS ---"
mkdir -p results/cartpole/runtime
echo
for fullfile in experiments/cartpole/dopamine/*.gin; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/clean_caches.sh
    python src/dopamine/run_evaluation.py --base_dir="results/cartpole/" --gin_files="experiments/cartpole/dopamine/${experiment}.gin"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
echo "--- DOPAMINE CARTPOLE EXPERIMENTS COMPLETED ---"
echo

echo "--- STARTING DOPAMINE SPACE INVADERS EXPERIMENTS ---"
mkdir -p results/space_invaders/runtime
echo
for fullfile in experiments/space_invaders/dopamine/*.gin; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/clean_caches.sh
    python src/dopamine/run_evaluation.py --base_dir="results/space_invaders/" --gin_files="experiments/space_invaders/dopamine/${experiment}.gin"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
echo "--- DOPAMINE SPACE INVADERS EXPERIMENTS COMPLETED ---"
echo

echo "--- DOPAMINE EXPERIMENTS COMPLETED ---"
echo