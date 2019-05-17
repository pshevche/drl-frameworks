source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING HORIZON EXPERIMENTS ---"
conda activate horizon-env
echo
echo "--- STARTING HORIZON CARTPOLE EXPERIMENTS ---"
mkdir -p results/cartpole/runtime
echo
for fullfile in experiments/cartpole/horizon/cpu/*.json; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/clean_caches.sh
    mkdir -p results/cartpole/${experiment}
    python src/horizon/run_evaluation.py -p experiments/cartpole/horizon/cpu/${experiment}.json -f results/cartpole/${experiment}/checkpoints.json -v results/cartpole/
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
for fullfile in experiments/cartpole/horizon/gpu/*.json; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    bash ./scripts/clean_caches.sh
    mkdir -p results/cartpole/${experiment}
    python src/horizon/run_evaluation.py -g 0 -p experiments/cartpole/horizon/gpu/${experiment}.json -f results/cartpole/${experiment}/checkpoints.json -v results/cartpole/
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
echo "--- HORIZON CARTPOLE EXPERIMENTS COMPLETED ---"
echo
echo "--- HORIZON EXPERIMENTS COMPLETED ---"
echo