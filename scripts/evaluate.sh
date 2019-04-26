echo "--- STARTING EVALUATION ---"
echo "--- REMOVING PREVIOUS RESULTS ---"
bash ./scripts/clean.sh
echo "--- CONFIGURING ANACONDA ---"
source ~/miniconda3/etc/profile.d/conda.sh

echo "--- STARTING DOPAMINE EXPERIMENTS ---"
conda activate dopamine-env
echo "--- STARTING DOPAMINE CARTPOLE EXPERIMENTS ---"
for fullfile in src/dopamine/experiments/cartpole/*.gin; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    python src/dopamine/run_evaluation.py --base_dir="src/dopamine/results/${experiment}" --gin_files="src/dopamine/experiments/cartpole/${experiment}.gin"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
done
echo "--- DOPAMINE CARTPOLE EXPERIMENTS COMPLETED ---"
echo "--- DOPAMINE EXPERIMENTS COMPLETED ---"

echo "--- STARTING RAY EXPERIMENTS ---"
conda activate ray-env
echo "--- STARTING RAY CARTPOLE EXPERIMENTS ---"
for fullfile in src/ray/experiments/cartpole/*.yml; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    python src/ray/run_evaluation.py -f="src/ray/experiments/cartpole/${experiment}.yml"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
done
echo "--- RAY CARTPOLE EXPERIMENTS COMPLETED ---"
echo "--- RAY EXPERIMENTS COMPLETED ---"

echo "--- STARTING HORIZON EXPERIMENTS ---"
conda activate horizon-env
echo "--- STARTING HORIZON CARTPOLE EXPERIMENTS ---"
for fullfile in src/horizon/experiments/cartpole/*.json; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    mkdir -p src/horizon/results/${experiment}
    python src/horizon/run_evaluation.py -p src/horizon/experiments/cartpole/${experiment}.json -f src/horizon/results/${experiment}/checkpoints_${experiment}.json -r src/horizon/results/${experiment}/rewards_${experiment}.csv
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
done
echo "--- HORIZON CARTPOLE EXPERIMENTS COMPLETED ---"
echo "--- HORIZON EXPERIMENTS COMPLETED ---"

conda deactivate
echo "--- EVALUATION COMPLETED ---"
