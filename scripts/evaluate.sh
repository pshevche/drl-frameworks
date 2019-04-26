echo "--- STARTING EVALUATION ---"
echo

echo "--- REMOVING PREVIOUS RESULTS ---"
bash ./scripts/clean.sh
echo

echo "--- CONFIGURING ANACONDA ---"
source ~/miniconda3/etc/profile.d/conda.sh
echo

echo "--- STARTING DOPAMINE EXPERIMENTS ---"
conda activate dopamine-env
echo
echo "--- STARTING DOPAMINE CARTPOLE EXPERIMENTS ---"
echo
for fullfile in src/dopamine/experiments/cartpole/*.gin; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    python src/dopamine/run_evaluation.py --base_dir="src/dopamine/results/${experiment}" --gin_files="src/dopamine/experiments/cartpole/${experiment}.gin"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
echo "--- DOPAMINE CARTPOLE EXPERIMENTS COMPLETED ---"
echo
echo "--- DOPAMINE EXPERIMENTS COMPLETED ---"
echo

echo "--- STARTING RAY EXPERIMENTS ---"
conda activate ray-env
echo
echo "--- STARTING RAY CARTPOLE EXPERIMENTS ---"
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
echo "--- RAY EXPERIMENTS COMPLETED ---"
echo

echo "--- STARTING HORIZON EXPERIMENTS ---"
conda activate horizon-env
echo
echo "--- STARTING HORIZON CARTPOLE EXPERIMENTS ---"
echo
for fullfile in src/horizon/experiments/cartpole/cpu/*.json; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    mkdir -p src/horizon/results/${experiment}
    python src/horizon/run_evaluation.py -p src/horizon/experiments/cartpole/cpu/${experiment}.json -f src/horizon/results/${experiment}/checkpoints_${experiment}.json -r src/horizon/results/${experiment}/rewards_${experiment}.csv
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
for fullfile in src/horizon/experiments/cartpole/gpu/*.json; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    mkdir -p src/horizon/results/${experiment}
    python src/horizon/run_evaluation.py -g 0 -p src/horizon/experiments/cartpole/gpu/${experiment}.json -f src/horizon/results/${experiment}/checkpoints_${experiment}.json -r src/horizon/results/${experiment}/rewards_${experiment}.csv
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
echo "--- HORIZON CARTPOLE EXPERIMENTS COMPLETED ---"
echo
echo "--- HORIZON EXPERIMENTS COMPLETED ---"
echo

conda deactivate
echo "--- EVALUATION COMPLETED ---"
