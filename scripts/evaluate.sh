cd ..
for fullfile in src/dopamine/experiments/*.gin; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    python src/dopamine/evaluation.py --base_dir="src/dopamine/results/${experiment}" --gin_files="src/dopamine/experiments/${experiment}.gin"
done
