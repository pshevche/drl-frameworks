echo "--- STARTING EVALUATION ---"
echo

echo "--- REMOVING PREVIOUS RESULTS ---"
rm -rf results/
bash ./scripts/clean_caches.sh
echo

echo "--- CONFIGURING ANACONDA ---"
source ~/miniconda3/etc/profile.d/conda.sh
echo

bash ./scripts/evaluate_dopamine.sh
bash ./scripts/evaluate_horizon.sh
bash ./scripts/evaluate_ray.sh

conda deactivate
echo "--- EVALUATION COMPLETED ---"
