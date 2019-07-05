echo "--- STARTING QOPT EVALUATION ---"
echo

echo "--- REMOVING PREVIOUS RESULTS ---"
rm -rf results/query_optimizer
echo

bash ./scripts/evaluation/query_optimizer/eval_dopamine.sh
bash ./scripts/evaluation/query_optimizer/eval_horizon.sh
bash ./scripts/evaluation/query_optimizer/eval_ray.sh

echo "--- QOPT EVALUATION COMPLETED ---"
echo