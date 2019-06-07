echo "--- STARTING CARTPOLE EVALUATION ---"
echo

echo "--- REMOVING PREVIOUS RESULTS ---"
rm -rf results/cartpole
echo

bash ./scripts/evaluation/cartpole/eval_dopamine.sh
bash ./scripts/evaluation/cartpole/eval_horizon.sh
bash ./scripts/evaluation/cartpole/eval_ray.sh

echo "--- CARTPOLE EVALUATION COMPLETED ---"
echo