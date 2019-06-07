echo "--- STARTING SPACE INVADERS EVALUATION ---"
echo

echo "--- REMOVING PREVIOUS RESULTS ---"
rm -rf results/space_invaders
echo

bash ./scripts/evaluation/space_invaders/eval_dopamine.sh
bash ./scripts/evaluation/space_invaders/eval_ray.sh

echo "--- SPACE INVADERS EVALUATION COMPLETED ---"
echo