echo "--- STARTING SETUP ---"
# Init Miniconda in the script
source ~/miniconda3/etc/profile.d/conda.sh

# Create environments
echo "--- SETTING UP DOPAMINE ENVIRONMENT ---"
conda env create -f config/dopamine.yml
conda activate dopamine-env
pip install lib/dopamine_rl-2.0.3.tar.gz
conda deactivate
echo "--- SETTING UP HORIZON ENVIRONMENT ---"
conda env create -f config/horizon.yml
conda activate horizon-env
pip install lib/horizon-0.1.tar.gz
conda deactivate
echo "--- SETTING UP RAY ENVIRONMENT ---"
conda env create -f config/ray.yml
conda activate ray-env
pip install lib/ray-0.7.0-cp36-cp36m-manylinux1_x86_64.whl
conda deactivate
echo "--- SETUP COMPLETED ---"
