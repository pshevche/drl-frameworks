echo "--- STARTING SETUP ---"
# Init Miniconda in the script
source ~/miniconda3/etc/profile.d/conda.sh

# Create environments
echo "--- SETTING UP DOPAMINE ENVIRONMENT ---"
conda env create -f config/dopamine.yml
echo "--- SETTING UP HORIZON ENVIRONMENT ---"
conda env create -f config/horizon.yml
conda activate horizon-env
pip install lib/horizon-0.1.tar.gz
conda deactivate
echo "--- SETTING UP RAY ENVIRONMENT ---"
conda env create -f config/ray.yml
echo "--- SETUP COMPLETED ---"
