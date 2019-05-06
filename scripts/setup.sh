echo "--- STARTING SETUP ---"
# Init Miniconda in the script
source ~/miniconda3/etc/profile.d/conda.sh

# Create environments
echo "--- SETTING UP DOPAMINE ENVIRONMENT ---"
conda env create -f config/dopamine.yml
echo "--- SETTING UP HORIZON ENVIRONMENT ---"
conda env create -f config/horizon.yml
conda activate horizon-env
cd src/horizon/Horizon
thrift --gen py --out . ml/rl/thrift/core.thrift
pip install -e .
cd ../../..
conda deactivate
echo "--- SETTING UP RAY ENVIRONMENT ---"
conda env create -f config/ray.yml
echo "--- SETUP COMPLETED ---"
