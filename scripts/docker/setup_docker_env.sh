echo "--- STARTING ENVIRONMENT SETUP ---"
# Init Miniconda in the script
source ~/miniconda3/etc/profile.d/conda.sh

# Create environment
conda env create -f config/environment.yml
conda activate drl-frameworks-env
# install packaged Horizon
pip install lib/horizon-0.1.tar.gz
# install some dependencies missing in environment.yml
pip install zmq psycopg2-binary networkx wget
conda deactivate
echo "--- ENVIRONMENT SETUP COMPLETED ---"
