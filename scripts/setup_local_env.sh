echo "--- STARTING ENVIRONMENT SETUP ---"
# Init Miniconda in the script
source ~/miniconda3/etc/profile.d/conda.sh

# Create environment
conda env create -f config/environment.yml
conda activate drl-frameworks-env
# install packaged Horizon
pip install lib/horizon-0.1.tar.gz
pip install -e src/
# install some dependencies missing in environment.yml
pip install zmq psycopg2-binary networkx wget

# Setup Postgres container
docker build -t pg park/query-optimizer/docker/
docker start docker-pg || docker run --name docker-pg -p 0.0.0.0:5432:5432 --net drl-net --privileged -d pg

conda deactivate
echo "--- ENVIRONMENT SETUP COMPLETED ---"
