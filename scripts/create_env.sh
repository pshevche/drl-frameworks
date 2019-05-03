# Init Miniconda in the script
source ~/miniconda3/etc/profile.d/conda.sh

# Create environments
conda env create -f config/dopamine.yml
conda env create -f config/horizon.yml
conda env create -f config/ray.yml
