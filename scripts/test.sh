source ~/miniconda3/etc/profile.d/conda.sh
conda activate drl-frameworks-env
export TUNE_RESULT_DIR=test/drl_fw/test_results
pytest --show-capture=no --timeout=180 test/
conda deactivate