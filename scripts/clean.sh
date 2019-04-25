# remove dopamine results
find src/dopamine -type d -name __pycache__ -exec rm -rf {} \;
find src/dopamine -type d -name results -exec rm -rf {} \;

# remove ray results
find src/ray -type d -name results -exec rm -rf {} \;
