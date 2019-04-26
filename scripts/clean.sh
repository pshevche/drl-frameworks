# remove dopamine results
find src/dopamine -name __pycache__ -type d -exec rm -r {} +
find src/dopamine -name results -type d -exec rm -r {} +

# remove ray results
find src/ray -name results -type d -exec rm -r {} +

# remove horizon results
find src/horizon -name __pycache__ -type d -exec rm -r {} +
find src/horizon -name results -type d -exec rm -r {} +
