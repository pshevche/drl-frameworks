# remove dopamine cache
find src/dopamine -name __pycache__ -type d -exec rm -r {} +
# remove horizon cache
find src/horizon -name __pycache__ -type d -exec rm -r {} +
# remove results
rm -rf results