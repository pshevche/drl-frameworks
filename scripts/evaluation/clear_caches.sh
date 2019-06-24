# remove dopamine cache
find drl_fw/dopamine -name __pycache__ -type d -exec rm -r {} +
# remove horizon cache
find drl_fw/horizon -name __pycache__ -type d -exec rm -r {} +
# remove ray cache
find drl_fw/ray -name __pycache__ -type d -exec rm -r {} +