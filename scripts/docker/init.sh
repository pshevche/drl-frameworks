echo "--- STARTING DOCKER SETUP ---"
docker network create drl-net
docker build -t pg park/query-optimizer/docker/
docker build -t drl-frameworks:base .
echo "--- DOCKER SETUP COMPLETED ---"