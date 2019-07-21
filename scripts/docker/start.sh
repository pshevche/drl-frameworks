echo "--- STARTING DOCKER CONTAINERS ---"
docker start docker-pg || docker run -u $(id -u):$(id -g) --name docker-pg -p 0.0.0.0:5432:5432 --net drl-net --privileged -d pg
docker start -i drl-fw || docker run -u $(id -u):$(id -g) --name drl-fw -p 0.0.0.0:6006:6006 --net drl-net --runtime=nvidia -v $PWD:/home/drl-frameworks -it drl-frameworks:base
