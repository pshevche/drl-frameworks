# A Comparative Evaluation of Deep Reinforcement Learning Frameworks
This repository contains the implementation part of our work in which we compared several deep reinforcement learning frameworks: [Dopamine][dopamine], [Horizon][horizon] and [Ray][ray].

## How to use

### Using Docker (recommended)
1. Install Docker following the instructions from [here][docker].
2. Install `nvidia-docker` from [here][nvidia-docker] to run experiments with GPU inside the container.
3. Build necessary Docker images by executing the following script inside the repository folder:
```
bash ./scripts/docker/init.sh
```
4. Once the images are built, start Postgres and project containers by running `bash ./scripts/docker/start.sh`. When the script is done, you'll have access to the interactive terminal inside the project container.
5. Once inside the container, you can validate the setup by running `pytest tests/`.
6. Alternatively, procedd with the experiments by executing `bash ./scripts/evaluation/{environment}/eval_all.sh` for the complete evaluation or `bash ./scripts/evaluation/{environment}/eval_{framework}.sh` inside the container for evaluating single frameworks.
7. Run `tensorboard --logdir=results` inside the container, and view the Tensorboard summary of evaluation results in your local browser at `localhost:6006`.

### Install from source (development)
1. Install Anaconda from [here][miniconda] (make sure to download the Python 3 version). Leave Anaconda's installation directory default (home/miniconda3).
2. Check Anaconda's version by executing `conda -V`. If Anaconda's version is `<=4.6.8`, run `conda update conda`.
3. Create project environment by running `bash ./scripts/setup_env.sh`.
4. Build the project by running `bash ./scripts/build_project.sh`.
5. You can validate your setup by running `pytest tests/`.
6. If you'd like to evaluate the frameworks against the [Park][park]'s Query Optimizer environment, then do the following extra steps:
    - edit `/etc/hosts` file and add the following alias: `127.0.0.1   docker-pg`;
    - build the Postgres Docker image used by the environment: `docker build -t pg park/query-optimizer/docker/`;
    - start the Postgres container: `docker start docker-pg || docker run --name docker-pg -p 0.0.0.0:5432:5432 --net drl-net --privileged -d pg`.
5. Run experiments for a specific environment by executing `./scripts/evaluation/{environment}/eval_all.sh`. Alternatively, you can run experiments for individual frameworks by running `./scripts/evaluation/{environment}/eval_{framework}.sh`.
6. View evaluation results in Tensorboard (`localhost:6006`) after running `tensorboard --logdir=results`. You may need to activate the proper environment first (`conda activate drl-frameworks-env`).


[dopamine]: https://github.com/google/dopamine
[horizon]: https://github.com/facebookresearch/Horizon
[ray]: https://github.com/ray-project/ray
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[docker]: https://docs.docker.com/install/
[nvidia-docker]: https://github.com/NVIDIA/nvidia-docker
[park]: https://github.com/park-project/park