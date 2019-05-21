# drl-frameworks
This repository contains the implementation part of our work in which we compared several deep reinforcement learning frameworks: [Dopamine][dopamine], [Horizon][horizon] and [Ray][ray].

## How to use

### Install from source
1. Install Anaconda from [here][miniconda] (make sure to download the Python 3 version). Leave Anaconda's installation directory default (home/miniconda3).
2. Check Anaconda's version by executing `conda -V`. If Anaconda's version is `<=4.6.8`, run `conda update conda`.
3. Create Anaconda environments by running `./scripts/setup.sh`.
4. Run experiments by executing `./scripts/evaluate.sh`. Alternatively, you can run experiments for individual frameworks by running `./scripts/evaluate_{framework}.sh`.
5. View evaluation results in Tensorboard (`localhost:6006`) after running `tensorboard --logdir=results`. You may need to activate one of the framework environments first (e.g. `conda activate dopamine-env`).

### Using Docker
1. Install Docker following the instructions from [here][docker].
2. Install `nvidia-docker` from [here][nvidia-docker] to run experiments with GPU inside the container.
3. Build the Docker image by running the following command inside the repository folder:
```
docker build -t drl-frameworks:dev .
```
4. Once the image is built, start an interactive shell in the container. To run with GPU, include `--runtime=nvidia` option. To have the ability to edit files locally and have changes available in the container, mount your local repository as a volume by setting the `-v` flag. Map ports by setting `-p` flag, which will allow you to view Tensorboard visualizations locally.
```
docker run --runtime=nvidia -v $PWD:/home/drl-frameworks -p 0.0.0.0:6006:6006 -it drl-frameworks:dev 
```
5. Once inside the container, run experiments by executing `./scripts/evaluate.sh` for the complete evaluation or `./scripts/evaluate_{framework}.sh` for evaluating single frameworks.
6. Run `tensorboard --logdir=results` inside the container, and view the Tensorboard summary of evaluation results in your local browser at `localhost:6006`. You may need to activate one of the framework environments first:
```
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dopamine-env
```

[dopamine]: https://github.com/google/dopamine
[horizon]: https://github.com/facebookresearch/Horizon
[ray]: https://github.com/ray-project/ray
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[docker]: https://docs.docker.com/install/
[nvidia-docker]: https://github.com/NVIDIA/nvidia-docker