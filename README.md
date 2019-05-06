# drl-frameworks

## Getting started
1. Install Anaconda from [here][miniconda] (make sure to download the Python 3 version). Leave Anaconda's installation directory default.
2. Check Anaconda's version by executing `conda -V`. If Anaconda's version is `<=4.6.8`, run `conda update conda`.
3. Create Anaconda environments by running `./scripts/setup.sh`.
4. Run experiments by executing `./scripts/evaluate.sh`.
5. View evaluation results in Tensorboard (`localhost:6006`) after running `tensorboard --logdir=results`.

[miniconda]: https://docs.conda.io/en/latest/miniconda.html