# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The entry point for running a Dopamine agent.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags

from dopamine.discrete_domains import run_experiment
import checkpoint_runner

import tensorflow as tf

import os
import time


flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')

FLAGS = flags.FLAGS


def main(unused_argv):
    """Main method.

    Args:
      unused_argv: Arguments (unused).
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = checkpoint_runner.create_runner(FLAGS.base_dir)
    start_time = time.time()
    runner.run_experiment()
    end_time = time.time()

    # save runtime to file
    ginfile = str(FLAGS.gin_files[0])
    experiment_name = ginfile[ginfile.rfind('/') + 1: ginfile.rfind('.gin')]
    filename = 'runtime_' + experiment_name + '.txt'
    runtime_path = os.path.join(FLAGS.base_dir, filename)

    f = open(runtime_path, 'w+')
    f.write(experiment_name + ' took ' +
            str(end_time - start_time) + ' seconds.')
    f.close()

    if runner.propagation_steps:
        print("--- STARTING DOPAMINE CARTPOLE PROPAGATION EXPERIMENT ---\n")
        start_time = time.time()
        runner.run_propagation_test()
        end_time = time.time()
        filename = 'propagation_runtime_' + experiment_name + '.txt'
        runtime_path = os.path.join(FLAGS.base_dir, filename)
        f = open(runtime_path, 'w+')
        f.write(experiment_name + ' took ' +
                str(end_time - start_time) + ' seconds.')
        f.close()
        print("--- DOPAMINE CARTPOLE PROPAGATION EXPERIMENT COMPLETED ---\n")


if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)
