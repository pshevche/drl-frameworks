import tensorflow as tf


class Tensorboard:
    """
    Custom Tensorboard to post experiment summaries in a same format for all frameworks.
    """

    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_summary(self, average_reward_train, num_episodes_train, average_reward_eval, num_episodes_eval, iteration):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Train/NumEpisodes',
                             simple_value=num_episodes_train),
            tf.Summary.Value(tag='Train/AverageReturns',
                             simple_value=average_reward_train),
            tf.Summary.Value(tag='Eval/NumEpisodes',
                             simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Eval/AverageReturns',
                             simple_value=average_reward_eval)
        ])
        self.writer.add_summary(summary, iteration)
        self.writer.flush()
