from ray.rllib.models import Model
from ray.rllib.models.misc import normc_initializer
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class ParametricDQNModel(Model):
    """
    Custom Ray model that handles varying-length action spaces. The action mask for the current state is passed
    via input_dict["obs"]["action_mask"].
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        action_mask = input_dict["obs"]["action_mask"]

        # Standard FC net component.
        last_layer = input_dict["obs"]["graph"]
        hiddens = [512, 512]
        for i, size in enumerate(hiddens):
            label = "fc{}".format(i)
            last_layer = tf.layers.dense(
                last_layer,
                size,
                # kernel_initializer=normc_initializer(1.0),
                activation=tf.nn.relu,
                name=label)
        output = tf.layers.dense(
            last_layer,
            action_mask.shape[1],
            # kernel_initializer=normc_initializer(0.01),
            activation=None,
            name="fc_out")

        # # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        # intent_vector = tf.expand_dims(output, 1)
        # avail_actions = tf.expand_dims(action_mask, 2)

        # # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        # action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=2)

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        masked_logits = inf_mask + output

        return masked_logits, last_layer


class ParametricRainbowModel(Model):
    """
    Custom Ray model that handles varying-length action spaces. The action mask for the current state is passed
    via input_dict["obs"]["action_mask"].
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        num_atoms = 51
        vmax = 10.
        support = tf.linspace(-vmax, vmax, num_atoms)

        action_mask = input_dict["obs"]["action_mask"]

        # Standard FC net component.
        last_layer = input_dict["obs"]["graph"]
        hiddens = [512, 512]
        for i, size in enumerate(hiddens):
            label = "fc{}".format(i)
            last_layer = tf.layers.dense(
                last_layer,
                size,
                # kernel_initializer=normc_initializer(1.0),
                activation=tf.nn.relu,
                name=label)
        last_layer = tf.layers.dense(
            last_layer,
            num_atoms * action_mask.shape[1],
            # kernel_initializer=normc_initializer(0.01),
            activation=None)
        logits = tf.reshape(
            last_layer, [-1, action_mask.shape[1], num_atoms])
        probabilities = tf.contrib.layers.softmax(logits)
        output = tf.reduce_sum(support * probabilities, axis=2, name='fc_out')

        # # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        # intent_vector = tf.expand_dims(output, 1)
        # avail_actions = tf.expand_dims(action_mask, 2)

        # # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        # action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=2)

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        masked_logits = inf_mask + output

        return masked_logits, output
