import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import scipy.signal

ALPHABET_SIZE = 4
CONV_1_KERNEL = 3
CONV_1_DEPTH = 20
CONV_2_KERNEL = 3
CONV_2_DEPTH = 50
AFFINE_1_WIDTH = 100
AFFINE_2_WIDTH = 50




class QNetwork(object):
    def __init__(self, sequence_length):
        print('Building Q network')
        self.LEARNING_RATE = 0.00001
        self.VALUE_LOSS_FACTOR = 0.5
        self.POLICY_LOSS_FACTOR = 0.5
        self.ENTROPY_LOSS_FACTOR = 0.01
        self.sequence_length = sequence_length
        self._init_inputs()
        # self._init_action_layers()
        self._init_general_state_layer()
        self._init_hidden_layers()
        self._init_output_layer()
        self._init_loss()

    def _init_loss(self):
        self.action_onehot = tf.one_hot(self.action_input, self.sequence_length*ALPHABET_SIZE, dtype=tf.float32)
        self.responsible_outputs = tf.reduce_sum(self.policy * self.action_onehot, [1])

        # self.indexes = tf.range(0, tf.shape(self.policy)[0]) * tf.shape(self.policy)[1] + self.action_input
        # self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        rewards_plus = np.asarray(self.reward.tolist() + [self.BOOTSTRAP_VALUE])
        self.discounted_rewards = discout(rewards_plus, self.DISCOUNT_RATE)[:-1]
        self.values_plus = np.asarray(self.values.tolist() + [self.BOOTSTRAP_VALUE])
        marginal_advantages = self.reward + self.DISCOUNT_RATE * self.values_plus[1:] - self.values_plus[:-1]
        self.advantages = discount(marginal_advantages, self.DISCOUNT_RATE)

        # self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.discounted_rewards)

        with tf.name_scope('error'):
            self.value_loss = tf.reduce_sum(tf.square(self.discounted_rewards - self.value))
            self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
            self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
            self.loss = self.VALUE_LOSS_FACTOR * self.value_loss +
                        self.POLICY_LOSS_FACTOR * self.policy_loss -
                        self.ENTROPY_LOSS_FACTOR * self.entropy
    def _init_trainger(self):
        trainer = tf.train.GradientDescentOptimizer(learning_rate=self.LEARNING_RATE)
        self.updateModel = trainer.minimize(self.loss)

    def _init_summaries(self):
            tf.summary.histogram('total_loss', self.loss)
            tf.summary.histogram('value_loss', self.value_loss)
            tf.summary.histogram('policy_loss', self.policy_loss)
            tf.summary.histogram('entropy', self.entropy)
            tf.summary.histogram('advantages', self.advantages)
            tf.summary.histogram('discounted_rewards', self.discounted_rewards)
            tf.summary.histogram('policy', self.policy)
            tf.summary.histogram('value', self.value)


    def _init_output_layer(self):
        self.policy = self._gen_affine_layer(self.affine2, , "policy", tf.nn.softmax)
        self.chosen_action = tf.argmax(self.policy, 1)
        self.value = self._gen_affine_layer(self.affine2, 1, "value", None)


    # def _init_outputs(self):
    #     # advantages = reward + self.DISCOUNT_RATE * self.values_plus[1:] - self.values_plus[:-1]
    #     # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    #         with tf.name_scope('regularization'):
    #             self.regularizers = sum([tf.nn.l2_loss(weight) for weight in self.weights])



    def train(self, rewards, states, actions, sess):
        feed_dict = {}
        feed_dict[self.state_input] = states
        feed_dict[self.action_input] = actions
        feed_dict[self.rewards_plus] = rewards
        sess.run(self.update_model,actions=feed_dict)
        return feed_dict

    def get_regularizer(self, sess):
        return sess.run(self.regularizers)

    # get state, predict action by simple per-dimension ascent
    # states - batch_size x state_size
    # actions - batch_size x state_size
    def predict(self, state, sess):
        feed_dict = self.states_to_feed_dict(state)
        policy = sess.run(self.policy, feed_dict=feed_dict)
        action = []
        for i in range(policy.shape[1]):
            action.append(np.random.choice(ACTION_VALUES, 1, p=policy[0,i,:])[0])
            print(policy[0,i,:], action)
            # [aa, qq] = sess.run([self.output_action, self.policy], feed_dict=feed_dict)
        return action

    def get_value(self, states, sess):
        feed_dict = self.states_to_feed_dict(states)
        value = sess.run(self.value, feed_dict=feed_dict)
        self.total_value += value[0]
        return value

    def pop_total_value(self):
        ret = self.total_value
        self.total_value = 0
        return ret


    def _init_inputs(self):
        self.action_input = tf.placeholder(shape=[None], dtype=tf.int32, name="input_action")
        seq_eng_mat_shape = [None, self.sequence_length, self.sequence_length, (ALPHABET_SIZE ** 2) + 1]
        self.state_input = tf.placeholder(shape=seq_eng_mat_shape, dtype=tf.float32, name="input_state")
        self.reward_input = tf.placeholder(shape=[None], dtype=tf.float32, name="input_reward")


    def _init_hidden_layers(self):
        self.conv1 = self._gen_conv2d_layer(self.state_input, CONV_1_DEPTH, CONV_1_KERNEL, "Conv 1")
        self.conv2 = self._gen_conv2d_layer(self.conv1, CONV_2_DEPTH, CONV_2_KERNEL, "Conv 2")
        # conv2_flat_size = np.prod(tf.shape(self.conv2)[1:])
        # conv2_flat = tf.reshape(self.conv2, [-1, conv2_flat_size])
        conv2_flat = tf.contrib.layers.flatten(self.conv2)
        self.affine1 = self._gen_affine_layer(self.conv2_flat, AFFINE_1_WIDTH, "Affine 1")
        self.affine2 = self._gen_affine_layer(self.affine1, AFFINE_2_WIDTH, "Affine 2")

    def _variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def _gen_affine_layer(self, input_layer, out_size, layer_name, activation=tf.nn.relu):
        layer = tf.layers.dense(features, out_size, activation=activation, name=layer_name)
        layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name)
        self._variable_summaries(layer_vars[0], 'weights')
        self._variable_summaries(layer_vars[1], 'bias')
        self._variable_summaries(layer, 'act')
        return layer

    def _gen_conv2d_layer(self, input_layer, depth, kernel_size, layer_name):
        # TODO: diagonal mask? this will reduce parameters might speed up the learning
        # TODO: should check is kernels are usually diagonally dominant
        layer = tf.layers.conv2d(inputs=input_layer,
                                filters=depth,
                                kernel_size=[kernel_size, kernel_size],
                                padding="same",
                                activation=tf.nn.relu,
                                name=layer_name)
        layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name)
        self._variable_summaries(layer_vars[0], 'kernels')
        self._variable_summaries(layer_vars[1], 'bias')
        self._variable_summaries(layer, 'act')
        return layer


def discount(x, self.DISCOUNT_RATE):
    return scipy.signal.lfilter([1], [1, -self.DISCOUNT_RATE], x[::-1], axis=0)[::-1]