# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

ALPHABET_SIZE = 4
CONV_1_KERNEL = 3
CONV_1_DEPTH = 20
CONV_2_KERNEL = 3
CONV_2_DEPTH = 50
AFFINE_1_WIDTH = 100
AFFINE_2_WIDTH = 50

A = 1
ACTION_LAYER_1_FACTOR = 2
ACTION_LAYER_2_FACTOR = 1
MERGED_LAYER_FATOR_1 = 1.1
MERGED_LAYER_FATOR_2 = 1.1


class QNetwork(object):
    def __init__(self, sequence_length):
        print('Building Q network')
        self.total_value = 0
        self.LEARNING_RATE = 0.00001
        self.BETA_W = 10 ** -3
        self.weights = []
        self.sequence_length = sequence_length
        # ? x SIG x L
        self.seq1hot_shape = [None, ALPHABET_SIZE, self.sequence_length]
        # ? x L x L
        self.seq_eng_mat_shape = [None, self.sequence_length, self.sequence_length, (ALPHABET_SIZE ** 2) + 1]
        self._init_inputs()
        # self._init_action_layers()
        self._init_general_state_layer()
        self._init_hidden_layers()
        self._init_outputs()

    def _init_outputs(self):
        # advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.target_value = tf.placeholder(shape=[None], dtype=tf.float32, name='target_value')
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name='advantages')
        with tf.name_scope('error'):
            value_loss = tf.reduce_sum(tf.square(self.target_value - self.value))
            policy_loss = -tf.reduce_sum(tf.log(self.q_est)*self.advantages)
            entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
            print(entropy.shape)
            input("4")
            with tf.name_scope('regularization'):
                self.regularizers = sum([tf.nn.l2_loss(weight) for weight in self.weights])

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(value_loss + policy_loss + self.BETA_W * self.regularizers + entropy)
            tf.summary.scalar('total_loss', self.loss)
            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('policy_loss', policy_loss)
            tf.summary.scalar('regularizer', self.regularizers)
            tf.summary.histogram('advantages', self.advantages)
            tf.summary.histogram('target_value', self.target_value)

        trainer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        self.update_model = trainer.minimize(self.loss)

    def train(self, discounted_rewards, states, actions, advantages, sess):
        feed_dict = self.states_to_feed_dict(states)
        feed_dict[self.target_value] = discounted_rewards
        feed_dict[self.action_input] = actions
        feed_dict[self.advantages] = advantages
        sess.run(self.update_model, feed_dict=feed_dict)
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
        with tf.name_scope('input_action'):
            self.action_input = tf.placeholder(shape=self.seq1hot_shape, dtype=tf.int8, name="action_input")
        with tf.name_scope('input_state'):
            self.state_input = tf.placeholder(shape=self.seq_eng_mat_shape, dtype=tf.float32, name="input_state")

    def _init_hidden_layers(self):
        self.conv1 = self._gen_conv2d_layer(self.state_input, CONV_1_DEPTH, CONV_1_KERNEL, "Conv 1")
        self.conv2 = self._gen_conv2d_layer(self.conv1, CONV_2_DEPTH, CONV_2_KERNEL, "Conv 2")
        # conv2_flat_size = np.prod(tf.shape(self.conv2)[1:])
        # conv2_flat = tf.reshape(self.conv2, [-1, conv2_flat_size])
        conv2_flat = tf.contrib.layers.flatten(self.conv2)
        self.affine1 = self._gen_affine_layer(self.conv2_flat, AFFINE_1_WIDTH, "Affine 1")
        self.affine2 = self._gen_affine_layer(self.affine1, AFFINE_2_WIDTH, "Affine 2")

    def _init_out_layer(self):
        policy_flat = self._gen_affine_layer(self.affine2, self.sequence_length*ALPHABET_SIZE, "policy_flat", false)
        ###############
        policy_raw = tf.reshape(policy_flat, [-1, self.action_size, len(ACTION_VALUES)])
        self.policy = tf.nn.softmax(policy_raw, dim=2, name='policy_softmax')
        shifted_actions = self.action_input + tf.constant(1, dtype=tf.int8)
        shifted_actions = tf.cast(shifted_actions, tf.uint8)
        onehot_action_input = tf.one_hot(indices=shifted_actions, depth=len(ACTION_VALUES), on_value=1.0, off_value=0.0, axis=2, dtype=tf.float32,)
        self.q_est = tf.reduce_sum(self.policy * onehot_action_input, axis=[1,2], name='q_est')

        # tf.summary.image('q_est', self.q_est)
        # tf.summary.image('onehot_action_input', onehot_action_input)
        # policy = tf.argmax(self.policy, axis=2, name='policy_max')
        # self.output_action = policy - tf.constant(1, dtype=tf.int64)
        self.value = self._gen_affine_layer(mixed2, in_size, 1, 'value', use_relu=False    )


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

    def _gen_affine_layer(self, input_layer, out_size, layer_name, use_relu=True):
        activation = tf.nn.relu if use_relu else None
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
