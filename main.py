import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf
from experience_buffer import experience_buffer
from dqn import Qnetwork
import os
from skiing import skiing

batch_size = 32  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
START_EXPLORE_P = 1  # Starting chance of random action
END_EXPLORE_P = 0.1  # Final chance of random action
ANNELING_STEPS = 10000.  # How many steps of training to reduce START_EXPLORE_P to END_EXPLORE_P.
num_episodes = 10000  # How many episodes of game environment to train network with.
pre_train_steps = 10000  # How many steps of random actions before training begins.
max_epLength = 5000  # The max allowed length of our episode.
load_model = True  # Whether to load a saved model.
test_model = True  # Exit after "done" flag is True
path = "./dqn"  # The path to save our model to.
sequence_length = 100
tau = 0.001  # Rate to update target network toward primary network


class Trainer(object):

    def __init__(self):
        self.game = Game.Game()
        tf.reset_default_graph()
        self.mainQN = Qnetwork(sequence_length)
        self.targetQN = Qnetwork(sequence_length)
        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        self.trainables = tf.trainable_variables()
        self.targetOps = self.update_traget_graph()
        self.exp_buffer = ExperienceBuffer()

        # Set the rate of random action decrease.
        self.explore_probability = START_EXPLORE_P
        self.step_drop_explore_p = (START_EXPLORE_P - END_EXPLORE_P) / ANNELING_STEPS

    # create lists to contain total rewards and steps per episode
        self.jList = []
        self.rList = []
        self.total_steps = 0

    def _load_model(self):
        if not os.path.exists(path):
            os.makedirs(path)
        if self.load_model is True:
            ckpt = tf.train.get_checkpoint_state(path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def update_traget_graph(self):
        total_vars = len(self.trainables)
        op_holder = []
        for idx, var in enumerate(self.trainables[0:total_vars / 2]):
            op_holder.append(self.trainables[idx + total_vars / 2].assign(
                (var.value() * tau) + ((1 - tau) * self.trainables[idx + total_vars / 2].value())))
        return op_holder

    def update_traget(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)

    def train(self):
        # Make a path for our model to be saved in.
        with tf.Session() as sess:
            self._load_model()
            sess.run(init)
            update_traget(targetOps, sess)  # Set the target network to be equal to the primary network.
            for i in range(num_episodes):
                episodeBuffer = experience_buffer()
                # Reset environment and get first new observation
                state = self.game.reset()
                is_finished = False
                rAll = 0
                j = 0
                # The Q-Network
                while j < max_epLength:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                    j += 1
                    # Choose an action by greedily (with explore_probability chance of random action) from the Q-network
                    if np.random.rand(1) < explore_probability or total_steps < pre_train_steps:
                        action = np.random.randint(0, 3)
                    else:
                        action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [state]})[0]
                    next_state, reward, is_finished = game.step(action)
                    game.env.rEND_EXPLORE_Pr()
                    total_steps += 1
                    episodeBuffer.add(
                        np.reshape(np.array([state, action, reward, next_state, is_finished]), [1, 5]))  # Save the experience to our episode buffer.

                    if total_steps > pre_train_steps:
                        if explore_probability > END_EXPLORE_P:
                            explore_probability -= step_drop_explore_p

                        if total_steps % (update_freq) == 0:
                            trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                            # Below we perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                            Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                            end_multiplier = -(trainBatch[:, 4] - 1)
                            doubleQ = Q2[range(batch_size), Q1]
                            targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                            # Update the network with our target values.
                            sess.run(mainQN.updateModel, \
                                         feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                                    mainQN.actions: trainBatch[:, 1]})

                            update_traget(targetOps, sess)  # Set the target network to be equal to the primary network.
                    rAll += reward
                    state = next_state

                    if d is True:
                        if test_model:
                            game.env.monitor.close()
                        break


                # Get all experiences from this episode and discount their rewards.
                myBuffer.add(episodeBuffer.buffer)
                jList.append(j)
                rList.append(rAll)
                # Periodically save the model.
                if i % 1000 == 0:
                    saver.save(sess, path + '/model-' + str(i) + '.cptk')
                    print "Saved Model"
                if len(rList) % 10 == 0:
                    print total_steps, np.mean(rList[-10:]), explore_probability
            saver.save(sess, path + '/model-' + str(i) + '.cptk')
        print "Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%"

def main():
