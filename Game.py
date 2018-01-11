import numpy as np
import random

class Game(object):
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.ALPHABET_SIZE = 4
        self.eng_target = self._get_energy_rand()

    def reset(self):
        self.sequence = np.random.choice(ALPHABET_SIZE, self.sequence_length)

    def get_state(self):
        eng = self.eng_target - self._get_energy_rand()
        a = np.matlib.repmat(self.sequence, self.sequence_length, 1)
        b = np.matlib.repmat(np.transpose(self.sequence), 1, self.sequence_length)
        cross = a * self.ALPHABET_SIZE + b
        cross_onehot = onehot_mat(cross, n)
        np.concatenate((eng,cross_onehot), 2)

    def _get_energy_rand(self):
        # todo: replace with Yann's code
        return np.random.rand(self.sequence_length,self.sequence_length)


# out - |A|_1 x |A|_2 x n
def onehot_mat(A, n):
    flat_onehot = onehot(A.flatten(), n);
    flat_onehot_per = np.transpose( np.expand_dims(flat_onehot, axis=2), (0, 1, 2))
    out = np.reshape(flat_onehot, list(A.shape)+[n])
    return out

# out - |v| x n
def onehot(v, n):
    out = np.zeros((v.size, n))
    out[np.arange(v.size), v] = 1
    return out
