import numpy as np


class Game(object):
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.ALPHABET_SIZE = 4
        # todo: read target from file? how should it be generated?
        self.eng_target = self._get_energy_rand()

    def reset(self):
        self.sequence = np.random.choice(self.ALPHABET_SIZE, self.sequence_length)
        return self.get_state()

    def get_state(self):
        eng = self.eng_target - self._get_energy_rand()
        cross_onehot = self._gen_cross_onehot()
        state = np.concatenate((eng, cross_onehot), 2)
        mask = np.triu(np.ones(self.sequence_length), 1)
        mask = np.matlib.repmat(mask, 2, self.ALPHABET_SIZE * self.ALPHABET_SIZE + 1)
        state *= mask
        return state

    def _gen_cross_onehot(self):
        a = np.matlib.repmat(self.sequence, self.sequence_length, 1)
        b = np.matlib.repmat(np.transpose(self.sequence), 1, self.sequence_length)
        cross = a * self.ALPHABET_SIZE + b
        return onehot_mat(cross, self.ALPHABET_SIZE * self.ALPHABET_SIZE)

    def _get_energy_rand(self):
        # todo: replace with Yann's code
        return np.random.rand(self.sequence_length, self.sequence_length)

    def step(self, action):
        self.sequence[np.floor(action / self.ALPHABET_SIZE)] = action % self.ALPHABET_SIZE
        return self.get_state()


# out - |A|_1 x |A|_2 x n
def onehot_mat(A, n):
    flat_onehot = onehot(A.flatten(), n)
    out = np.reshape(flat_onehot, list(A.shape) + [n])
    return out


# out - |v| x n
def onehot(v, n):
    out = np.zeros((v.size, n))
    out[np.arange(v.size), v] = 1
    return out
