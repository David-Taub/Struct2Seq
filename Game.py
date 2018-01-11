import numpy as np


class Game(object):
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.ALPHABET_SIZE = 4
        # todo: read target from file? how should it be generated?
        self.eng_target = self._get_energy()
        self.DONE_THRESHOLD = 0.01 / (self.sequence_length * self.sequence_length)

    def reset(self):
        self.sequence = np.random.choice(self.ALPHABET_SIZE, self.sequence_length)
        return self.get_state()

    def get_state(self):
        cross_onehot = self._gen_cross_onehot()
        state = np.concatenate((self._get_energy_diff(), cross_onehot), 2)
        mask = np.triu(np.ones(self.sequence_length), 1)
        mask = np.matlib.repmat(mask, 2, self.ALPHABET_SIZE * self.ALPHABET_SIZE + 1)
        state *= mask
        return state

    def _gen_cross_onehot(self):
        a = np.matlib.repmat(self.sequence, self.sequence_length, 1)
        b = np.matlib.repmat(np.transpose(self.sequence), 1, self.sequence_length)
        cross = a * self.ALPHABET_SIZE + b
        return onehot_mat(cross, self.ALPHABET_SIZE * self.ALPHABET_SIZE)

    def _get_energy(self):
        # todo: replace with Yann's code
        return np.random.rand(self.sequence_length, self.sequence_length)

    def _get_energy_diff(self):
        return self.eng_target - self._get_energy()

    def step(self, action):
        pre_eng_diff = self._get_energy_diff()
        self.sequence[np.floor(action / self.ALPHABET_SIZE)] = action % self.ALPHABET_SIZE
        post_eng_diff = self._get_energy_diff()
        reward = np.norm(pre_eng_diff) - np.norm(post_eng_diff)
        next_state = self.get_state()
        is_finished = np.norm(post_eng_diff) < self.DONE_THRESHOLD
        return next_state, reward, is_finished


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
