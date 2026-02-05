import numpy as np


class Generator:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def random_normal(self, shape, mean=0.0, std=1.0):
        return self.rng.normal(loc=mean, scale=std, size=shape)

    def random_uniform(self, shape, low=0.0, high=1.0):
        return self.rng.uniform(low=low, high=high, size=shape)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.rng = np.random.default_rng(value)


if __name__ == "__main__":
    gen = Generator(seed=42)

    normal_samples = gen.random_normal(shape=(3, 3), mean=0.0, std=1.0)
    uniform_samples = gen.random_uniform(shape=(2, 5), low=0.0, high=10.0)

    print("Normal Samples:\n", normal_samples)
    print("Uniform Samples:\n", uniform_samples)

    normal_samples = gen.random_normal(shape=(3, 3), mean=0.0, std=1.0)
    uniform_samples = gen.random_uniform(shape=(2, 5), low=0.0, high=10.0)

    print("Normal Samples:\n", normal_samples)
    print("Uniform Samples:\n", uniform_samples)
