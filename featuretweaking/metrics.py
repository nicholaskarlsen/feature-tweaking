import numpy as np


class Metric:
    def __init__(self):
        pass

    def filter_nan(a, b):
        return

    def __call__(self, *args, **kwargs):
        pass


class L1(Metric):
    def __init__(self):
        return

    def __call__(self, a, b):
        """

        :param a:
        :param b:
        :return:
        """
        d = np.subtract(b, a)
        return np.linalg.norm(d, ord=1)


class L2(Metric):
    def __init__(self):
        return

    def __call__(self, a, b):
        """

        :param a:
        :param b:
        :return:
        """
        return np.linalg.norm(b - a, ord=2)


class L1Relative(Metric):
    def __init__(self):
        return

    def __call__(self, a, b):
        """

        :param a:
        :param b:
        :return:
        """
        return np.linalg.norm((b - a) / a, ord=1)


class L2Relative(Metric):
    def __init__(self):
        return

    def __call__(self, a, b):
        """

        :param a:
        :param b:
        :return:
        """
        return np.linalg.norm((b - a) / b, ord=2)


class KroneckerDelta(Metric):
    def __init__(self):
        return

    def __call__(self, a, b):
        """Kronecker-Delta function
        :param a:
        :param b:
        :return:
        """
        return np.sum(np.equal(a, b))
