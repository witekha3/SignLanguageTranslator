import os

import numpy as np

import config


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _Actions(metaclass=Singleton):

    def __init__(self):
        self._actions = np.array([action.split(".")[0] for action in os.listdir(config.BODY_POINTS_DIR)])

    @property
    def available_actions(self):
        return self._actions


Actions = _Actions()