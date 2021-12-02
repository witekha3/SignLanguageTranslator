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
        if len(self._actions) == 0:
            actions = input("No actions provided. Type the name of the new actions splitted by semicolon or exist: ")
            if ";" not in actions:
                return [actions]
            actions = [action.strip() for action in actions.split(";")]
            return actions
        input("New data will be append. Press any key to continue...")
        return self._actions


Actions = _Actions()