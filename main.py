from actions_collector.collect_uta_data import collect_actions
from body_detecotr import BodyDetector

if __name__ == '__main__':

    # collect_actions()
    x = BodyDetector.get_points().sort_values(by=['action'])
    a=2