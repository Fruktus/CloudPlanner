from enum import Enum
from statistics import median, stdev


class StateAnalyzer:
    def __init__(self, max_values=100, tolerance=0.2, tolerance_multipler=2):
        self.states = Enum('states', 'learning normal anomaly')

        self._normal_state = 0
        self._current_state = self.states.learning
        self._max_values = max_values
        self._values = []
        self._tolerance = tolerance
        self._tolerance_multiplier = tolerance_multipler

    def get_normal_state(self):
        return self._normal_state

    def get_stdev_tolerance(self):
        if len(self._values) < 2:
            return 0
        return self._tolerance_multiplier * stdev(self._values)

    def update(self, value):
        # TODO for now does not include weights for older values
        # idea for calculating the weights:
        # keep an array of them, sorted in appearing order (most to least recent)
        # Vi - ith value, i - index in table (lower index value - more recent)
        # Vi * 0.95^i

        if self._current_state == self.states.learning:
            self._values.insert(0, value)

            # if not enough values to continue with calculations stop early
            if len(self._values) <= self._max_values:
                return self._current_state

        # add new value and remove old to keep only max_values stored
        self._values.insert(0, value)
        self._values.pop()

        # recalculate normal state
        self._normal_state = median(self._values)

        # recalculate current state
        # TODO add tolerances
        self._current_state = self.states.normal if value < self._normal_state + self._normal_state*self._tolerance\
            else self.states.anomaly

        # self._current_state = States.normal if value < self._normal_state + \
        # self._tolerance_multiplier*stdev(self._values)\
        #     else States.anomaly

        # return current state
        return self._current_state
