from cloudplanner.anomaly_detection.algorithms.base_algorithm import BaseAlgorithm


class AnomalyDetector:
    def __init__(self, algorithms: [BaseAlgorithm]):
        self._algorithms = algorithms

    def analyze_dataframe(self, df):
        pass

    def analyze_dataframe_batch(self, dfs):
        pass