from cloudplanner.anomaly_detection.algorithms.base_algorithm import BaseAlgorithm


class AnomalyAnalyser:
    def __init__(self, algorithms: [BaseAlgorithm]):
        self._algorithms = algorithms

    def analyze_dataframe(self, df, metric):
        for index, row in df.iterrows():
            for alg in self._algorithms:
                alg.update(row['timestamp'], row[metric])

    def analyze_dataframe_batch(self, dfs):
        pass
