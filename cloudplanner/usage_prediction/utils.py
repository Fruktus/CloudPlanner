from math import sqrt
from sys import maxsize

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler


def filter_dataframe(dataframe, filter, metric='cpu.usage.average'):
    filtered_df = dataframe.copy()
    for index, row in filtered_df.iterrows():
        filter.update(row['month'] + row['day_of_month'], row[metric])
        if filter.get_current_state() == filter.states.overutil_anomaly:
            filtered_df.drop(index, inplace=True)

    filtered_df.reset_index(inplace=True)
    filtered_df.drop(columns=['index'], inplace=True)
    return filtered_df


def split_dataframe(dataframe, ratio=0.9):
    train_size = int(len(dataframe) * ratio)
    train, test = dataframe.iloc[0:train_size], dataframe.iloc[train_size:len(dataframe)]
    return train, test


def create_dataset(x, y, time_steps=1):
    xs, ys = [], []
    for i in range(len(x) - time_steps):
        v = x.iloc[i:(i + time_steps)].values
        xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(xs), np.array(ys)


def run_experiment(dataframe, network, adfilter=None, metric='cpu.usage.average', show_plot=False):
    predict_df = pd.DataFrame()
    # predict_df['hour'] = df.timestamp.dt.hour
    predict_df['day_of_month'] = dataframe.timestamp.dt.day
    predict_df['day_of_week'] = dataframe.timestamp.dt.dayofweek
    predict_df['month'] = dataframe.timestamp.dt.month
    predict_df[metric] = dataframe[metric]

    if adfilter:
        predict_df = filter_dataframe(predict_df, adfilter, metric)
    train, test = split_dataframe(predict_df)

    time_steps = 1

    sc = RobustScaler()
    sc = sc.fit(train[[metric]])
    train[metric] = sc.transform(train[[metric]])
    test[metric] = sc.transform(test[[metric]])

    x_train, y_train = create_dataset(train, train[metric], time_steps)
    x_test, y_test = create_dataset(test, list(test[metric]), time_steps)

    network.fit_model(x_train, y_train)

    reshaped_df = np.array(predict_df).reshape((predict_df.shape[0], 1, predict_df.shape[1]))
    y_pred = sc.inverse_transform(network.predict(reshaped_df))

    fig = go.Figure()  # changed from FigureWidget
    fig.add_scatter(x=dataframe['timestamp'][:-2],
                    y=predict_df[metric][:-2],
                    name="Actual resource consumption",
                    mode='lines',
                    line=dict(color='black'))
    fig.add_scatter(x=dataframe['timestamp'][:-2],
                    y=y_pred.flatten(), name="Predicted resource consumption",
                    mode='lines',
                    line=dict(color='black', dash='dot'))

    if show_plot:
        fig.show()
    return {'timestamp': predict_df['timestamp'][:-2], 'true': predict_df[metric][:-2],
            'prediction': y_pred, 'plot': fig}


def run_batch_experiment(dataframes, filters):
    for df in dataframes:
        result = run_experiment(df)
        result['plot'].update_layout(title='filter = None')
        result['plot'].show()
        print('experiment results: ', analyze_experiment(result))

        for adfilter in filters:
            result = run_experiment(df, filter=adfilter)
            result['plot'].update_layout(title='filter = ' + str(adfilter))
            result['plot'].show()
            print('experiment results: ', analyze_experiment(result))


def analyze_experiment(ex_result: dict):
    # calculate extrema: overestimation, underestimation, crosscorrelation, average positive/negative/total difference,
    sample_len = len(ex_result['true'])

    total_difference = 0

    overestimate_difference = 0
    total_overestimates = 0

    underestimate_difference = 0
    total_underestimates = 0

    highest_overestimate = 0
    lowest_underestimate = maxsize

    correlation = float(np.correlate(ex_result['true'], ex_result['prediction'].flatten()))

    for sample in zip(ex_result['prediction'].flatten(), ex_result['true']):
        diff = sample[0] - sample[1]

        total_difference += abs(diff)

        if diff > 0:
            overestimate_difference += diff
            total_overestimates += 1
            highest_overestimate = diff if diff > highest_overestimate else highest_overestimate
        if diff < 0:
            underestimate_difference += diff
            total_underestimates += 1
            lowest_underestimate = diff if diff < lowest_underestimate else lowest_underestimate

    return {'avg_total_diff': total_difference/sample_len,
            'avg_overestimate_diff': overestimate_difference/total_overestimates,
            'avg_underestimate_diff': underestimate_difference/total_underestimates,
            'highest_overestimate': highest_overestimate,
            'lowest_underestimate': lowest_underestimate,
            'correlation': correlation,
            'RMSE': sqrt(mean_squared_error(ex_result['true'], ex_result['prediction'].flatten()))}
