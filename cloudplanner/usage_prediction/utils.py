from math import sqrt
from sys import maxsize
from copy import deepcopy
from collections import defaultdict
from statistics import mean, stdev
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from cloudplanner.usage_prediction.networks.lstm_2layer import LSTM2Layer

pd.options.mode.chained_assignment = None


def filter_dataframe(dataframe, filter, metric='cpu.usage.average'):
    filtered_df = dataframe.copy()
    for index, row in filtered_df.iterrows():
        filter.update(row['timestamp'], row[metric])
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


def run_experiment(dataframe, network, adfilter=None, metric='cpu.usage.average', show_plot=False,
                   show_tresholds=False, mark_anomalies=False):

    base_df = pd.DataFrame()
    # predict_df['hour'] = df.timestamp.dt.hour
    base_df['day_of_month'] = dataframe.timestamp.dt.day
    base_df['day_of_week'] = dataframe.timestamp.dt.dayofweek
    base_df['month'] = dataframe.timestamp.dt.month
    base_df[metric] = dataframe[metric]

    filtered_df = filter_dataframe(dataframe, adfilter, metric) if adfilter else dataframe

    base_filtered_df = pd.DataFrame()
    # predict_df['hour'] = df.timestamp.dt.hour
    base_filtered_df['day_of_month'] = filtered_df.timestamp.dt.day
    base_filtered_df['day_of_week'] = filtered_df.timestamp.dt.dayofweek
    base_filtered_df['month'] = filtered_df.timestamp.dt.month
    base_filtered_df[metric] = filtered_df[metric]

    sc = RobustScaler()
    sc = sc.fit(base_filtered_df[[metric]])  # TODO fit before or after filter?

    train, test = split_dataframe(base_filtered_df)

    train[metric] = sc.transform(train[[metric]])
    test[metric] = sc.transform(test[[metric]])

    x_train, y_train = create_dataset(train, train[metric], time_steps=1)
    x_test, y_test = create_dataset(test, list(test[metric]), time_steps=1)

    network.fit_model(x_train, y_train, verbose=False)

    reshaped_df = base_df.copy()
    reshaped_df[metric] = sc.transform(reshaped_df[[metric]])
    reshaped_df = np.array(reshaped_df).reshape((reshaped_df.shape[0], 1, reshaped_df.shape[1]))

    y_pred = sc.inverse_transform(network.predict(reshaped_df))

    fig = go.Figure()  # changed from FigureWidget
    fig.add_scatter(x=dataframe['timestamp'],
                    y=dataframe[metric],
                    name="Actual resource consumption",
                    mode='lines',
                    line_width=1,
                    line=dict(color='black'))
    fig.add_scatter(x=dataframe['timestamp'],
                    y=y_pred.flatten(), name="Predicted resource consumption",
                    mode='lines',
                    line=dict(color='black', dash='dot'))

    if adfilter and show_tresholds:
        fig.add_trace(go.Scatter(x=adfilter.get_tresholds()['timestamp'],
                                 y=adfilter.get_tresholds()['upper_treshold'],
                                 mode='lines',
                                 fill=None,
                                 name='Anomaly Detection Treshold',
                                 line_color='black',
                                 line_width=0.5,
                                 showlegend=False,
                                 fillcolor='rgba(0, 0, 0, 0.1)'))
        fig.add_trace(go.Scatter(x=adfilter.get_tresholds()['timestamp'],
                                 y=adfilter.get_tresholds()['lower_treshold'],
                                 mode='lines',
                                 fill='tonexty',
                                 name='Anomaly Detection Treshold',
                                 line_color='black',
                                 line_width=0.5,
                                 fillcolor='rgba(0, 0, 0, 0.1)'))
    if adfilter and mark_anomalies:
        fig.add_scatter(x=adfilter.get_anomaly_overutil()['timestamp'],
                        y=adfilter.get_anomaly_overutil()['value'], name="Anomaly",
                        mode='markers',
                        marker={'color': 'black'},
                        marker_symbol='square-open-dot')

    title = str(adfilter) if adfilter else 'None'
    fig.update_layout(
        title='Filter = ' + title,
        xaxis_title='Date',
        yaxis_title=metric,
        yaxis_range=[0, 100]
    )

    if show_plot:
        fig.show()
    return {'timestamp': dataframe['timestamp'], 'true': dataframe[metric],
            'prediction': y_pred, 'plot': fig}


def run_batch_experiment(dataframes, filters, verbose=False, show_progress=True, **kwargs):
    results = []

    dataframes = tqdm(dataframes) if show_progress else dataframes

    for df in dataframes:
        try:
            result = run_experiment(df, LSTM2Layer(input_shape=(1, 4)), **kwargs)
            if verbose:
                print('experiment results: ', analyze_experiment(result))
            tmp_res = [{'filter': 'None', 'result': analyze_experiment(result)}]

            for adfilter in filters:
                result = run_experiment(df, LSTM2Layer(input_shape=(1, 4)), adfilter=deepcopy(adfilter), **kwargs)
                if verbose:
                    print('experiment results: ', analyze_experiment(result))
                tmp_res.append({'filter': str(adfilter), 'result': analyze_experiment(result)})
            results.append(tmp_res)
        except Exception:
            print(traceback.format_exc())
    return results


def analyze_batch_result(batch_result, full=False):
    filter_results = defaultdict(lambda: defaultdict(list))
    for df_run in batch_result:
        for filter_run in df_run:
            for metric in filter_run['result'].keys():
                filter_results[filter_run['filter']][metric].append(filter_run['result'][metric])
    if not full:
        for adfilter in filter_results.keys():
            for metric in filter_results[adfilter].copy():
                filter_results[adfilter][metric + '_stdev'] = stdev(filter_results[adfilter][metric])
                filter_results[adfilter][metric] = mean(filter_results[adfilter][metric])
    return filter_results


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
            'total_overestimates': total_overestimates,
            'total_underestimates': total_underestimates,
            'correlation': correlation,
            'RMSE': sqrt(mean_squared_error(ex_result['true'], ex_result['prediction'].flatten()))}


def run_prediction_feedback(dataframe, network, adfilter=None, metric='cpu.usage.average', feedback_len=10,
                            show_plot=True):

    filtered_df = filter_dataframe(dataframe, adfilter, metric) if adfilter else dataframe

    base_filtered_df = pd.DataFrame()
    # predict_df['hour'] = df.timestamp.dt.hour
    base_filtered_df['day_of_month'] = filtered_df.timestamp.dt.day
    base_filtered_df['day_of_week'] = filtered_df.timestamp.dt.dayofweek
    base_filtered_df['month'] = filtered_df.timestamp.dt.month
    base_filtered_df[metric] = filtered_df[metric]

    sc = RobustScaler()
    sc = sc.fit(base_filtered_df[[metric]])  # TODO fit before or after filter?

    train, test = split_dataframe(base_filtered_df)

    train[metric] = sc.transform(train[[metric]])
    test[metric] = sc.transform(test[[metric]])

    x_train, y_train = create_dataset(train, train[metric], time_steps=1)
    x_test, y_test = create_dataset(test, list(test[metric]), time_steps=1)

    network.fit_model(x_train, y_train, verbose=False)

    data_pred = np.concatenate((train, test))
    data_pred = data_pred.reshape((data_pred.shape[0], 1, data_pred.shape[1]))

    for index in range(len(data_pred)):
        if index > feedback_len:
            data_pred[index][0][3] = network.predict(np.array([data_pred[index-1]]))
        else:
            network.predict(np.array([data_pred[index]]))  # needed for stateful networks

    y_pred = sc.inverse_transform(data_pred[:, :, 3])
    # data_pred contains unscaled raw predictions, need to scale them back

    fig = go.Figure()  # changed from FigureWidget
    fig.add_scatter(x=dataframe['timestamp'], y=dataframe['cpu.usage.average'],
                    name="Actual resource consumption", mode='lines',
                    line=dict(color='black'))
    fig.add_scatter(x=dataframe['timestamp'], y=y_pred.flatten(),
                    name="Predicted resource consumption", mode='lines',
                    line=dict(color='black', dash='dot'))

    title = str(adfilter) if adfilter else 'None'
    fig.update_layout(
        title='Filter = ' + title,
        xaxis_title='Date',
        yaxis_title=metric,
        yaxis_range=[0, 100]
    )

    if show_plot:
        fig.show()
    return fig
