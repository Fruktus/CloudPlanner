import keras

from cloudplanner.usage_prediction.networks.base_network import BaseNetworkModel


class LSTM2Layer(BaseNetworkModel):
    def __init__(self, input_shape):
        self.history = None

        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))

        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=128,
                    return_sequences=True
                )
            )
        )

        model.add(keras.layers.Dropout(rate=0.2))

        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=128
                )
            )
        )

        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.01))

    def fit_model(self, x_train, y_train, epochs=15):
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            shuffle=False,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=6)]
        )
