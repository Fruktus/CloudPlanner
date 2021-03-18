import keras

from cloudplanner.usage_prediction.networks.base_network import BaseNetworkModel


class LSTM2Layer(BaseNetworkModel):
    def __init__(self, input_shape):
        self.history = None

        self.model = keras.Sequential()
        self.model.add(keras.layers.InputLayer(input_shape=input_shape))

        self.model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=128,
                    return_sequences=True
                )
            )
        )

        self.model.add(keras.layers.Dropout(rate=0.2))

        self.model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=128
                )
            )
        )

        self.model.add(keras.layers.Dropout(rate=0.2))
        self.model.add(keras.layers.Dense(units=1))
        self.model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.01))

    def fit_model(self, x_train, y_train, verbose, epochs=15):
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            shuffle=False,
            verbose=verbose,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=6)]
        )
