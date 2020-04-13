import pandas as pd
from sklearn.metrics import r2_score
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.callbacks import ReduceLROnPlateau
from tensorflow_core.python.keras.layers import Conv1D, LeakyReLU, BatchNormalization, GRU, Flatten, Dense, Activation
from tensorflow_core.python.keras.losses import mean_absolute_error, MAE
from tensorflow_core.python.keras.metrics import RootMeanSquaredError
from tensorflow_core.python.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


class LSTM:
    def Train(self, input, target):
        X_train, X_test, Y_train, Y_test = train_test_split(input, target, train_size=0.75)
        Y_train = np.asarray(Y_train)
        Y_test = np.array(Y_test)
        X_train = np.reshape(X_train, [-1, X_train[0].shape[0], X_train[0].shape[1]])
        X_test = np.reshape(X_test, [-1, X_train[0].shape[0], X_train[0].shape[1]])

        model = Sequential()
        model.add(Conv1D(16, 3, padding='same', input_shape=input[0].shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(GRU(16, return_sequences=True))
        # model.add(Activation("sigmoid"))
        # model.add(LSTM(lstm_out))

        model.add(Flatten())
        model.add(Dense(8, activity_regularizer=l2(0.001)))
        # model.add(GRU(lstm_out, return_sequences=True))
        # model.add(LSTM(lstm_out))
        # model.add(Dense(20, activity_regularizer=l2(0.001)))
        model.add(Activation("relu"))
        model.add(Dense(2))

        model.compile(loss=mean_absolute_error, optimizer='nadam',
                      metrics=[RootMeanSquaredError(), MAE])
        print(model.summary())

        batch_size = 12
        epochs = 100
        reduce_lr_acc = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=epochs / 10, verbose=1, min_delta=1e-4, mode='max')
        model.fit(X_train, Y_train,
                  epochs=epochs,
                  batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=[reduce_lr_acc])
        model.save("PositionEstimation.h5", overwrite=True)
        # acc = model.evaluate(X_test,
        #                      Y_test,
        #                      batch_size=batch_size,
        #                      verbose=0)

        predicted = model.predict(X_test, batch_size=batch_size)
        # predicted = out.ravel()

        res = pd.DataFrame({"predicted_x": predicted[:, 0],
                            "predicted_y": predicted[:, 1],
                            "original_x": Y_test[:, 0],
                            "original_y": Y_test[:, 1]})
        res.to_excel("res.xlsx")
        # sns.set(color_codes=True)
        # ax = sns.regplot(x=Y_test2, y=predicted2, color="g")

        # r2 = r2_score(Y_test, predicted)

        # print("R2:{0}".format(r2))
        # plt.plot(Y_train[])
        # plt.plot(predicted)
        # plt.legend(['Target', 'Estimated'], loc='upper left')
        # plt.show()
