import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv1D, Flatten


def deep_model(csi_df):
    # # Min-Max Normalization for test data
    # scaler = MinMaxScaler()
    # scaler.fit(test_df.iloc[:, 0:100])
    # scaled_df = scaler.transform(test_df.iloc[:, 0:100])
    # test_df.iloc[:, 0:100] = scaled_df

    # # Split model, test data with different dataset
    # train_feature = df.drop(columns=['label'])
    # train_target = df['label']
    #
    # test_feature = test_df.drop(columns=['label'])
    # test_target = test_df['label']

    # Split dataset
    train_data, test_data = train_test_split(csi_df, test_size=0.3)

    train_feature = train_data.drop(columns=['label'])
    train_target = tf.keras.utils.to_categorical(train_data['label'], num_classes=2)

    test_feature = test_data.drop(columns=['label'])
    test_target = tf.keras.utils.to_categorical(test_data['label'], num_classes=2)

    print(train_feature.shape)
    print(train_target.shape)

    input_ft = 64  # total subcarrier number
    model = tf.keras.Sequential([Dense(units=192, activation='relu', input_shape=(input_ft,)),
                                Dense(units=96, activation='relu'),
                                Dense(units=48, activation='relu'),
                                Dense(units=24, activation='relu'),
                                Dense(units=12, activation='relu'),
                                Dense(units=2, activation='sigmoid')])

    #-----------------------
    # Conv1D model
    # ---------------------

    # model = tf.keras.Sequential()
    # model.add(Conv1D(4, 3, activation='relu', input_shape=(input_ft,)))  # 4 kernel, size of kernel 3
    # model.add(Flatten())  # 2D -> 1D
    # model.add(Dense(unit=10, activation='relu'))
    # model.add(Dense(unit=10, activation='relu'))
    # model.add(Dense(unit=2, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_feature, train_target, epochs=50, batch_size=25, validation_split=0.25, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])


    print("\n Training is done! \n")

    print("Evaluation:")
    model.evaluate(test_feature, test_target)

    # print("Save model")
    # model.save('trained_dl_model')
    #
    # # Convert to TF Lite model
    # converter = tf.lite.TFLiteConverter.from_saved_model('trained_dl_model')
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)
