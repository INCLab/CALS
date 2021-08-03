import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def deep_model(csi_df):
    # Drop timestamp
    csi_df.drop([csi_df.columns[0]], axis=1, inplace=True)

    # Min-Max Normalization
    scaler = MinMaxScaler()
    scaler.fit(csi_df.iloc[:, 0:-1])
    scaled_df = scaler.transform(csi_df.iloc[:, 0:-1])
    csi_df.iloc[:, 0:-1] = scaled_df

    # # Min-Max Normalization for test data
    # scaler = MinMaxScaler()
    # scaler.fit(test_df.iloc[:, 0:100])
    # scaled_df = scaler.transform(test_df.iloc[:, 0:100])
    # test_df.iloc[:, 0:100] = scaled_df

    # # Split train, test data with different dataset
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
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=192, activation='relu', input_shape=(input_ft,)),
                                tf.keras.layers.Dense(units=96, activation='relu'),
                                tf.keras.layers.Dense(units=48, activation='relu'),
                                tf.keras.layers.Dense(units=24, activation='relu'),
                                tf.keras.layers.Dense(units=12, activation='relu'),
                                tf.keras.layers.Dense(units=2, activation='sigmoid')])

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
