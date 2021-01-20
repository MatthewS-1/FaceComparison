from FaceDataPreprocessing import init, std_filter
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.backend import *
from tensorflow.keras.layers.experimental.preprocessing import *
from tensorflow import abs
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np


def create_siamese(input_shape, summary=False):
    def create_encoder():
        _encoder = Sequential()
        _encoder.add(RandomFlip("horizontal"))
        _zoom_range = (-0.2, 0.2)
        _translation_range = (-0.2, 0.2)
        _encoder.add(RandomZoom(height_factor=_zoom_range, width_factor=_zoom_range, fill_mode="constant"))
        _encoder.add(RandomTranslation(height_factor=_translation_range, width_factor=_translation_range, fill_mode="constant"))
        for i in range(1, 6):
            _kernel_width = ((i <= 2) * 3) + 2
            _encoder.add(Conv2D(filters=16 * i, kernel_size=(_kernel_width, _kernel_width), padding='same', activation='elu',
                                name="encoder_conv" + str(i)))
            _encoder.add(MaxPool2D((2, 2), padding='same', name="encoder_pool" + str(i)))
            _encoder.add(Dropout(0.3, name="encoder_dropout" + str(i)))
        _encoder.add(Flatten())
        _encoder.add(Dense(128, activation="sigmoid"))
        _encoder.add(Dense(16, activation="sigmoid"))
        return _encoder

    encoder = create_encoder()

    not_encoded_in_1 = Input(shape=input_shape)
    not_encoded_in_2 = Input(shape=input_shape)
    encoded_in_1 = encoder(not_encoded_in_1)
    encoded_in_2 = encoder(not_encoded_in_2)

    def absolute(a, b):
        return abs(subtract([a, b]))

    absolute_distance = absolute(encoded_in_1, encoded_in_2)
    output = Dense(1, activation="sigmoid")(absolute_distance)

    siamese_network = Model(inputs=[not_encoded_in_1, not_encoded_in_2], outputs=output)

    if summary:
        siamese_network.summary()

    return siamese_network


NUM_EPOCHS = 15
VERSION = 5


def main():
    in_1, in_2, out = init(num_data=50000, filters=[std_filter], params=[[(0.5, None)]])
    model = create_siamese(in_1.shape[1:])
    model.compile(optimizer="RMSprop", loss="binary_crossentropy", metrics=["AUC"])
    set_value(model.optimizer.learning_rate, 0.00035)  # after some experimentation this is the best learning rate

    cp_path = "Siamese_Training/weights{epoch:08d}.h5"
    cp_callback = ModelCheckpoint(filepath=cp_path, save_weights_only=True)

    hist = model.fit([in_1, in_2], out, epochs=NUM_EPOCHS, validation_split=0.15, callbacks=[cp_callback])
    np.save("face_siamese_" + str(VERSION) + ".npy", hist.history)

    model.save("saved_model/face_siamese_" + str(VERSION))


if __name__ == '__main__':
    main()
