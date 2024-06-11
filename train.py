import tensorflow as tf

from preprocess import generate_training_sequences, SEQUENCE_LENGTH, \
    SINGLE_FILE_DATASET, \
    MAPPING_PATH

OUTPUT_UNITS = 38
LOSS = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001
NUM_UNITS = [256]
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model(output_units, num_units, loss, learning_rate):
    pass
    # create the model architecture
    input_layer = tf.keras.Input(shape=(None, output_units))
    x = tf.keras.layers.LSTM(num_units[0])(input_layer)
    x = tf.keras.layers.Dropout(0.2)(x)

    output_layer = tf.keras.layers.Dense(output_units, activation='softmax')(x)

    model = tf.keras.Model(input_layer, output_layer)

    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    model.summary()

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE,
          save_path=SAVE_MODEL_PATH):

    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH, SINGLE_FILE_DATASET, MAPPING_PATH)
    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)
    # train the model

    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(save_path)


if __name__ == "__main__":
    train()
