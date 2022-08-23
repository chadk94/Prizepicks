import tensorflow as tf


def PrizeNet(input_shape):
    input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(name="h1",
                              units=input_shape[0] // 2,
                              activation='relu')(input)
    x = tf.keras.layers.Dropout(name="drop1",
                                rate=0.2)(x)
    x = tf.keras.layers.Dense(name="h2",
                              units=input_shape[0] // 4,
                              activation='relu')(x)
    x = tf.keras.layers.Dropout(name="drop2",
                                rate=0.2)(x)
    outputs = tf.keras.layers.Dense(name="output",
                                    units=1,
                                    activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input,
                           outputs=outputs,
                           name='ChadNet')
    model.summary()
    return model
