import tensorflow as tf

def conv2D(filter_num, kernel_size, strides = (1, 1), padding='valid', use_activation = True):
    def block(x):
        x = tf.keras.layers.Conv2D(filter_num, kernel_size, strides=strides, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if use_activation:
            x = tf.keras.layers.Activation('relu')(x)
        return x
    return block


def identity_block(filters, kernel_size):
    def block(x_shortcut):
        x = conv2D(filters[0], kernel_size, padding='same')(x_shortcut)
        x = conv2D(filters[1], kernel_size, padding='same', use_activation = False)(x)

        x = tf.keras.layers.Add()([x, x_shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    return block


def convolutional_block(filters, kernel_size, strides = (2, 2)):
    def block(x_shortcut):
        x = conv2D(filters[0], kernel_size, padding='same', strides = strides)(x_shortcut)
        x = conv2D(filters[1], kernel_size, padding='same', use_activation = False)(x)

        x_shortcut = conv2D(filters[1], (1, 1), strides=strides, use_activation=False)(x_shortcut)

        x = tf.keras.layers.Add()([x, x_shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    return block


def ResNetV1(input_shape, output_dims, number_of_identity_blocks):
    inp = tf.keras.Input(shape = input_shape)

    x = tf.keras.layers.ZeroPadding2D((3, 3))(inp)
    x = conv2D(64, (7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = convolutional_block([64, 64], (3, 3), strides=(1, 1))(x)
    for i in range(number_of_identity_blocks[0]):
        x = identity_block([64, 64], (3, 3))(x)

    x = convolutional_block([128, 128], (3, 3))(x)
    for i in range(number_of_identity_blocks[1]):
        x = identity_block([128, 128,], (3, 3))(x)

    x = convolutional_block([256, 256], (3, 3))(x)
    for i in range(number_of_identity_blocks[2]):
        x = identity_block([256, 256], (3, 3))(x)

    x = convolutional_block([512, 512], (3, 3))(x)
    for i in range(number_of_identity_blocks[3]):
        x = identity_block([512, 512], (3, 3))(x)

    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_dims)(x)

    model = tf.keras.Model(inp, x)
    return model
