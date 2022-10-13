def conv2D(filter_num, kernel_size, strides = (1, 1), padding='valid', use_activation = True):
    def block(x):
        x = tf.keras.layers.Conv2D(filter_num, kernel_size, strides=strides, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if use_activation:
            x = tf.keras.layers.Activation('relu')(x)
        return x
    return block


def identity_block(filters, kernel_size, cardinality = 32):
    def block(x_shortcut):
        sum_of_pathes = []
        for i in range(cardinality):
            _x = conv2D(filters[0]//cardinality, (1, 1))(x_shortcut)
            _x = conv2D(filters[1]//cardinality, kernel_size, padding='same')(_x)
            _x = conv2D(filters[2], (1, 1), use_activation = False)(_x)
            sum_of_pathes.append(_x)

        x = tf.zeros_like(sum_of_pathes[0])
        for i in range(cardinality):
            x = tf.keras.layers.Add()([x, sum_of_pathes[i]])

        x = tf.keras.layers.Add()([x, x_shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    return block


def convolutional_block(filters, kernel_size, strides = (2, 2), cardinality = 32):
    def block(x_shortcut):
        sum_of_pathes = []
        for i in range(cardinality):
            _x = conv2D(filters[0]//cardinality, (1, 1), strides=strides)(x_shortcut)
            _x = conv2D(filters[1]//cardinality, kernel_size, padding='same')(_x)
            _x = conv2D(filters[2], (1, 1), use_activation = False)(_x)
            sum_of_pathes.append(_x)

        x = tf.zeros_like(sum_of_pathes[0])
        for i in range(cardinality):
            x = tf.keras.layers.Add()([x, sum_of_pathes[i]])

        # Not sure if shortcut need to be splited
        x_shortcut = conv2D(filters[2], (1, 1), strides=strides, use_activation=False)(x_shortcut)

        x = tf.keras.layers.Add()([x, x_shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    return block


def ResNextV2(input_shape, output_dims, number_of_identity_blocks):
    inp = tf.keras.Input(shape = input_shape)

    x = tf.keras.layers.ZeroPadding2D((3, 3))(inp)
    x = conv2D(64, (7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = convolutional_block([64, 64, 256], (3, 3), strides=(1, 1))(x)
    for i in range(number_of_identity_blocks[0]):
        x = identity_block([64, 64, 256], (3, 3))(x)

    x = convolutional_block([128, 128, 512], (3, 3))(x)
    for i in range(number_of_identity_blocks[1]):
        x = identity_block([128, 128, 512], (3, 3))(x)

    x = convolutional_block([256, 256, 1024], (3, 3))(x)
    for i in range(number_of_identity_blocks[2]):
        x = identity_block([256, 256, 1024], (3, 3))(x)

    x = convolutional_block([512, 512, 2048], (3, 3))(x)
    for i in range(number_of_identity_blocks[3]):
        x = identity_block([512, 512, 2048], (3, 3))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_dims)(x)

    model = tf.keras.Model(inp, x)
    return model
