def ResNet50(input_shape, output_dims):
    return ResNetXtV2(input_shape, output_dims, [2, 3, 5, 2])
