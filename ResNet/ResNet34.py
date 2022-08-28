from ResNetV1_backbone import ResNetV1

def ResNet34(input_shape, output_dims):
    return ResNetV1(input_shape, output_dims, [2, 3, 5, 2])
