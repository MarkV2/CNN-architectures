from ResNetV1_backbone import ResNetV1

def ResNet18(input_shape, output_dims):
    return ResNetV1(input_shape, output_dims, [1, 1, 1, 1])
