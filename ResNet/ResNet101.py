from ResNetV2_backbone import ResNetV2

def ResNet101(input_shape, output_dims):
    return ResNetV2(input_shape, output_dims, [2, 3, 22, 2])
