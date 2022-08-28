from ResNetV2_backbone import ResNetV2

def ResNet152(input_shape, output_dims):
    return ResNetV2(input_shape, output_dims, [2, 7, 35, 2])
