from src.model.alexnet import AlexNet
from src.model.convmixer import convmixer, convmixer_detector, convmixer_detector_feature
from src.model.convnet import ConvNet
from src.model.dnn import DNN
from src.model.resnet import ResNet, resnet_detector, resnet_detector_feature
from src.model.vgg import vgg11_bn, vgg_detector, vgg_detector_feature
from src.model.vgg_encrypted import init_vgg


def model_choice(model: str, input_size, num_classes, num_classes_watermarking, out_layer=0, feature=False):
    if model == "ConvNet":
        return ConvNet(False, out_layer), ConvNet(True, out_layer)
    elif model == "DNN":
        return DNN(input_size, num_classes, False)
    elif model == "VGG":
        if feature:
            detector = vgg_detector_feature(num_classes_watermarking)
        else:
            detector = vgg_detector(num_classes_watermarking)
        return vgg11_bn(linear=False), vgg11_bn(linear=True), detector
    elif model == "VGG_encrypted":
        return init_vgg(num_classes), init_vgg(num_classes)
    elif model == "ConvMixer":
        if feature:
            detector= convmixer_detector_feature(num_classes_watermarking)
        else:
            detector = convmixer_detector(num_classes_watermarking)
        return convmixer(linear=False, num_classes=num_classes), convmixer(linear=True, num_classes=num_classes), detector
    elif model == "ResNet":
        if feature:
            detector= resnet_detector_feature(num_classes_watermarking)
        else:
            detector = resnet_detector(num_classes_watermarking)
        return ResNet(False, num_classes), ResNet(True, num_classes), detector
    elif model == "AlexNet":
        return AlexNet(False, num_classes), AlexNet(True, num_classes)
    else:
        raise ValueError("Model not found")
