from src.model.alexnet import AlexNet
from src.model.convmixer import convmixer, convmixer_detector
from src.model.convnet import ConvNet
from src.model.dnn import DNN
from src.model.resnet import ResNet, resnet_detector
from src.model.vgg import vgg11_bn, vgg_detector
from src.model.vgg_encrypted import init_vgg


def model_choice(model: str, input_size, num_classes, num_classes_watermarking, out_layer=0):
    if model == "ConvNet":
        return ConvNet(False, out_layer), ConvNet(True, out_layer)
    elif model == "DNN":
        return DNN(input_size, num_classes, False)
    elif model == "VGG":
        return vgg11_bn(linear=False), vgg11_bn(linear=True), vgg_detector(num_classes_watermarking)
    elif model == "VGG_encrypted":
        return init_vgg(num_classes), init_vgg(num_classes)
    elif model == "ConvMixer":
        return convmixer(linear=False), convmixer(linear=True), convmixer_detector(num_classes_watermarking)
    elif model == "ResNet":
        return ResNet(False, num_classes), ResNet(True, num_classes), resnet_detector(num_classes_watermarking)
    elif model == "AlexNet":
        return AlexNet(False, num_classes), AlexNet(True, num_classes)
    else:
        raise ValueError("Model not found")
