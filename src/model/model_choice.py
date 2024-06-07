from src.model.alexnet import AlexNet
from src.model.convmixer import ConvMixer, convmixer
from src.model.convnet import ConvNet
from src.model.dnn import DNN
from src.model.resnet import ResNet
from src.model.vgg import vgg11, vgg11_bn


def model_choice(model : str, input_size, num_classes, out_layer=0):
    if model == "ConvNet":
        return ConvNet(False, out_layer), ConvNet(True, out_layer)
    elif model == "DNN":
        return DNN(input_size, num_classes, False)
    elif model == "VGG":
        return vgg11_bn(linear=False), vgg11_bn(linear=True) # VGG11(num_classes)
    elif model == "ConvMixer":
        return convmixer(linear=False), convmixer(linear=True)
    elif model == "ResNet":
        return ResNet(False, num_classes), ResNet(True, num_classes)
    elif model == "AlexNet":
        return AlexNet(False, num_classes), AlexNet(True, num_classes)
    else:
        raise ValueError("Model not found")