from src.model.convnet import ConvNet
from src.model.dnn import DNN
from src.model.vgg import vgg11, vgg11_bn


def model_choice(model : str, input_size, num_classes):
    if model == "ConvNet":
        return ConvNet(False)
    elif model == "DNN":
        return DNN(input_size, num_classes, False)
    elif model == "VGG":
        return vgg11_bn(linear=False), vgg11_bn(linear=True) # VGG11(num_classes)
    else:
        raise ValueError("Model not found")