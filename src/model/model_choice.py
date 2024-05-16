from src.model.convnet import ConvNet
from src.model.dnn import DNN
from src.model.vgg import init_vgg


def model_choice(model : str, input_size, num_classes):
    if model == "ConvNet":
        return ConvNet(False)
    elif model == "DNN":
        return DNN(input_size, num_classes, False)
    elif model == "VGG":
        return init_vgg(num_classes)
    else:
        raise ValueError("Model not found")