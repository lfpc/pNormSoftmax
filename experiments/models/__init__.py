from CIFAR import list_models as list_models_cifar
from CIFAR import get_model as get_model_cifar
from ImageNet import list_models as list_models_imagenet
from ImageNet import get_model as get_model_imagenet

def list_models(data:str = 'ImageNet'):
    if data.lower() == 'imagenet':
        return list_models_imagenet()
    elif data.lower() == 'cifar':
        return list_models_cifar()
    
def get_model(MODEL_ARC:str,data:str = 'ImageNet',pretrained = True,return_transforms = True):
    if data.lower() == 'imagenet':
        return get_model_imagenet(MODEL_ARC,pretrained,return_transforms)
    elif data.lower() == 'cifar':
        return get_model_cifar(MODEL_ARC,pretrained,return_transforms)