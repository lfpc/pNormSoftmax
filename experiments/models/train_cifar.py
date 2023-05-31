import os
MAIN_PATH = r'/home/luis-felipe'
DATA_PATH = os.path.join(MAIN_PATH,'data')
PATH_MODELS = os.path.join(MAIN_PATH,'torch_models')

import torch
from torchvision import transforms, datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../..')

import models
from utils import measures,metrics
from data_utils import split
import pNormSoftmax
import CIFAR

PROJECT = 'pNorm during training'
GROUP = 'Cifar100'
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)



# Define o computador utilizado como cuda (gpu) se existir ou cpu caso contrário
print('cuda:', torch.cuda.is_available())
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ["WANDB_SILENT"] = "True"
wandb.login()


CREATE_DIR = True #If true, creates directories to save model (weights_path)
LIVE_PLOT = False #If True, plot* loss while training. If 'print', print loss per epoch
SAVE_CHECKPOINT = True #If True, save (and update) model weights for the best epoch (smallest validation loss)
SAVE_ALL = False #If True, saves weights and trainer at the end of training

DATA = 'Cifar100'

VAL_SIZE = 0.1
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR.MEAN, CIFAR.STD),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR.MEAN, CIFAR.STD),
])

train_data,val_data = split.split_dataset(datasets.CIFAR100(
    root=DATA_PATH, train=True, download=True, transform=transform_train),VAL_SIZE)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True, num_workers=2)
testset = datasets.CIFAR100(
    root=DATA_PATH, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
num_classes = 100



loss_criterion = torch.nn.CrossEntropyLoss()
N_EPOCHS = 200




risk_dict = {'accuracy': TE.correct_total}#{'selective_risk_mcp':  lambda x,label: unc_comp.selective_risk(x,label,unc_type = unc.MCP_unc)}

from uncertainty import entropy, MCP_unc
risk_dict_accumulate = {'AURC - MCP': AURC_fn,
                        'AURC - Entropy':lambda x,y: metrics.AURC(x,y,entropy(x,normalize = True)),
                        'AUROC - MCP': AUROC_fn,
                        'AUROC - Entropy':lambda x,y: metrics.AUROC(x,y,entropy(x,normalize = True)),
                        'ECE': metrics.ECE(10,softmax = True),
                        'NormL1': lambda x,y: x.norm(dim=-1,p=1).mean(),
                        'NormL2': lambda x,y: x.norm(dim=-1,p=2).mean(),
                        'NormL4': lambda x,y: x.norm(dim=-1,p=4).mean(),
                        'AUROC L4': lambda x,y: metrics.AUROC(x,y,MCP_unc(norm_p_heuristic(x,4),normalize = True)),
                        'AURC L4': lambda x,y: metrics.AURC(x,y,MCP_unc(norm_p_heuristic(x,4),normalize = True)),}



def train(MODEL_ARC:str):
    weights_path = os.path.join(WEIGHTS_PATH,DATA,MODEL_ARC)
    if CREATE_DIR and not os.path.isdir(weights_path):
        os.makedirs(weights_path)
    model_class = models.__dict__[MODEL_ARC]
    model = model_class(num_classes = data.n_classes).to(dev)

    name = f'{MODEL_ARC}_{DATA}'
    CONFIG = {'Architecture': MODEL_ARC, 'Dataset': DATA, 'N_epochs':N_EPOCHS, 'Validation' : VAL_SIZE}
    WandB = {'project': PROJECT, 'group': GROUP, 'config': CONFIG, 'name': name}

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,momentum = 0.9,weight_decay = 5e-4,nesterov = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)#StepLR(optimizer, 25, gamma=0.5)
    CONFIG['optimizer'] = type(optimizer).__name__
    CONFIG['scheduler'] = type(scheduler).__name__
    model_trainer = TE.Trainer_WandB(model,optimizer,loss_criterion, data.train_dataloader,data.test_dataloader,lr_scheduler = scheduler,**WandB, risk_dict = risk_dict, risk_dict_accumulate=risk_dict_accumulate)
    print('start')
    model_trainer.fit(data.train_dataloader,N_EPOCHS, live_plot = LIVE_PLOT,save_checkpoint = SAVE_CHECKPOINT,PATH = weights_path)

if __name__ == '__main__':
    for MODEL_ARC in models.list_models_cifar()[17:]:
        print(MODEL_ARC)
        train(MODEL_ARC)