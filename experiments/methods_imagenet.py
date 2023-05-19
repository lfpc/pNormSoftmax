import sys
import os
sys.path.insert(1, '..')
sys.path.insert(1, '../..')
MAIN_PATH = r'/home/luis-felipe'
DATA_PATH = os.path.join(MAIN_PATH,'data')
PATH_RESULTS = os.path.join(MAIN_PATH,'results')
PATH_MODELS = os.path.join(MAIN_PATH,'torch_models','outputs')

DATASET = 'test'
NAME = f'Results_2.csv'

import torch
torch.cuda.empty_cache()
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

import numpy as np
np.random.seed(42)
import pandas as pd
import NN_models as models
import NN_utils.train_and_eval as TE
import torch_data
from collections import defaultdict
from uncertainty import MCP_unc, entropy, energy
from uncertainty import metrics
from math import sqrt,factorial



# Define o computador utilizado como cuda (gpu) se existir ou cpu caso contrário
print('cuda: ',torch.cuda.is_available())
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA = 'ImageNet'
num_classes = 1000
VAL_SIZE = 0.1
data_params = {'train_batch_size': 40, 'validation_size': 0, 'test_batch_size': 40, 'split_train': 0.08}

c_list = torch.arange(0.05,1.05,0.05)
T_range = torch.arange(0.01,1.5,0.01)
p_range = list(range(7))
metrics_dict = {}
metrics_dict['AURC'] = lambda x,y,m: metrics.AURC(x,y,m(x))
metrics_dict['AUROC'] = lambda x,y,m: 1-metrics.AUROC(x,y,m(x)) #1- for the minimum the best
metrics_dict['ECE'] = lambda x,y,m: metrics.ECE(n_bins = 10,softmax=True)(x,y)
#metrics_dict['NLL'] = lambda x,y,m: torch.nn.functional.cross_entropy(x,y)
loss_criterion = torch.nn.CrossEntropyLoss()
measures_dict = {'MCP': lambda x: MCP_unc(x,True), 'Entropy': lambda y: entropy(y,normalize=True), 
                 'MaxLogit': lambda y: y.max(-1).values, 'Diff_p': lambda y: y.softmax(-1).topk(2,dim=-1).values.t()[1]-y.softmax(-1).topk(2,dim=-1).values.t()[0],
                   'Diff_logit': lambda y: y.topk(2,dim=-1).values.t()[1]-y.topk(2,dim=-1).values.t()[0]}#,'Energy': lambda y: y.exp().sum(-1).log().maximum(torch.tensor(100,device=dev))}

def T_grid(outputs,labels,method,metric,T_range = T_range):
    vals = []
    for T in T_range:
        vals.append(metric(outputs.div(T),labels,method).item())
    return vals
def optimize_T(outputs, labels,method,metric,T_range = T_range):
    vals = T_grid(outputs,labels,method,metric,T_range)
    return T_range[np.argmin(vals)]
def p_momentum(y:torch.tensor,p):
     if p is None or p == 0: return torch.ones(y.size(0),1,device=dev)
     else: return y.pow(p).mean(-1).abs().pow(1/p).unsqueeze(-1)
def p_norm(y:torch.tensor,p):
    if p is None or p == 0: return torch.ones(y.size(0),1,device=dev)
    return y.norm(dim=-1,p=p).unsqueeze(-1)

class split_class():
    def __init__(self,validation_size, n = 50000):
        self.val_index = torch.randperm(n)[:int(validation_size*n)]
        self.test_index = torch.randperm(n)[int(validation_size*n):]
    def split_val_test(self,outputs,labels):
        outputs_val,labels_val = outputs[self.val_index],labels[self.val_index]
        outputs_test,labels_test = outputs[self.test_index],labels[self.test_index]
        return outputs_val,labels_val,outputs_test,labels_test
split = split_class(VAL_SIZE)
def get_outputs(MODEL_ARC:str, split = 'test'):
    if f'{MODEL_ARC}_{DATA}_outputs_{split}.pt' in os.listdir(PATH_MODELS):
        outputs = torch.load(os.path.join(PATH_MODELS,f'{MODEL_ARC}_{DATA}_outputs_{split}.pt')).to(dev)
        labels = torch.load(os.path.join(PATH_MODELS,f'{DATA}_labels_{split}.pt')).to(dev)
        return outputs,labels
    else:
        data = torch_data.__dict__[DATA](data_dir = os.path.join(DATA_PATH),validation_as_train = False,params = data_params, train = (DATASET == 'train'),test=(DATASET != 'train'))
        
        model_class = models.__dict__[MODEL_ARC]
        weights = models.get_weight(MODEL_ARC)
        try: classifier = model_class(weights = weights).to(dev)
        except: classifier = model_class(weights).to(dev)
        classifier = classifier.to(torch.float32)
        data.transforms_test = weights.transforms()
        data.change_transforms(transforms_test = data.transforms_test)
        classifier.eval();

        if split == 'test':
            outputs,labels =  TE.accumulate_results(classifier,data.test_dataloader)
        elif split == 'train':
            outputs,labels = TE.accumulate_results(classifier,data.train_dataloader)
        elif split == 'val':
            outputs,labels = TE.accumulate_results(classifier,data.val_dataloader)
        torch.save(outputs, os.path.join(PATH_MODELS,f'{MODEL_ARC}_{DATA}_outputs_{split}.pt'))
        torch.save(labels,os.path.join(PATH_MODELS,f'{DATA}_labels_{split}.pt'))
        return outputs.to(torch.get_default_dtype()),labels
    


def main(outputs:torch.tensor,labels) -> dict:
    outputs_val,labels_val,outputs,labels = split.split_val_test(outputs,labels)
    d = defaultdict(list)
    for p in p_range:
        for centralize in [True,False]:
            for p_method in ['norm']:
                if centralize:
                    y_pred = outputs-outputs.mean(-1).unsqueeze(-1)
                    y_pred_val = outputs_val-outputs_val.mean(-1).unsqueeze(-1)
                else: 
                    y_pred = outputs
                    y_pred_val = outputs_val
                if p_method =='momentum':
                    norm_p = p_momentum(y_pred,p)
                else: 
                    norm_p = p_norm(y_pred,p)
                    norm_p_val = p_norm(y_pred_val,p)
                y_pred = y_pred.div(norm_p)    
                for metric_opt in ['AURC', 'AUROC', 'ECE']:
                    for measure in measures_dict.keys():
                        if (metric_opt == 'ECE' or metric_opt == 'NLL') and measure != 'MCP':
                            continue
                        for TS_method in ['heuristic','optimal']:
                            if TS_method == 'heuristic':
                                T =  1/norm_p.mean().item()
                                T_val =  1/norm_p_val.mean().item()
                            elif TS_method =='optimal': #tem que entrar se é mcp, entropy, etc
                                T = optimize_T(y_pred,labels,measures_dict[measure],metric=metrics_dict[metric_opt],T_range = T_range).item()
                                T_val = optimize_T(y_pred_val,labels_val,measures_dict[measure],metric=metrics_dict[metric_opt],T_range = T_range).item()
                            if centralize:
                                y_pred = (outputs-outputs.mean(-1).unsqueeze(-1)).div(norm_p*T)
                            else: y_pred = (outputs).div(norm_p*T)

                            d['metric_opt'].append(metric_opt)
                            d['TS_method'].append(TS_method)
                            d['T'].append(T)
                            d['T_val'].append(T_val)
                            d['logits mean'].append(y_pred.mean().item())
                            d['logits mean var'].append(y_pred.mean(-1).var().item())
                            d['logits max'].append(y_pred.max(-1).values.mean().item())
                            d['logits margin'].append((y_pred.topk(2,dim=-1).values.t()[0]-y_pred.topk(2,dim=-1).values.t()[1]).mean().item())
                            d['logits low999'].append(y_pred.topk(outputs.size(-1)-1,dim=-1,largest=False).values.mean().item())
                            d['logits top5'].append(y_pred.topk(5,dim=-1,largest=True).values[:,1:].mean().item())
                            d['logits low100'].append(y_pred.topk(100,dim=-1,largest=False).values.mean().item())
                            d['pnorm auroc'].append(metrics.AUROC(outputs,labels,norm_p))
                            d['pnorm mean'].append(norm_p.mean().item())
                            d['pnorm low999 mean'].append(outputs.topk(outputs.size(-1)-1,dim=-1,largest=False).values.norm(dim=-1,p=p).mean().item())
                            d['pnorm top5 mean'].append(outputs.topk(5,dim=-1,largest=True).values[:,1:].norm(dim=-1,p=p).mean().item())
                            d['P-method'].append(p_method)
                            d['centralize'].append(centralize)
                            d['p'].append(p)
                            d['acc'].append(outputs.argmax(-1).eq(labels).float().mean().item()*100)
                            d['acc top_5'].append(TE.top_k_accuracy(outputs,labels,5).item())
                            d['measure'].append(measure)
                        
                            
                            unc = measures_dict[measure](y_pred)
                            d['unc mean'].append(unc.mean().item())
                            d['unc means var'].append(unc.var().item())
                            for name,metric in metrics_dict.items():
                                d[name].append(metric(y_pred,labels,measures_dict[measure]).item())
                                if centralize:
                                    y_pred = (outputs-outputs.mean(-1).unsqueeze(-1)).div(norm_p*T_val)
                                else: y_pred = (outputs).div(norm_p*T_val)
                                d[name + '- val'].append(metric(y_pred,labels,measures_dict[measure]).item())

    return d
    


if __name__ == '__main__':
    try: df = pd.read_csv(os.path.join(PATH_RESULTS,NAME),index_col = 0)
    except: df = pd.DataFrame()
    for MODEL_ARC in models.list_models():
        print(MODEL_ARC)
        if MODEL_ARC in df['model'].values:
            print('ja foi')
            continue
        if 'quantized' in MODEL_ARC:
            print('quantized - skip')
            continue
        with torch.no_grad():
            if MODEL_ARC in models.__dict__:
                outputs,labels = get_outputs(MODEL_ARC, DATASET)
            else: 
                print('cant find')
                continue
            d = pd.DataFrame(main(outputs.to(dev,torch.get_default_dtype()),labels.to(dev)))
        d['model'] = MODEL_ARC
        d = d.round(5)
        df = pd.concat((df,d))
        df.to_csv(os.path.join(PATH_RESULTS,NAME))
    print(df)