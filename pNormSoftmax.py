import torch
from utils.metrics import AURC
import numpy as np
from utils.measures import MSP

def centralize(y:torch.tensor):
    return y-(y.mean(-1).view(-1,1))
def p_norm(y:torch.tensor,p, eps:float = 1e-12):
    if p is None or p == 0: return torch.ones(y.size(0),1,device=y.device)
    else: return y.norm(p=p,dim=-1).clamp_min(eps).view(-1,1)
def beta_heuristic(y:torch.tensor,p):
    if p==0 or p is None: return 1
    return (p_norm(y,p).mean())

def beta_generalized_mean(n,p):
    if p==0 or p is None: return 1
    else: return (n*np.math.factorial(p))**(1/p) 
    
def pNormSoftmax(y:torch.tensor,p,beta= None, out = 'MSP'):
    '''Implement pNormSoftmax (centralize the logits, p-normalize, scale by beta and
    apply MSP).
    If beta is None, defines beta as the heuristic beta (mean of the p-norms).
    If out is passes as 'logits', return the normalized logits (skip MSP)'''
    
    y = centralize(y)
    norm = p_norm(y,p)
    if beta is None: beta = norm.mean()
    if out == 'logits':
        return y.mul(beta).div(norm)
    else: return MSP(y.mul(beta).div(norm))
    


class optimize:
    '''Gradient methods could be used, but a grid search
    on a small set of p's show to be strongly efficient for pNormSoftmax optimization.
    Also, AURC and AUROC are not differentiable'''
    p_range = torch.arange(8)
    T_range = torch.arange(0.01,2,0.01)
    @staticmethod
    def p_and_beta(logits,risk,metric = AURC,p_range = p_range,T_range =T_range):
        vals = optimize.p_T_grid(logits,risk,metric,p_range,T_range)
        p,T = np.unravel_index(np.argmin(vals),np.shape(vals))
        p = p_range[p]
        T = T_range[T]
        return p,beta_heuristic(logits,p)/T
    @staticmethod
    def p(logits, risk,metric = AURC,p_range = p_range, heuristic = True):
        if heuristic: beta = None
        else: beta = 1.0
        vals = optimize.p_grid(logits,risk,metric,p_range, beta)
        p = p_range[np.argmin(vals)]
        return p
    @staticmethod
    def T(logits, risk,metric = AURC,T_range = T_range):
        vals = optimize.T_grid(logits,risk,metric,T_range)
        return T_range[np.argmin(vals)]
    @staticmethod
    def T_grid(logits,risk,metric = AURC,T_range = T_range):
        vals = []
        for T in T_range:
            vals.append(metric(risk,MSP(logits.div(T))).item())
        return vals
    @staticmethod
    def p_grid(logits,risk,metric = AURC,p_range = p_range, beta = None):
        vals = []
        for p in p_range:
            vals.append(metric(risk,pNormSoftmax(logits,p,beta)).item())
        return vals
    @staticmethod
    def p_T_grid(logits,risk,metric = AURC,p_range = p_range,T_range = T_range):
        vals = []
        for p in p_range:
            vals_T = optimize.T_grid(pNormSoftmax(logits,p,None,'logits'),risk,metric,T_range)
            vals.append(vals_T)
        return vals
    
def optimal_pNormSoftmax(z:torch.tensor,risk:torch.tensor,metric = AURC,optimize_beta = False,**kwargs):
    if optimize_beta: p,beta = optimize.p_and_beta(z,risk,metric,**kwargs)
    else: 
        p = optimize.p(z, risk,metric, **kwargs)
        beta = beta_heuristic(z,p)
    return pNormSoftmax(z,p,beta)