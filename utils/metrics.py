from sklearn.metrics import roc_curve,auc
import torch
from .measures import wrong_class,correct_class

def accuracy(y_pred,y_true):
    '''Returns the accuracy in a batch'''
    return correct_class(y_pred,y_true).sum()/y_true.size(0)


def ROC_curve(loss, uncertainty, return_threholds = False):
    fpr, tpr, thresholds = roc_curve(loss.cpu(),uncertainty.cpu())
    if return_threholds:
        return fpr,tpr,thresholds
    else:
        return fpr,tpr
    
def RC_curve(loss:torch.tensor, uncertainty:torch.tensor = None,coverages = None, expert=False, expert_cost=0, confidence = None,return_coverages = False):
    loss = loss.view(-1)
    if uncertainty is None:
        if confidence is not None:
            uncertainty = -confidence
    uncertainty = uncertainty.view(-1)
    n = len(loss)
    assert len(uncertainty) == n
    uncertainty,indices = uncertainty.sort(descending = False)
    loss = loss[indices]
    if coverages is not None:
        coverages = torch.as_tensor(coverages,device = uncertainty.device)
        thresholds = uncertainty.quantile(coverages)
        indices = torch.searchsorted(uncertainty,thresholds).minimum(torch.as_tensor(uncertainty.size(0)-1,device=uncertainty.device))
    else:
        thresholds,indices = uncertainty.unique_consecutive(return_inverse = True)
    coverages = (1 + indices)/n
    risks = (loss.cumsum(0)[indices])/n

    if expert:
        if torch.any(expert_cost):
            expert_cost = torch.as_tensor(expert_cost,device=uncertainty.device).reshape(-1)
            if expert_cost.size == 1:
                risks += (1 - coverages)*expert_cost
            else:
                assert len(expert_cost) == n
                expert_cost = torch.cumsum(expert_cost)
                expert_cost = expert_cost[-1] - expert_cost
                risks += expert_cost[indices]/n
    else:
        risks /= coverages
    if return_coverages:
        return coverages.cpu().numpy(), risks.cpu().numpy()
    else: return risks.cpu().numpy()

def AUROC(loss,uncertainty):
    fpr,tpr = ROC_curve(loss,uncertainty)
    return auc(fpr, tpr)

def AURC(loss,uncertainty, coverages = None):
    coverages,risk_list = RC_curve(loss,uncertainty, coverages,return_coverages = True)
    return auc(coverages,risk_list)

def AUROC_fromlogits(y_pred,y_true,uncertainty, risk_fn = wrong_class):
    risk = risk_fn(y_pred,y_true).float()
    return AUROC(risk,uncertainty)

def AURC_fromlogits(y_pred,y_true,uncertainty, risk_fn = wrong_class, coverages = None):
    risk = risk_fn(y_pred,y_true).float()
    return AURC(risk,uncertainty,coverages)


class ECE(torch.nn.Module):
    
    '''From https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py :'''
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence y_preds into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10, softmax = False):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.SM = softmax

    def forward(self, y, labels):
        if self.SM:
            y = torch.nn.functional.softmax(y, dim=1)
        confidences, predictions = torch.max(y, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=y.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class AdaptiveECE(torch.nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15, softmax = False):
        super(AdaptiveECE, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return torch.nn.functional.interpolate(torch.linspace(0, npt, self.nbins + 1),
                     torch.arange(npt),
                     torch.sort(x))
    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = torch.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ClasswiseECE(torch.nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce