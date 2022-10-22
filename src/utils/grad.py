import torch as tc
from torch.autograd import grad
from itertools import product

from src.utils.pyhessian import hessian

def input_grad(imgs, targets, model, criterion):
    output = model(imgs)
    loss = criterion(output, targets)
    ig = grad(loss, imgs)[0]
    return ig

def hessian_loss(loss, x):
    ig1 = grad(loss, x, create_graph=True, allow_unused=True)[0]
    return hessian_ig(ig1, x)
    
def hessian_ig(ig1, x):
    dim = ig1.size()
    ig2 = tc.zeros(dim+dim[1:]).cuda()
    for i, j, k in product(range(dim[1]), range(dim[2]), range(dim[3])):
        ig1_sum = tc.zeros(0).cuda()
        for l in range(dim[0]):
            ig1_sum = ig1[l][i][j][k]
            
        if i==dim[1]-1 and j==dim[2]-1 and k==dim[3]-1:
            retain = False
        else:
            retain = True
        ig2_sub = grad(ig1_sum, x, retain_graph=retain)[0]
        for l in range(dim[0]):
            ig2[l][i][j][k] = ig2_sub[l]
    return ig2.detach()
    
def hessian_spectrum(imgs, targets, model, criterion, top_n=1):
    hessian_comp = hessian(model, criterion, data=(imgs, targets), cuda=imgs.device)
    eigen, _ = hessian_comp.eigenvalues(top_n=top_n)
    return eigen
