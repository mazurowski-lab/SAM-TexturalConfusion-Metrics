import torch, os
import torch.nn.functional as F
from skimage.morphology import diamond

def get_GiniStd(object_msk, scale, device):
    '''
    Assume scale (window size) to be an odd number, 
    object mask is expected to have dimension (1, H, W)'''
    assert torch.is_tensor(object_msk)
    assert scale%2==1
    assert object_msk.ndim == 3

    # add one batch dim to make work with F.conv2d:
    object_msk = torch.unsqueeze(object_msk, 0)
    
    kernel = torch.ones(1,1,scale,scale).float().to(device)/(scale**2)
    try:
        out = F.conv2d(object_msk.float(), kernel)
        gini = 1-out**2-(1-out)**2
        gini_std = gini.std().item()
    except:
        gini_std  = -1

    return gini_std

def get_DoGD(object_msk, a=127, b=3, device="cuda"):
    '''return DoGD if it is computable, otherwise return -1'''
    assert a > b
    a_gini_std = get_GiniStd(object_msk, a, device)
    b_gini_std = get_GiniStd(object_msk, b, device)
    
    return a_gini_std - b_gini_std if a_gini_std >0 else -1

def get_CPR(object_msk, rad=5, device="cuda"):
    '''rad is the radius of the kernel for determining the neighborhood distance, 
    object mask is expected to has dimension (1,H,W)'''
    
    assert object_msk.ndim == 3
    assert torch.is_tensor(object_msk)

    # add one batch dim to make work with F.conv2d:
    object_msk = torch.unsqueeze(object_msk, 0)

    kernel = torch.from_numpy(diamond(rad)).to(device).float()[None, None]
    conv_map = F.conv2d(object_msk.float(), kernel, padding=rad) 

    # a convolved kernal should have full sume if it is not adjacent to any background pixel
    isPosContourPx = torch.logical_and(object_msk>0, conv_map <= (kernel.sum()-1))
    
    pos_ratio = (isPosContourPx.double().clip(max=1).sum()/object_msk.double().sum()).item()
    assert pos_ratio<=1
    return pos_ratio