import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skimage.morphology import diamond

class TexturalMetric:
    def __init__(self, device, rad=2):
        self.device = device
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights = 'ResNet18_Weights.IMAGENET1K_V1').to(device)
        self.latent = {}
        self.model.bn1.register_forward_hook(self.hook)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.rad = rad
        self.neighbour_kernel = torch.from_numpy(diamond(self.rad)).to(self.device).float()[None, None]
        
    def hook(self, module, input, output):
        self.latent['latent'] = output
        return output

    def get_separability_score(self, img, msk, clf_model_key='LogisticRegression(n_jobs=64,C = 2)'):
        '''Requiring the image to have dimension (3,H,W) and the object mask be have dimension(1,H,W)'''
        assert img.ndim ==3
        assert msk.ndim ==3
        assert torch.is_tensor(img) and torch.is_tensor(msk)
        
        clf = eval(clf_model_key)
        
        # Transform image into appropriate size and get latent representation
        h0, w0 = img.shape[1:]
        ratio = min(1024/h0, 1024/w0)
        h, w = int(h0*ratio+.5), int(w0*ratio+.5)
        img_trans = self.normalize(T.functional.resize(img, (h,w)))
        _ = self.model(img_trans.unsqueeze(0))
        
        latent_X = self.latent['latent'][0].detach().squeeze().permute([1,2,0])

        # Get texture (feature) on pixels around target object
        raw_msk = T.functional.resize(msk, latent_X.shape[:2], interpolation=T.InterpolationMode.NEAREST).squeeze().bool()
        
        dilated_msk = F.conv2d(raw_msk[None,None].float(), self.neighbour_kernel, padding=self.rad).squeeze()>0 
        out_contour = torch.logical_and(dilated_msk, ~raw_msk)
        
        latent_pos = latent_X[raw_msk].cpu().numpy()
        latent_out = latent_X[out_contour].cpu().numpy()

        if len(latent_pos)<10:
            return -1
                
        num_sampels  = min([500,len(latent_pos),len(latent_out)])
        for _ in range(3):
            np.random.shuffle(latent_pos)
            np.random.shuffle(latent_out)
                    
        XX = np.concatenate([latent_pos[:num_sampels],latent_out[:num_sampels]])
        yy = np.array([0]*num_sampels+[1]*num_sampels)

        # train weak clf to separate the model 
        clf = eval(clf_model_key)
        clf.fit(XX,yy)
        return clf.score(XX,yy)
    
            
        
        