from GuideDiffusion_Class    import GuideDiffusion_cls,GuideDiffusionTrainer
from util  import Dataloader as Loader_cls
from util   import *

from nn import Unet_3D       as Unet_3D

import torch
import os
import torch.utils.data as Data
import random

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def init():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
def Train_Diffusion():
    #----------------------------
    RootPath  = os.getcwd()
    Target    = "Diffusion"
    DataPath  = RootPath + "/Data/%s/"%Target
    SavePath  = RootPath + "/Parafile"
    DEVICE    = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    SHAPE     = (96,96,96)
    BATCH     = 1
    LR        = 1E-6
    T         = 1000
    Epoch     = 2000
    ModelS    = 72
    #----------------------------
    TrainSet  = Loader_cls     (DataPath,Range=[0,20],shape=SHAPE,scale=(1/5))
    TrainLoad = Data.DataLoader(dataset = TrainSet,batch_size = BATCH,shuffle = True)
    #----------------------------
    Diffusion = GuideDiffusion_cls(T=T,device=DEVICE,shape=SHAPE)
    Diffusion.InitModelPara(dim=ModelS)
    Opti      = torch.optim.Adam(Diffusion.Model.parameters(),lr = LR)
    Trainer   = GuideDiffusionTrainer(Diffusion=Diffusion,epoches=Epoch,optimizer=Opti,\
                            train_loader=TrainLoad,val_loader=TrainLoad,\
                            device=DEVICE,T=T,savepath=SavePath,Adjust_s=0.95)
    Diffusion = Trainer.train("%s-%s"%("Diffusion",ModelS))
    #----------------------------


if __name__ == "__main__":
    Train_Diffusion()