import torch
from torch import nn
import numpy as np
import math
from inspect import isfunction

import torch.nn.functional as F
from tqdm.auto import tqdm
import os
from torch.autograd import Variable
import tifffile
import time
from torch.utils.data import Dataset
import cv2
import numpy as np
from struct import unpack
import gzip
import warnings
from torch.fft import fftn,fftshift,ifftn,ifftshift
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import threading 
import queue

#coding=utf-8
import json
def saveobj(obj,path):
    with open(path,'wb') as file:
        s = json.dumps(obj)
        s = s.encode("utf-8")
        file.write(s)

def loadobj(path):
    with open(path, 'rb') as file:
        s   = file.read()
        s   = s.decode("utf-8")
        obj = json.loads(s)
    return obj

class SaveManager(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        while True:
            Mes = self.q.get(block=True)
            if Mes is None:
                return None
            Block,path = Mes
            tifffile.imwrite(path,Block.transpose(2,0,1))
    def GetQueue(self):
        self.q = queue.Queue()
        return self.q

def loadblock(path):
    try:
        orig        = np.float32(tifffile.imread(path))
        orig        = orig.transpose(2,1,0).transpose(1,0,2)
        orig        = torch.from_numpy(orig)
        return orig
    except:
        return None
  
def SaveImg(r,path):
    r = r.clone().cpu().numpy().transpose(2,0,1)
    tifffile.imwrite(path,r)

def TimeCheck(t = None):
    nt  = time.time()
    if not t is None:
        print("Time:%.6lfsec"%(nt-t))
    else:
        print("Time:start")
    return nt

def cos_loss(a,b):
    a      = (a - a.min())/(a.max() - a.min())
    a      = a.flatten()
    b      = (b - b.min())/(b.max() - b.min())
    b      = b.flatten()
    _a,_b  = torch.norm(a,2),torch.norm(b,2)
    Loss   = torch.sum(a*b)/(_a*_b)
    return 1-Loss

def load_seq(name):
    import pickle
    with open(name, 'rb') as f:
        seq = pickle.load(f)
    seq = np.array(seq).astype(np.float32)
    seq = torch.from_numpy(seq).cuda()
    return seq


def PrintToFile(txt):
    with open("output.txt","a") as f:
        f.write(txt + "\n")

def CheckDistu(img):
    img = img.flatten()
    print(torch.mean(img),torch.std(img))

def LoadStdImg(IMGPath,ST = 0,END = -1):
    orig        = np.float32(tifffile.imread(IMGPath))
    orig        = orig.transpose(2,1,0).transpose(1,0,2)
    orig        = (orig - np.min(orig))/(np.max(orig) - np.min(orig))
    if END == -1:
        orig        = orig[...,ST:] #
    else:
        orig        = orig[...,ST:END] #
    orig        = torch.from_numpy(orig)
    orig        = (orig - orig.mean())/orig.std()
    return  orig

def LoadImg(IMGPath):
    orig        = np.float32(tifffile.imread(IMGPath))
    # import matplotlib.pyplot as plt
    # plt.imshow(orig)
    # plt.show()
    orig        = orig.transpose(2,1,0).transpose(1,0,2)
    # orig        = orig - 300
    # orig[orig<1]=1
    orig        = (orig - np.min(orig))/(np.max(orig) - np.min(orig))
    orig        = torch.from_numpy(orig)
    return  orig
  
def VarianceSchedule(type = "liner",T = 1000, b0=1e-1, bT=1-1e-1,revese =False):
    if type == "liner":
        if not revese:
            return torch.linspace(b0,bT,T).to(torch.float64)
        else:
            return torch.linspace(bT,b0,T).to(torch.float64)
        
#------------------------------------------------------

def AdjuestRange(Img):
    return (Img - np.mean(Img))/np.std(Img)

def LoadFileitem(Path,shape,Augment):
    X,Y,Z = shape
    img   = tifffile.imread(Path).astype(np.float32)
    img   = img.transpose(2,1,0).transpose(1,0,2)
    H,W,C = img.shape
    res   = np.zeros((X,Y,Z))
    if Augment:
        SC    = np.random.randint(0,C-Z) if C>Z else 0
        SW    = np.random.randint(0,W-Y) if Y>W else 0
        SH    = np.random.randint(0,H-X) if X>H else 0
    else:
        SC = SW = SH = 0
    res   = img[SH:SH+X,SW:SW+Y,SC:SC+Z] 
    return AdjuestRange(res)
    
def RandomRot(Img):
    return np.rot90(Img,np.random.randint(0,4),axes=(0,1))

class Dataloader(Dataset):
    def __init__(self,DataPath,Range=[0,19],\
                 shape = (72,72,48),scale=3,Augment=True):
        self.shape    = shape
        self.DataPath = DataPath
        self.PathIter = os.listdir(DataPath)
        self.Augment  = Augment
        self.Data = []
        for i in range(len(self.PathIter)):
            if Range[0] <= (i%20) < Range[1]:
                self.Data.append(self.PathIter[i])
        self.Len   = int(len(self.Data)*scale)
        self.scale = scale
    def __len__(self):
        return self.Len 
    def __getitem__(self,i,):
        i   = i//self.scale
        i  += np.random.randint(0,int(1/self.scale)) if self.scale < 1 else 0
        img = LoadFileitem(self.DataPath+"/"+self.Data[int(i)],self.shape,self.Augment)
        tag = np.zeros((1))
        filename = self.Data[int(i)]
        Cls      = filename[0]
        tag[0] = 3 if Cls == "T"             else \
                 6 if Cls == "F"             else \
                 9 if Cls == "C"             else 0 
        if tag[0] != 0:
            Rank     = float(filename.split("-")[1])
            tag[0] -= (1-Rank)
        tag[0] = 0 if np.random.random()<0.5 else tag[0]
        if self.Augment:img = RandomRot(img).copy()
        img = (img - img.mean())/img.std()
        return torch.from_numpy(img),\
               torch.from_numpy(tag)

def Gausskernel2D(kernel,Sig=1):
    X     = torch.arange(-kernel, kernel+1, 1)
    Y     = torch.arange(-kernel, kernel+1, 1)
    x,y   = torch.meshgrid(X,Y)
    Gauss = torch.exp(-(((x**2)/(2*(Sig**2)) +\
                         (y**2)/(2*(Sig**2)) )))
    Gauss = (1/Gauss.sum())*Gauss
    return Gauss

def Gausskernel3D(kernel,R=(1,1)):
    if len(R) == 2:
        sxy = R[0] if R[0] >= 1e-3 else 1e-3
        sz  = R[1] if R[1] >= 1e-3 else 1e-3
        k1,k2 = kernel
        X     = torch.arange(-k1, k1+1, 1)
        Y     = torch.arange(-k1, k1+1, 1)
        Z     = torch.arange(-k2, k2+1, 1)
        x,y,z = torch.meshgrid(X,Y,Z)
        Gauss = torch.exp(-(((x**2)/(2*(sxy**2)) +\
                            (y**2)/(2*(sxy**2)) +\
                            (z**2)/(2*(sz**2)))))
    else:
        sx  = R[0] if R[0] >= 1e-3 else 1e-3
        sy  = R[1] if R[1] >= 1e-3 else 1e-3
        sz  = R[2] if R[2] >= 1e-3 else 1e-3
        k1,k2,k3 = kernel
        X     = torch.arange(-k1, k1+1, 1)
        Y     = torch.arange(-k2, k2+1, 1)
        Z     = torch.arange(-k3, k3+1, 1)
        x,y,z = torch.meshgrid(X,Y,Z)
        Gauss = torch.exp(-(((x**2)/(2*(sx**2)) +\
                             (y**2)/(2*(sy**2)) +\
                             (z**2)/(2*(sz**2)))))
    Div   = (1/Gauss.sum())
    Gauss = Gauss*Div
    return Gauss

def AddPSF(Img,Psf):
    Result   = torch.zeros_like(Img).unsqueeze(0).unsqueeze(0).cuda()
    Result   = F.conv2d(Img.unsqueeze(0).unsqueeze(0),\
                        Psf.unsqueeze(0).unsqueeze(0),\
                        padding = "same").squeeze(0).squeeze(0)
    return Result
#------------------------------------------------------
def DownSample3D(Img,step):
    _,H,W,C = Img.shape
    Img     = F.avg_pool3d(Img,step)
    Img     = Img.view  (_,H//step,1,W//step,1,C//step,1)
    Img     = Img.repeat(1,1,step,1,step,1,step)
    Img     = Img.view  (_,H,W,C)
    return Img

def DownSample2D(Img,step):
    _,H,W,C = Img.shape
    Img     = Img.permute(0,3,1,2)
    Img     = F.avg_pool2d(Img,step)
    Img     = Img.view  (_,C,H//step,1,W//step,1)
    Img     = Img.repeat(1,1,1,step,1,step)
    Img     = Img.view  (_,C,H,W)
    Img     = Img.permute(0,2,3,1)
    return Img

def DownSample2D_nopool(Img,step):
    _,H,W,C = Img.shape
    Img     = Img.permute(0,3,1,2)
    Img     = Img.view  (_,C,H,1,W,1)
    Img     = Img.repeat(1,1,1,step,1,step)
    Img     = Img.view  (_,C,H*step,W*step)
    Img     = Img.permute(0,2,3,1)
    return Img

def UpSample3D(Img,step):
    _,H,W,C = Img.shape
    Img     = Img.permute(0,3,1,2)
    Img     = Img.view  (_,C,1   ,H,   1,W,   1)
    Img     = Img.repeat(1,1,step,1,step,1,step)
    Img     = Img.view  (_,C*step,H*step,W*step)
    Img     = Img.permute(0,2,3,1)
    return Img

def UpSample2D(Img,step):
    _,H,W,C = Img.shape
    Img     = Img.permute(0,3,1,2)
    Img     = Img.view  (_,C,H,   1,W,   1)
    Img     = Img.repeat(1,1,1,step,1,step)
    Img     = Img.view  (_,C,H*step,W*step)
    Img     = Img.permute(0,2,3,1)
    return Img

#------------------------------------------------------

def load_mnist(rootpath, normalize=True, one_hot=True):
    def __read_image(path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            img=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        img = img.reshape(num,28,28)
        return img
    def __read_label(path):
        with gzip.open(path, 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            lab = np.frombuffer(f.read(), dtype=np.uint8)
            # print(lab[1])
        return lab 
    def __normalize_image(image):
        img = image.astype(np.float32) / 255.0
        return img
    def __one_hot_label(label):
        lab = np.zeros((label.size, 10))
        for i, row in enumerate(lab):
            row[label[i]] = 1
        return lab
    x_train_path,x_test_path = "train-images-idx3-ubyte.gz","t10k-images-idx3-ubyte.gz"
    y_train_path,y_test_path = "train-labels-idx1-ubyte.gz","t10k-labels-idx1-ubyte.gz"
    image = {
        'train' : __read_image(rootpath+"/"+x_train_path),
        'test'  : __read_image(rootpath+"/"+x_test_path)
    }
    label = {
        'train' : __read_label(rootpath+"/"+y_train_path),
        'test'  : __read_label(rootpath+"/"+y_test_path)
    }
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])
    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])
    return (image['train'], label['train']), (image['test'], label['test'])



class Mask_Manager_cls():
    def __init__(self,CR,\
                 MaskImg        = None,\
                 CurrentBalance = False,\
                 InclineCalibrationFlag = False,\
                 Rf=None,Rb=None,\
                 Device         = "cuda:0",\
                 ):
        self.Device                     = Device
        self.OMask                      = MaskImg
        H,W,_CR                         = self.OMask.shape
        #----------------
        self.Mask                       = self.OMask[...,:CR].to(self.Device)
        self.Mask_s                     = torch.sum(self.Mask,dim=-1)
        self.Mask_s[self.Mask_s==0]     = 1
        self.InclineCalibrationFlag     = InclineCalibrationFlag
        self.CR                         = CR
        self.InClineSwitch              = False
        self.DualCamGain                = 1
        self.CamScanMode                = "MiddleToEdge"
        self.AllPassMask                = torch.ones((*self.Mask.shape[:2],CR),\
                                            dtype = torch.float32).to(self.Device)
        #----------------
        self.Rf = Rf
        if not Rf is None:
            self.Kf   = Gausskernel3D(3,Rf).to(Device).unsqueeze(0).unsqueeze(1)
        #----------------
        self.Rb = Rb
        if not Rb is None:
            self.Kb   = Gausskernel3D(3,Rb).to(Device).unsqueeze(0).unsqueeze(1)
    def A(self,x,Maskpara=(0,0),Flag = "Pos",AddNoise = None):
        H,W,S = x.shape
        SH,SW = Maskpara
        CR    = self.CR
        #----------------
        if not self.Rf is None:
            rx        = F.conv3d(x.clone().unsqueeze(0).unsqueeze(0),\
                            self.Kf,padding = "same").squeeze(0).squeeze(0)
        else:
            rx        = x
        #----------------
        if not self.InClineSwitch:
            if   Flag == "Pos":
                Phi        = self.Mask[SH:SH+H,SW:SW+W,:]
            elif Flag == "AllPass":
                Phi        = torch.ones_like(self.Mask).to(self.Device)
                Phi        = Phi[SH:SH+H,SW:SW+W,:] 
            Step   = S//CR
            Result = torch.zeros((*Phi.shape[:2],Step),dtype = torch.float32).to(self.Device)

            for i in range(Step):
                Result[...,i] = torch.sum(Phi*rx[...,i*CR:(i+1)*CR],axis=2)
        else:
            CR     = self.CR
            if   Flag == "Pos":
                Phi_G    = [self.ClineMaskGroup[0][SH:SH+H,SW:SW+W,:].clone(),
                            self.ClineMaskGroup[1][SH:SH+H,SW:SW+W,:].clone()]
            elif Flag == "AllPass":
                Phi_G    = [torch.ones((H,W,CR)).to(self.Device)*1/CR,
                            torch.ones((H,W,CR)).to(self.Device)*1/CR]
            #----------------
            RR     = Phi_G[0].shape[-1]
            if self.CamScanMode == "MiddleToEdge":
                MeasN      = (S+self.ST)//CR + (1 if ((S+self.ST)%CR - 1e-16) >= 0 else 0)
                Result     = torch.zeros((self.MaskShape[0],self.MaskShape[1],int(MeasN)),\
                                          dtype = torch.float32).to(self.Device)
                Target_ST  = self.ST
                Target_END = self.ST + S
                for m in range(int(MeasN)):
                    Phi           = Phi_G[m%2]
                    P_ST          = max(m*CR    ,Target_ST)  - m*CR
                    P_END         = min(m*CR+RR ,Target_END) - m*CR
                    X_ST          = max(m*CR    ,Target_ST)  - Target_ST
                    X_END         = min(m*CR+RR ,Target_END) - Target_ST
                    Result[...,m] = torch.sum(Phi[...,P_ST:P_END]*rx[...,X_ST:X_END],axis = 2)
            elif self.CamScanMode in ["BottomToTop","TopToBottom"]:
                MeasN      = (S+self.ST)//CR + (1 if ((S+self.ST)%CR - 1e-16) >= 0 else 0)
                Result     = torch.zeros((self.MaskShape[0],self.MaskShape[1],int(MeasN)),\
                                          dtype = torch.float32).to(self.Device)
                Target_ST  = self.ST
                Target_END = self.ST + S
                for m in range(int(MeasN)):
                    Phi           = Phi_G[m%2]
                    P_ST          = max(m*CR    ,Target_ST)  - m*CR
                    P_END         = min(m*CR+RR ,Target_END) - m*CR
                    X_ST          = max(m*CR    ,Target_ST)  - Target_ST
                    X_END         = min(m*CR+RR ,Target_END) - Target_ST
                    Result[...,m] = torch.sum(Phi[...,P_ST:P_END]*rx[...,X_ST:X_END],axis = 2)
            #----------------
        if not self.Rb is None:
            Result = F.conv3d(Result.unsqueeze(0).unsqueeze(0),\
                            self.Kb,padding = "same").squeeze(0).squeeze(0)
        if not AddNoise is None:
            Result += AddNoise*torch.randn_like(Result).to(self.Device)*self.CR
        return Result   
    def GetPartOfMask(self,Pos,Size):
        sh,sw      = Size
        rh,rw      = Pos
        MaskPart   = self.Mask[rh:rh+sh,rw:rw+sw,:self.CR].clone()
        MaskPart_s = self.Mask_s[rh:rh+sh,rw:rw+sw].unsqueeze(-1).clone()
        return MaskPart,MaskPart_s

#------------------------------------------------------
class DeConvManager():
    def __init__(self,R=None):
        self.R        = R
        self.Enable   = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(R) == float:
                if R < 1e-3:
                    XYZR        = round(R) + 2 if R > 2   else 3
                    self.Enable = True
                    self.K  = Gausskernel3D((XYZR,XYZR),(R,R))\
                            .unsqueeze(0).unsqueeze(0)
            elif type(R) == tuple or type(R) == list:
                if len(R) == 2:
                    XYR           =  round(R[0]*2) if R[0] > 1.5 else 2
                    ZR            =  round(R[1]*2) if R[1] > 2   else 3
                    self.Enable = True
                    self.K  = Gausskernel3D((XYR,ZR),(R[0],R[1]))\
                            .unsqueeze(0).unsqueeze(0)
                else:
                    XR            =  round(R[0])*2 if R[0] > 1.5 else 2
                    YR            =  round(R[1])*2 if R[1] > 1.5 else 2
                    ZR            =  round(R[2])*2 if R[2] > 1.5 else 2
                    self.Enable = True if max((XR,YR,ZR)) > 1e-3 else False
                    self.K  = Gausskernel3D((XR,YR,ZR),(R[0],R[1],R[2]))\
                             .unsqueeze(0).unsqueeze(0)
    def __call__(self,X,Device=None):
        if self.Enable:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nX           = F.conv3d(X.clone().unsqueeze(1),\
                                self.K.to(X.device),\
                                padding = "same").squeeze(1)
            return nX
        else:
            return X
#------------------------------------------------------

class Nothing():
    def __init__(self):
        pass 
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def MaskLoad(Method="LoadFromFile",Path=None,Size=(256,256),B=1,CR=8):
    if Method == "Generate":
        torch.manual_seed(103409)
        Sizew,Sizeh   = Size
        Sizeh         = Sizeh//B + (1 if Sizeh%B != 0 else 0)
        Sizeh        *= B
        Sizeh         = int(Sizeh)
        Sizew         = Sizew//B + (1 if Sizew%B != 0 else 0)
        Sizew        *= B
        Sizew         = int(Sizew)
        Mask          = np.zeros((Sizeh,Sizew,CR),dtype = np.float32)
        for c in range(CR):
            for i in range(Sizeh//B):
                for j in range(Sizew//B):
                    Value = np.random.random()
                    Mask[i*B:(i+1)*B,j*B:(j+1)*B,c] = Value
        Mask[Mask>0.5] = 1
        Mask[Mask<0.5] = 0
        Mask           = Mask[:Sizeh,:Sizew,:]
    if Method == "LoadFromFile":
        Mask        = np.float32(tifffile.imread(Path))
        Mask        = Mask.transpose(2,1,0).transpose(1,0,2)
        Mask        = (Mask - Mask.min())/(Mask.max() - Mask.min())
    return torch.from_numpy(Mask)


def ResEval(Meas,Reco,CodeCam,AllpassCam,CR):
    H,W,C         = Meas.shape
    Reco          = (Reco - Reco.min())/(Reco.max() - Reco.min())
    #-----------------------------
    if CodeCam and AllpassCam:
        yc            = Meas[...,:(C//2)].clone().view(H,W,C//2,1)\
                                        .repeat(1,1,1,CR)\
                                        .view(H,W,(C//2)*CR)
        yc            = (yc - yc.min())/(yc.max() - yc.min())
        #-----------------------------
        ya            = Meas[...,(C//2):].clone().view(H,W,C//2,1)\
                                        .repeat(1,1,1,CR)\
                                        .view(H,W,(C//2)*CR)
        ya            = (ya - ya.min())/(ya.max() - ya.min())
        #-----------------------------
        H,W,C         = Reco.shape
        Reco          = torch.hstack((yc[:H,:W,:C],ya[:H,:W:C],Reco))
    else:
        y             = Meas.clone().view(H,W,C,1)\
                                        .repeat(1,1,1,CR).view(H,W,C*CR)
        y             = (y - y.min())/(y.max() - y.min())
        H,W,C         = Reco.shape
        Reco          = torch.hstack((y[:H,:W,:C],Reco))
    #-----------------------------
    Reco          = Reco.numpy()
    return Reco

def Merge(yf,yb,cr=7,sr=2):
    H,W,S_yf = yf.shape
    H,W,S_yb = yb.shape
    #-----------------------------------
    yf       = yf.view(H,W,S_yf,1)
    yf       = yf.repeat(1,1,1,cr)
    yf       = yf.view(H,W,S_yf*cr)
    #-----------------------------------
    yb       = yb.view(H,W,S_yb,1)
    yb       = yb.repeat(1,1,1,cr*sr)
    yb       = yb.view(H,W,S_yb*cr*sr)
    return [yf,yb]

