import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import cv2
from tqdm.auto import tqdm
import time
from util    import *
from nn      import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Kernel():
    def __init__(self,\
                T             = 1000,
                device        = "cuda:0"
                ):
        data      = {}
        self.T    = T
        self.device = device
        a         = torch.sqrt(1 - 0.02 * torch.arange(1, T + 1) / T)
        a         = a.to(device)
        b         = torch.sqrt(1 - a**2)
        Ba        = torch.cumprod(a,dim = 0)
        Bb        = torch.sqrt(1 - Ba**2)
        data["a"],data["b"],data["Ba"],data["Bb"] = a,b,Ba,Bb
        
        self.data  = data
        self.shape = (64,64,64)
        self.fp16  = False
        
    def Get_T_iter(self,Step=1):
        T_iter = [x for x in range(1,self.T)]
        T_iter = T_iter[::Step]
        if T_iter[-1] < self.T-1 :
            T_iter.append(self.T-1)
        T_iter = T_iter[::-1]
        T_iter = torch.Tensor(T_iter).long().to(self.device).unsqueeze(-1)
        if self.fp16:
            T_iter = T_iter.to(torch.int16)
        return T_iter,T_iter.shape[0]
           
    def SetShape(self,shape):
        self.shape = shape
        
    def __getitem__(self,i):
        key,index = i
        value     = self.data[key]
        if not isinstance(index,int):
            batch_size = index.shape[0]
        else:
            batch_size = 1
        if self.fp16:
            index = index.to(torch.int64)
        out = value.gather(-1,index)
        out = out.reshape(batch_size, *((1,) * (len(self.shape) - 1)))\
                 .unsqueeze(-1).unsqueeze(-1).to(self.device)
        if self.fp16:
            out = out.to(torch.float16)
        return out

class GuideDiffusion_cls(nn.Module):
    def __init__(self,T=1000,device="cuda:0",shape=(64,64,64),Parallel=False):
        super().__init__()
        self.shape     = shape
        self.Device    = device
        self.T         = T
        self.Coeff     = Kernel(T,device)
        self.Coeff.SetShape(shape)
        self.GuideEnable = False
        self.lamda     = 1
        self.SampGradRequire = False
        self.fpp       = torch.float32
        self.Para      = {}
        self.Para_dim  = {}
        self.Para_cur  = None
        self.Parallel  = Parallel
        self.Silence   = False
        self.n_init    = False
        self.PhaseChange = False

    def InitModelPara(self,dim = 96):
        self.Model = Unet_3D(dim=dim,dim_mults=(1,2,4,8),\
                             Act_initdeformconv = False,\
                             resnet_block_groups=4).to(self.Device)
        self.Model = self.Model.train()

    def RegisterPara(self,ParaPath,dim=64,name="Default"):
        if name == "Default" and "Default" in self.Para.keys():
            assert "RelayModel" not in self.Para.keys(),"Need para name"
            name = "RelayModel"
        self.Para[name]     = torch.load(ParaPath)
        self.Para_dim[name] = dim
    
    def ActivatePara(self,name="Default",b=None):
        self.Model = Unet_3D(dim=self.Para_dim[name],\
                             dim_mults=(1,2,4,8),\
                             resnet_block_groups=4).to(self.Device)
        try:
            self.Model.load_state_dict(self.Para[name])
        except:
            for key in list(self.Para[name].keys()):
                new_key = key[7:]
                self.Para[name][new_key] =  self.Para[name].pop(key)
            for key in list(self.Para[name]._metadata.keys()):
                new_key = key[7:]
                self.Para[name]._metadata[new_key] =  self.Para[name]._metadata.pop(key)
            self.Model.load_state_dict(self.Para[name])
        if self.Parallel:
            self.Model = torch.nn.DataParallel(self.Model)
        self.Model = self.Model.to(self.Device).eval()
        if not b is None:
            pass
    
    def q_sample(self,x0,t,n):
        return self.Coeff["Ba",t]*x0 + self.Coeff["Bb",t]*n
    
    def forward(self,x0,t=None,n=None):
        x0 = x0.unsqueeze(1)
        t     = torch.randint(0,self.T,(x0.shape[0],),\
                              device = self.Device).long()
        n     = torch.randn_like(x0).to(self.Device)
        xt    = self.q_sample(x0,t,n)
        n_est = self.Model(xt,t)
        loss  = torch.sum(torch.abs(n-n_est),dim = (0,1,2,3,4))
        return loss
    
    def SetGuider(self,Guide):
        self.Guide = Guide
        self.GuideEnable = True
        
    def GuiderSample(self,meas,
                     Step=None,Lamda=None,\
                     PhaseChange=None,\
                     Measpara=[(0,0)]):
        ch3  = False
        if Step         == None:Step        = self.Step
        if Lamda        == None:Lamda       = self.Lamda
        if PhaseChange  == None:PhaseChange = self.PhaseChange
        if len(meas.shape) == 3:
            ch3  = True
            meas = meas.unsqueeze(0)
        self.Guide.config(meas,self.T,Lamda,PhaseChange,Measpara)
        self.SampGradRequire = self.Guide.Requir_Gard
        StartTime = time.time()
        res       = self.Sample(Step=Step,meas=meas)
        if ch3:
            res = res.squeeze(0)
        return res,time.time()-StartTime

    def ConfigSolveFnc(self,
                        RelayModel=False,Step=10,Lamda=3,\
                        PhaseChange=0.7,Silence=False,\
                        Gain = 3):
        self.RelayModel    = RelayModel
        self.Step          = Step
        self.Lamda         = Lamda
        self.Silence       = Silence
        self.PhaseChange   = PhaseChange
        self.Gain          = Gain

    def p_sample(self,xt1,t1,t2 = None):
        C            = self.Coeff
        N3     = C["Bb",t2]*(1-1e-6) if t1 < self.T*self.PhaseChange else 0
        N1     = ((C["Bb",t2]**2)-(N3**2)).sqrt()/C["Bb",t1]
        N2     = C["Ba",t2] - N1*C["Ba",t1]
        if N3 > 1e-16:
            z          = self.get_n(xt1.shape)
        #------------------------------
        if self.GuideEnable:
            xt1        = xt1.requires_grad_()
        with torch.no_grad():
            n_est      = self.Model(xt1,t1)
            x0_est     = (xt1 - (n_est*C["Bb",t1]))/(C["Ba",t1])
        #------------------------------
        if self.GuideEnable:
            x0_est     = self.Guide(xt1,x0_est,t1,"front")
        with torch.no_grad():
            xt2        = N1*xt1 + N2*x0_est + (N3*z if N3 > 1e-16 else 0)
        if self.GuideEnable:
            xt2        = self.Guide(xt2,x0_est,t1,lambda x:self.q_sample(x,t1,z),"back")
        #------------------------------
        return xt2.detach(),x0_est.detach()

    def Sample(self,shape=None,meas=None, Step=12):
        #-------------------------
        if shape is None:
            assert self.GuideEnable,"GuideEnable first"
            x     = torch.randn(self.Guide.Proj_shape,\
                                dtype = torch.float32).to(self.Device)
        else:
            x     = torch.randn((1,1,*shape),\
                                dtype = torch.float32).to(self.Device)
        self.n_arr_init()
        #-------------------------
        Tlist,StepLen      = self.Coeff.Get_T_iter(self.T//Step)
        if self.Silence :
            tqdmIter       = range(StepLen-1)
        else:
            tqdmIter       = tqdm(range(StepLen-1),desc="Sample:",leave=False,\
                                unit="Step",total=StepLen-1)
        #-------------------------
        with torch.no_grad() if not self.GuideEnable else Nothing():
            for i in tqdmIter:
                x,x0_est = self.p_sample(x,t1=Tlist[i],t2=Tlist[i+1])
        #------------------------
        x0_est = (x0_est - x0_est.min())/(x0_est.max()-x0_est.min())
        return x0_est.squeeze(1)

    def n_arr_init(self,shape = (256,256,256),max_nt = 100):
        torch.manual_seed(103409)
        self.n_arr   = torch.randn((*shape,max_nt),\
                                   dtype=torch.float16).cpu()
        self.n_count = 0        
        self.n_init  = True
        self.max_nt  = max_nt

    def get_n(self,x_shape):
        H,W,S         = x_shape[-3:]
        if self.n_init == False:self.n_arr_init()
        n             = self.n_arr[:H,:W,:S,self.n_count].clone()
        self.n_count += 1
        return n.to(torch.float32).to(self.Device)

    def Solve(self,x,para):
        x = x.to(self.Device)
        x = (x-x.mean())/x.std()
        x = x*self.Gain 
        return self.GuiderSample(x,Measpara=para)[0].cpu().squeeze(0)

class GuideDiffusionTrainer():
    def __init__(self,
                 Diffusion    = None,\
                 epoches      = None,
                 train_loader = None,
                 val_loader   = None,
                 optimizer    = None,
                 device       = None,
                 T            = 1000,
                 EarlyStop    = False,
                 Adjust_lr    = False,
                 savepath     = None,
                 fp16_flag    = None,
                 Adjust_s     = 0.4):
        self.T            = T
        self.epoches      = epoches
        self.train_loader = train_loader
        self.opti         = optimizer
        self.EarlyStop    = EarlyStop
        self.Adjust_lr    = Adjust_lr
        self.Adjust_s     = Adjust_s
        self.Diffusion    = Diffusion
        self.savepath     = savepath
        self.val_loader   = val_loader
        self.device       = device
        self.fp16_flag    = fp16_flag
        self.state        = "half" if fp16_flag else "float"
            
    def adjust_lr(self):
        if self.epoch%10 == 0 and self.epoch != 0:
            olr = self.opti.param_groups[0]['lr']
            for param_group in self.opti.param_groups:
                param_group['lr'] *= self.Adjust_s 
            nlr = self.opti.param_groups[0]['lr']
            print("[Adjust LR]adjust lr %.1le to %.1le"%(olr,nlr))
            PrintToFile("[Adjust LR]adjust lr %.1le to %.1le"%(olr,nlr))
            
    def SaveModel(self,Titel):
        torch.save(self.Diffusion.Model.state_dict(),\
                   self.savepath+'/'+'BestModel_%s.pth'%Titel)
        print("[Save]Save the model to:"+str(self.savepath))

    def train(self,Title):
        self.Diffusion.Model = self.Diffusion.Model.train()
        Diffusion,opti       = self.Diffusion,self.opti
        for self.epoch in range(self.epoches):
            tqdmIter = tqdm(self.train_loader, 
                            desc  = "Epoch:%d"%(self.epoch),\
                            leave = False,\
                            unit  = "Imgblock",\
                            total=len(self.train_loader))
            meanloss = []
            ST       = time.time()
            for i,x0 in enumerate(tqdmIter):
                x0   = x0.to(self.device)
                loss = Diffusion(x0)
                opti.zero_grad()
                loss.backward()
                opti.step()
                meanloss.append(loss.item())
            meanloss = np.mean(meanloss)
            meanloss /= x0.shape[-3]*x0.shape[-2]*x0.shape[-1]*x0.shape[-4]
            print("[Train]Epoch:%d|Avg pixel loss:%1.7lf|Time:%.1lf"%\
                  (self.epoch,meanloss,time.time()-ST))
            PrintToFile("[Train]Epoch:%d|Avg pixel loss:%1.7lf|Time:%.1lf"%\
                        (self.epoch,meanloss,time.time()-ST))
            self.adjust_lr()
            self.SaveModel(Title)
        return self.Diffusion
      
class GuideDiffusion_Guider_cls():
    def __init__(self,InverseProblemCls):
        self.Requir_Gard = False
        self.I           = InverseProblemCls
        self.device      = self.I.device
    def GradDesc(self,x0_est,t):
        x         = x0_est.clone().squeeze(1)
        x         = nn.Parameter(x,requires_grad=True).to(self.device)
        fnc       = lambda x:torch.sum((self.y-self.I.A(x))**(2))*self.Lamda +torch.sum((x-x0_est)**2)
        D_fnc     = lambda x:torch.autograd.grad(outputs=fnc(x),inputs=x)[0] 
        lr_u,lr_d = 8e-2,8e-5
        lr        = (t/self.T)*(lr_u-lr_d) + lr_d
        st_u,st_d = 1,5e-3
        st        = (t/self.T)*(st_u-st_d) + st_d
        ir_u,ir_d = 1,4.5   
        ir        = (t/self.T)*(ir_u-ir_d) + ir_d
        for i in range(int(ir)):
            g  = D_fnc(x) if i == 0 else (0.3*g+0.7*D_fnc(x))
            x  = x - lr*g
        dx     = x.unsqueeze(1).detach() - x0_est
        x0_est = x0_est + dx*st 
        x0_est = (x0_est - x0_est.mean())/x0_est.std()
        return x0_est

    def config(self,y,T,Lamda=None,\
               PhaseChange=None,Measpara=[(0,0)]):
        self.PhaseChange = PhaseChange
        self.T           = T
        self.y           = y
        self.Proj_shape,self.Lamda =\
            self.I.register_Proj_shape(y.shape,Lamda)
        self.I.Measpara  = Measpara
        self.Requir_Gard = False
    def __call__(self,xt,x0_est,t,q_fnc=None,Pos="front"):
        if Pos == "front":
            if  t < self.T*self.PhaseChange: 
                x0_est = self.GradDesc(x0_est,t)
            return x0_est
        if Pos == "back": 
            if t >=   self.T*self.PhaseChange: 
                xt = self.GradDesc(xt,t)
            return xt
        

