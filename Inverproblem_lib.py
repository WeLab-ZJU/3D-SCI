import torch
import tifffile
import numpy as np
import torch.nn.functional as F
from torch import nn
from util   import *
import random 

def LoadMask(IMGPath,ST = 0,END = -1):
    orig        = np.float32(tifffile.imread(IMGPath))
    orig        = orig.transpose(2,1,0).transpose(1,0,2)
    orig        = (orig - np.min(orig))/(np.max(orig) - np.min(orig))
    if END == -1:
        orig        = orig[...,ST:] #
    else:
        orig        = orig[...,ST:END] #
    orig        = torch.from_numpy(orig)
    return  orig

class InverseProblem_basecls():
    def __init__(self,name,device):
        self.device  = device
        self.name    = name
        self.forward = self.A
    def V(self, x):
        raise NotImplementedError()
    def Vt(self, x):
        raise NotImplementedError()
    def U(self, x):
        raise NotImplementedError()
    def Ut(self, x):
        raise NotImplementedError()
    def Ap(self,x):
        temp = self.Ut(x)
        singulars = self.singulars()
        factors = 1. / singulars
        factors[singulars == 0] = 0.
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.I.Add_zeros(temp))
    def A(self, vec):
        temp      = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    def AT(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.I.Add_zeros(singulars * temp[:, :singulars.shape[0]]))
    def singulars(self):
        raise NotImplementedError()
    def register_Proj_shape(self):
        raise NotImplementedError()
#---------------------------------------   


def NorMask_To_InclineMask(OrigMask,CR,LineScanTime,MaskHoldTime,\
                           Device,CamScanMode = "MiddleToEdge"):
    Reslut = []
    Background = torch.min(OrigMask).item()
    for r in range(2):
        if r == 0:
            Mask                  = OrigMask.clone()
        else:
            TempMask              = torch.zeros_like(OrigMask).to(Device)
            TempMask[:,:,:CR]     = OrigMask[:,:,CR:2*CR] 
            TempMask[:,:,CR:2*CR] = OrigMask[:,:,:CR] 
            Mask                  = TempMask
        #----------
        H,W,_     = Mask.shape
        assert (H%2 == 0),"Line number of mask should be a even number"
        if    CamScanMode == "MiddleToEdge":
            HalfH     = H//2
            ReMaskN   = CR + int((HalfH*LineScanTime)//MaskHoldTime)
            ReMaskN  += 1 if (HalfH*LineScanTime)%MaskHoldTime - 1e-16 >= 0 else 0
            ClineMask = torch.full((H,W,ReMaskN),\
                                   fill_value=Background,\
                                    dtype = torch.float32).to(Device)
            TransStep = MaskHoldTime//LineScanTime
            for s in range(ReMaskN):
                if s < CR:
                    #EQZone
                    Eq_Span    = min((int(TransStep*s),HalfH))
                    ClineMask[HalfH-Eq_Span:HalfH+Eq_Span-1,:,s] =\
                        Mask[HalfH-Eq_Span:HalfH+Eq_Span-1,:,s%(2*CR)]
                    #TransZone
                    Trans_Span = min((int(Eq_Span+TransStep),HalfH))
                    for l in range(Eq_Span,Trans_Span):
                        ClineMask[[HalfH-l,HalfH+l-1],:,s] =\
                            Mask[[HalfH-l,HalfH+l-1],:,s%(2*CR)]*(1-((l - Eq_Span)/TransStep))
                if s >= CR:
                    #ZeroZone
                    Zero_Span   = min((int(TransStep*(s-CR)),HalfH))
                    #TransZone1
                    Trans_Zone1 = min((int(Zero_Span+TransStep),HalfH))
                    for l in range(Zero_Span,Trans_Zone1):
                        ClineMask[[HalfH-l,HalfH+l-1],:,s] =\
                            Mask[[HalfH-l,HalfH+l-1],:,s%(2*CR)]*(((l - Zero_Span)/TransStep))
                    #EQZone
                    Eq_Span   = min((int(TransStep*s),HalfH))
                    ClineMask [HalfH+Trans_Zone1-1:HalfH+Eq_Span,:,s] \
                        = Mask[HalfH+Trans_Zone1-1:HalfH+Eq_Span,:,s%(2*CR)]
                    ClineMask [HalfH-Eq_Span:HalfH-Trans_Zone1+1,:,s] \
                        = Mask[HalfH-Eq_Span:HalfH-Trans_Zone1+1,:,s%(2*CR)]
                    #TransZone2
                    Trans_Span = min((int(TransStep*(s+1)),HalfH))
                    for l in range(Eq_Span,Trans_Span):
                            ClineMask[[HalfH-l,HalfH+l-1],:,s] =\
                                Mask[[HalfH-l,HalfH+l-1],:,s%(2*CR)]*(1-((l - Eq_Span)/TransStep))
                    #ZeroZone2
            Reslut.append(ClineMask)
        elif  CamScanMode in ["BottomToTop","TopToBottom"]:
            if CamScanMode == "TopToBottom":
                Mask  = torch.flip(Mask, dims = (0,)) 
            ReMaskN   = CR + int((H*LineScanTime)//MaskHoldTime)
            ReMaskN  += 1 if (H*LineScanTime)%MaskHoldTime - 1e-16 >= 0 else 0
            ClineMask = torch.full((H,W,ReMaskN),\
                                   fill_value=Background,\
                                   dtype = torch.float32).to(Device)
            TransStep = MaskHoldTime//LineScanTime
            for s in range(ReMaskN):
                if s < CR:
                    #EQZone
                    Eq_Span    = min((int(TransStep*s),H))
                    ClineMask[H-Eq_Span:,:,s] =\
                        Mask[H-Eq_Span:,:,s%(2*CR)]
                    #TransZone
                    Trans_Span = min((int(Eq_Span+TransStep),H))
                    for l in range(Eq_Span,Trans_Span):
                        ClineMask[H-l-1,:,s] =\
                             Mask[H-l-1,:,s%(2*CR)]*(1-((l - Eq_Span)/TransStep))    
                if s >= CR:
                    #ZeroZone
                    Zero_Span   = min((int(TransStep*(s-CR)),H))
                    #TransZone1
                    Trans_Zone1  = min((int(Zero_Span+TransStep),H))
                    for l in range(Zero_Span,Trans_Zone1):
                        ClineMask[H-l-1,:,s] =\
                             Mask[H-l-1,:,s%(2*CR)]*(((l - Zero_Span)/TransStep))
                    #EQZone
                    Eq_Span   = min((int(TransStep*s),H))
                    ClineMask [H-Eq_Span:H-Trans_Zone1,:,s] \
                        = Mask[H-Eq_Span:H-Trans_Zone1,:,s%(2*CR)]
                    #TransZone2
                    Trans_Span2 = min((int(TransStep*(s+1)),H))
                    for l in range(Eq_Span,Trans_Span2):
                            ClineMask[H-l-1,:,s] =\
                                 Mask[H-l-1,:,s%(2*CR)]*(1-((l - Eq_Span)/TransStep))
                    #ZeroZone2
            if CamScanMode == "TopToBottom":
                Reslut.append(torch.flip(ClineMask, dims = (0,)))
            else:
                Reslut.append(ClineMask)
    Reslut = torch.cat((Reslut[0],Reslut[1]),dim=-1).to(Device)
    return Reslut

class SCI_Rolling(InverseProblem_basecls):
    def __init__(self,device,CR,MaskImg,\
                 CodeCam=False,\
                 AllpassCam=False,\
                 Rc=None,Ra=None,\
                 AllPassShift=None,\
                 SwitchGroupMask=False,\
                 SynchronousMode="Aligen",\
                 CamScanMode="MiddleToEdge",\
                 LineScanTime=0,\
                 MaskHoldTime=0,\
                 FractureSurfaceRepair = False,\
                 ):
        self.CR           = CR
        self.Rc           = Rc
        self.device       = device
        self.AllpassCam   = AllpassCam
        self.CodeCam      = CodeCam
        name              = "SCI_Rolling_CR-%d"%CR 
        self.Mask         = MaskImg.to(self.device)
        self.Mask_s       = torch.sum(self.Mask,dim=-1)
        self.Mask_s[self.Mask_s==0] = 1
        self.Measpara     = [(0,0)]
        self.Aug          = False
        self.addnoise     = None
        self.FSR          = FractureSurfaceRepair
        #--------------------------------
        if self.FSR == True:
            self.CR            = 1
            self.AllpassCam    = True 
            self.CodeCam       = False
            Ra                 = None
            self.AllPassShift  = 0
        super().__init__(name,device)
        #--------------------------------
        self.Dc   = DeConvManager(Rc)
        self.Da    = DeConvManager(Ra)
        if self.AllpassCam:
            self.Gain = torch.mean(self.Mask,dim=(0,1,2))
        #--------------------------------
        #for incline mask mode
        self.SwitchGroupEnable = SwitchGroupMask
        self.SynchronousMode   = SynchronousMode
        self.CamScanMode       = CamScanMode
        self.LineScanTime      = LineScanTime
        self.MaskHoldTime      = MaskHoldTime
        self.AllPassShift      = AllPassShift
        #--------------------------------
        self.Mask = NorMask_To_InclineMask(self.Mask,self.CR,\
                                            LineScanTime = LineScanTime,\
                                            MaskHoldTime = MaskHoldTime,\
                                            Device       = self.device,\
                                            CamScanMode  = "MiddleToEdge")

        #--------------------------------
    def register_Proj_shape(self,y_shape,Lamda=None):
        B,H,W,C        = y_shape
        Lamda          = Lamda if not Lamda is None else 1.2
        if self.AllpassCam and self.CodeCam:
            Proj_shape = (B,1,H,W,(C//2)*self.CR)
        else:
            Proj_shape = (B,1,H,W,C*self.CR)
        if self.FSR:
            Proj_shape = (B,1,H,W,(C*self.CR) - 8)
        return Proj_shape,Lamda
        #--------------------------------
    def A(self,x,Measpara=None,Flag = "all",addnoise=None,TempDevice=None):
        #---------------------------------
        if  Flag in ["All","all"]:
            yc_r  = True if self.CodeCam    else False
            ya_r  = True if self.AllpassCam else False
        elif Flag == "Pos":
            yc_r  = True
            ya_r  = False
        elif Flag in ["AllPass","Allpass"]:
            yc_r  = False
            ya_r  = True
        #---------------------------------
        if not Measpara is None :
            self.Measpara = Measpara
        if len(x.shape) == 3:x = x.unsqueeze(0)
        #---------------------------------
        if not addnoise is None :
            self.addnoise = addnoise
        #---------------------------------
        if not TempDevice is None:
            self.Old_device = self.device
            self.device     = TempDevice   
        #---------------------------------  
        x_device    = x.device
        if x_device != self.device:x = x.to(self.device)
        #---------------------------------  
        M           = self.Mask
        HH,HW       = M.shape[:2]
        RR          = M.shape[-1]//2
        CR          = self.CR
        B,H,W,C     = x.shape
        #---------------------------------  
        assert H<=HH and W<=HW,"Mask shape error"
        assert len(self.Measpara) == B,"Need len(self.Measpara) == B"
        #---------------------------------  
        if self.SwitchGroupEnable:
            def SwitchGroupMask_A(x,Measpara,OM_G):
                M_G   = []
                H,W,C = x.shape[:3] 
                x     = x.unsqueeze(0) 
                for i in range(2):
                    M_Glist     = []
                    for paraitem in Measpara:
                        ph,pw   = paraitem
                        M_Glist.append(OM_G[i][ph:ph+H,pw:pw+W,:].unsqueeze(0).clone())
                    M_G.append(torch.cat(M_Glist,dim=0))
                RR     = M_G[0].shape[-1]//2
                M_G    = [torch.cat((M_G[0],M_G[1]),dim = -1),\
                          torch.cat((M_G[1],M_G[0]),dim = -1)]                  
                MeasN  = (C+self.ST)//CR + (1 if ((C+self.ST)%CR - 1e-16) >= 0 else 0)
                Result = torch.zeros((1,H,W,int(MeasN)),dtype = torch.float32).to(self.device)
                Target_ST  = self.ST
                Target_END = self.ST + C
                for m in range(int(MeasN)):
                    Phi           = M_G[m%2]
                    P_ST          = max(m*CR    ,Target_ST)  - m*CR
                    P_END         = min(m*CR+RR ,Target_END) - m*CR
                    X_ST          = max(m*CR    ,Target_ST)  - Target_ST
                    X_END         = min(m*CR+RR ,Target_END) - Target_ST
                    Result[...,m] = torch.sum(Phi[...,P_ST:P_END]*x[...,X_ST:X_END],axis = -1)
                return Result
            #-------------------------------------------------
            self.ST               = 0
            #-------------------------------------------------
            if yc_r:
                rx         = self.Dc(x,Device=self.device).squeeze(0) 
                RR         = self.Mask.shape[-1]//2
                OM_G       = [self.Mask[...,:RR],self.Mask[...,RR:]]
                yc         = SwitchGroupMask_A(rx.clone(),self.Measpara,OM_G)
                if not self.addnoise is None:
                    noise  = self.CR*self.addnoise*torch.randn_like(yc).to(self.device)
                    yc    += noise
            #-------------------------------------------------
            if  self.AllpassCam and ya_r:
                Step       = C//CR
                rx         = self.Da(x,Device=self.device).squeeze(0) 
                if self.AllPassShift is None or self.AllPassShift == 0:
                    ya         = torch.zeros((B,H,W,Step),dtype = torch.float32).to(self.device)
                    RR         = self.Mask.shape[-1]//2
                    OM_G        = [torch.full_like(self.Mask[...,:RR],fill_value=self.Gain),\
                                   torch.full_like(self.Mask[...,:RR],fill_value=self.Gain)]
                    ya         = SwitchGroupMask_A(rx.clone(),self.Measpara,OM_G)
                    if not self.addnoise is None:
                        noise    = self.CR*self.addnoise*torch.randn_like(ya).to(self.device)
                        ya      += noise          
                else:
                    APS = self.AllPassShift
                    ya          = torch.zeros((B,H,W,Step),dtype = torch.float32).to(self.device)
                    RR          = self.Mask.shape[-1]//2
                    OM_G        = [torch.full_like(self.Mask[...,:RR],fill_value=self.Gain),\
                                   torch.full_like(self.Mask[...,:RR],fill_value=self.Gain)]

                    ya[...,:-1] = SwitchGroupMask_A(rx[...,APS:-(CR-APS)].clone(),self.Measpara,OM_G)

                    if not self.addnoise is None:
                        noise    = self.CR*self.addnoise*torch.randn_like(ya).to(self.device)
                        ya      += noise  
                    ya[...,-1]   = 0   
            #-------------------------------------------------      
        else:
            Mlist       = []
            for paraitem in self.Measpara:
                ph,pw   = paraitem
                Mlist.append(M[ph:ph+H,pw:pw+W,:].unsqueeze(0))
            M           = torch.cat(Mlist,dim=0)                       
            Step        = C//CR
            #-------------------------------------------------
            if   yc_r:
                #B,H,W,C
                rx         = self.Dc(x,Device=self.device).squeeze(0)
                yc         = torch.zeros((B,H,W,Step),dtype = torch.float32).to(self.device)
                for s in range(Step):
                    yc[...,s] = torch.sum(M*rx[...,s*CR:(s+1)*CR].clone(),axis = -1)
                if not self.addnoise is None:
                    noise  = self.CR*self.addnoise*torch.randn_like(yc).to(self.device)
                    yc    += noise
            #-------------------------------------------------
            if   self.AllpassCam and ya_r:
                rx         = self.Da(x,Device=self.device).squeeze(0)
                if self.AllPassShift is None or self.AllPassShift == 0:
                    ya  = torch.zeros((B,H,W,Step),dtype = torch.float32).to(self.device)
                    for s in range(Step):
                        ya[...,s]   = torch.sum(rx[...,s*CR:(s+1)*CR].clone(),axis = -1)*self.Gain
                    if not self.addnoise is None:
                        noise       = self.CR*self.addnoise*torch.randn_like(ya).to(self.device)
                        ya         += noise
                else:
                    APS = self.AllPassShift
                    ya  = torch.zeros((B,H,W,Step),dtype = torch.float32).to(self.device)
                    for s in range(Step-1):
                        ya[...,s]   = torch.sum(rx[...,APS+(s*CR):APS+((s+1)*CR)],axis = -1)*self.Gain
                    if not self.addnoise is None:
                        noise       = self.CR*self.addnoise*torch.randn((B,H,W,Step-1),\
                                        dtype = torch.float32).to(self.device)
                        ya[...,:-1]+= noise
        #---------------------------------
        if   yc_r and ya_r:
            y = torch.cat((yc,ya),axis = -1)
        elif yc_r and (not ya_r):
            y = yc
        elif ya_r and (not yc_r):
            y = ya
        #---------------------------------
        if self.FSR:
            temp_y = torch.zeros(B,H,W,C-8).to(self.device)
            temp_y[...,:(C//2)-4] = y[...,:(C//2)-4]
            temp_y[...,(C//2)-4:] = y[...,(C//2)+4:]
            del y
            y  = temp_y
        #---------------------------------
        if x.device != x_device:y = y.to(x_device)
        if not TempDevice is None:self.device = self.Old_device
        return y       
    def GetPartOfMask(self,Pos,Size):
        assert not self.SwitchGroupEnable,"No incline mask mode"
        sh,sw      = Size
        rh,rw      = Pos
        MaskPart   = self.Mask[rh:rh+sh,rw:rw+sw,:self.CR].clone()
        MaskPart_s = self.Mask_s[rh:rh+sh,rw:rw+sw].unsqueeze(-1).clone()
        return MaskPart,MaskPart_s
    def RandomAug(self):
        self.Aug   = True
        if random.random() > 0.5:
            self.addnoise = random.random()*2e-1
