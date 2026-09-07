import math
import os
import time 
import cv2
import torch
import warnings
import tifffile
import numpy as np
from util      import *
from torch import multiprocessing as tmp
from tqdm.auto import tqdm

from Inverproblem_lib import SCI_Rolling as SCI_invprob
from GuideDiffusion_Class import GuideDiffusion_cls,GuideDiffusion_Guider_cls

from copy import deepcopy
from ImgEval import PSNR,SSIM

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def AgentFnc(para):
    Tid,BlockArr,AdjPara,Fnc,TargetPath = para
    time.sleep(Tid*10)
    Saver  = SaveManager()
    SaverQ = Saver.GetQueue()
    Saver.start()
    for n in range(len(BlockArr)):
        Block = Fnc(BlockArr[n],[AdjPara[n],])
        Mes   = (Block.cpu().numpy(),TargetPath+"/%d-%d.tif"%(Tid,n))
        SaverQ.put(Mes,block=False)
    SaverQ.put(None,block=False)
    Saver.join()
    return None

class Stitch_Base():
    def __init__(self,BlockShape,TargetShape,\
                 Corp=(0,0,0),OverLap=(4,4,4),\
                 TempImgPath = None):
        self.BlockShape  = BlockShape
        self.TargetShape = TargetShape
        self.Corp        = Corp
        self.OverLap     = OverLap
        self.Parallel    = 1
        self.TempImgPath = TempImgPath
        try:os.mkdir(TempImgPath)
        except:pass
    def GetPos(self,C,S,D):
        ch,cw,cs    = C
        sh,sw,ss    = S
        dh,dw,ds    = D
        rh,rw,rs    = 0,0,0
        rh          = ch*(sh - (dh if ch!=0 else 0))
        rw          = cw*(sw - (dw if cw!=0 else 0))
        rs          = cs*(ss - (ds if cs!=0 else 0))
        return (rh,rw,rs)
    
    def Adjust(self,x,P):
        x = x + 10.0
        h,w,s       = P
        H,W,S       = self.Size
        CH,CW,CS    = self.Corp 
        LH,LW,LS    = self.OverLap 
        a,b,c       = self.TarShape 
        trh,trw,trs = self.TarRate

        if h != 0  :
            if CH  != 0:x = x[CH*trh:,:,:]
        if h != H-1:
            if CH  != 0:x = x[:-CH*trh,:,:]
        if w != 0  :
            if CW  != 0:x = x[:,CW*trw:,:]
        if w != W-1:
            if CW  != 0:x = x[:,:-CW*trw,:]
        if s != 0  :
            if CS  != 0:x = x[:,:,CS*trs:]
        if s != S-1:
            if CS  != 0:x = x[:,:,:-CS*trs]
        
        if h != 0  :
            if LH*trh > 1:
                for i in range(LH*trh):x[i   ,:,:] *= i/(LH*trh-1)
        if h != H-1:
            if LH*trh > 1:
                for i in range(LH*trh):x[-i-1,:,:] *= i/(LH*trh-1)
        if w != 0  :
            if LW*trw > 1:
                for i in range(LW*trw):x[:,i   ,:] *= i/(LW*trw-1)
        if w != W-1:
            if LW*trw > 1:
                for i in range(LW*trw):x[:,-i-1,:] *= i/(LW*trw-1)
        if s != 0  :
            if LS*trs > 1:
                for i in range(LS*trs):x[:,:,   i] *= i/(LS*trs-1)
        if s != S-1:
            if LS*trs > 1:
                for i in range(LS*trs):x[:,:,-i-1] *= i/(LS*trs-1)
    
        rh          = (h*(a - ((LH+2*CH)*trh)) + CH*trh) if h!=0 else 0
        rw          = (w*(b - ((LW+2*CW)*trw)) + CW*trw) if w!=0 else 0
        rs          = (s*(c - ((LS+2*CS)*trs)) + CS*trs) if s!=0 else 0

        NPOS        = (rh,rw,rs)
        
        tifffile.imwrite(self.TempImgPath+'/%d_%d_%d.tif'%(h,w,s),x.cpu().numpy())
        return x,NPOS

    def Solve__(self,Imglist,registering = 10,):
        TarRate      = self.TarRate
        OrigPosSeqlen = len(Imglist)
        shape_o      = Imglist[0].shape
        self.shape   = Imglist[0].shape
        #----------------------------
        OH,OW,OS     = shape_o          #Original
        CH,CW,CS     = self.Corp        #
        LH,LW,LS     = self.OverLap 
        BH,BW,BS     = self.BlockShape
        TRH,TRW,TRS  = TarRate          #1 , 1 , cr
        TH,TW,TS     = int(BH*TRH),int(BW*TRW),int(BS*TRS)
        self.TarShape = (TH,TW,TS)
        Pad          = (2*CH+LH,2*CW+LW,2*CS+LS)         
        #----------------------------
        S_H          = (OH-(CH*2)-LH)//(BH-(CH*2)-LH)     #COUNT
        S_W          = (OW-(CW*2)-LW)//(BW-(CW*2)-LW)
        S_S          = (OS-(CS*2)-LS)//(BS-(CS*2)-LS)
        self.Size    = (S_H,S_W,S_S)
        RH,RW,RS     = (S_H*(TH-(LH+CH*2)*TRH)+(LH+2*CH)*TRH,\
                        S_W*(TW-(LW+CW*2)*TRW)+(LW+2*CW)*TRW,\
                        S_S*(TS-(LS+CS*2)*TRS)+(LS+2*CS)*TRS)    # RESULT SHAPE
        #----------------------------
        OrigPosSeq = []
        for s_s in range(S_S):
            for s_w in range(S_W):
                for s_h in range(S_H):
                    OrigPosSeq.append((s_h,s_w,s_s))
        BlockNum     = len(OrigPosSeq)
        SubSeqLen    = math.ceil(BlockNum/self.Parallel)
        PosSeqlist   = [OrigPosSeq[i*SubSeqLen:(i+1)*SubSeqLen] \
                                    for i in range(self.Parallel-1)]
        PosSeqlist.append(OrigPosSeq[(self.Parallel-1)*SubSeqLen:] )
        BlockSeqlist  = []
        AdjPosSeqlist = []
        StateSeqlist  = []
        for PosSeq in PosSeqlist:
            BlockSeq  = []
            AdjPosSeq = []
            StateSeq  = []
            for Pos in PosSeq:
                H,W,S   = self.GetPos(Pos,self.BlockShape,Pad)
                BlockItem = []
                for b in range(OrigPosSeqlen):
                    if b == 1 and registering !=0:
                        block0 = (Imglist[0][H:H+BH,W:W+BW,S:S+BS]).cpu().numpy()
                        block1 = np.zeros((BH+registering*2,BW+registering*2,BS)).astype(np.float32)
                        block1[registering:BH+registering,registering:BW+registering,:] = (Imglist[1][H:H+BH,W:W+BW,S:S+BS]).cpu().numpy()
                        result = cv2.matchTemplate(np.max(block1,axis=2), np.max(block0,axis=2), cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        bottom_right = (max_loc[0] + BW, max_loc[1] + BH)
                        matched_region = block1[max_loc[1]:bottom_right[1], max_loc[0]:bottom_right[0]]
                        BlockItem.append(torch.from_numpy(matched_region).clone().unsqueeze(0)) 

                    else:
                        Block   = Imglist[b][H:H+BH,W:W+BW,S:S+BS]

                        BlockItem.append(Block.clone().unsqueeze(0)) 
                AdjPosSeq.append((H,W))
                StateSeq .append((Block.mean().item(),Block.std().item()))
                BlockSeq .append(torch.cat(BlockItem,dim=-1))
            BlockSeqlist .append(deepcopy(BlockSeq))
            AdjPosSeqlist.append(deepcopy(AdjPosSeq))
            StateSeqlist .append(deepcopy(StateSeq))
        #----------------------------
        Fnc_list = []
        for i in range(0,self.Parallel):
            SilenceFlag = True if i != 0 else False
            Fnc_list.append(self.Inv_Fnc_Creater(self.DeviceList[i],SilenceFlag).Solve)          
        #----------------------------
        print("=======================================================")
        print("[Device list]",self.DeviceList)
        print("[Recon Size ] H:%d W:%d S:%d"%(S_H*BH*TRH,S_W*BW*TRW,S_S*BS*TRS))
        print("[Block Size ] H:%d W:%d S:%d"%(BH*TRH,BW*TRW,BS*TRS))
        print("[Parallel:%d] Main process block seq length:%d"%\
                            (self.Parallel,len(BlockSeqlist[0])))
        if self.Parallel > 1:
            print(">>>>>Subprocess block seq length:",\
                  [len(PosSeqlist[x]) for x in range(1,self.Parallel)])
        print("=======================================================")
        #----------------------------
        Recinfo = {"shape"       :(RH,RW,RS),\
                   "PosSeqlist"  :PosSeqlist,\
                   "StateSeqlist":StateSeqlist,\
                   "Parallel"    :self.Parallel,\
                   "size"        :self.Size,\
                   "TarShape"    :self.TarShape}
        saveobj(Recinfo,self.TempImgPath+"/Recinfo.rec")
        Saver  = SaveManager()
        SaverQ = Saver.GetQueue()
        Saver.start()
        #----------------------------
        ST         = time.time()
        if self.Parallel > 1:
            print("Creating Subprocess.")
            Pool   = tmp.Pool(processes=self.Parallel-1)
            MP_res = Pool.map_async(AgentFnc,[(i,BlockSeqlist[i],
                                               AdjPosSeqlist[i],
                                               Fnc_list[i],
                                               self.TempImgPath) \
                                              for i in range(1,self.Parallel)])
        print("Runing main process.")
        #----------------------------
        tqdmSeq = tqdm(range(len(BlockSeqlist[0])),\
                         desc="Reconstruct:",\
                         unit="Block",total=len(BlockSeqlist[0]))
        for n in tqdmSeq:
            Block              = BlockSeqlist[0][n]
            AdjPos             = AdjPosSeqlist[0][n]
            NewBlock           = Fnc_list[0](Block,[AdjPos,])
            Mes = (NewBlock.cpu().numpy(),self.TempImgPath+"/%d-%d.tif"%(0,n))
            SaverQ.put(Mes,block=False)
        SaverQ.put(None,block=True)
        Saver.join()
        #----------------------------
        if self.Parallel > 1:
            MP_res.wait()
            MP_res.get()
            Pool.close()
            Pool.join()
            del Pool
        print("Done! Time:%.3lfs"%(time.time()-ST))
        del Fnc_list
    def Merge(self):
        torch.cuda.empty_cache()
        print("=======================================================")
        #----------------------------
        Recinfo      = loadobj(self.TempImgPath+"/Recinfo.rec")
        shape        = Recinfo["shape"]
        PosSeqlist   = Recinfo["PosSeqlist"]
        StateSeqlist = Recinfo["StateSeqlist"]
        Parallel     = Recinfo["Parallel"]
        self.Size    = Recinfo["size"]
        self.TarShape= Recinfo["TarShape"]
        RH,RW,RS     = shape
        #----------------------------
        print("Merge and reduce results")
        ST        = time.time()
        Res       = torch.zeros((RH,RW,RS))
        for i in range(Parallel):
            SeqLen = len(PosSeqlist[i])
            for j in range(SeqLen):
                Pos        = PosSeqlist[i][j]
                Block      = loadblock(self.TempImgPath+"/%d-%d.tif"%(i,j))
                if not Block is None:
                    Mean,Std   = StateSeqlist[i][j]
                    Block     -= Block.mean()
                    Block     /= Block.std()
                    Block     *= Std
                    Block     += Mean
                    Block,NPos = self.Adjust(Block,Pos)
                    h,w,s      = NPos
                    H,W,S      = Block.shape
                    print("h:%d w:%d s:%d, H:%d W:%d S:%d"%(h,w,s,H,W,S))
                    
                    Res[h:h+H,w:w+W,s:s+S] += Block.clone()
        print("Done! Time:%.3lfs"%(time.time()-ST))
        print("=======================================================")
        #----------------------------
        return Res
    def Solve(self,Img):
        START_Time        = time.time()
        self.Solve__([Img])
        return self.Merge(),time.time()-START_Time
    def ConfigDevice(self,DeviceList):
        self.MainDevice = DeviceList[0]
        self.Parallel   = len(DeviceList)
        self.DeviceList = DeviceList
        if self.Parallel > 1:
            tmp.set_start_method("spawn",force=True)

class Stitch_SCI(Stitch_Base):
    def __init__(self,
                 CR             =  8,\
                 BlockShape     = (72,72,96),\
                 Corp           = (0,0,0),\
                 OverLap        = (4,4,0),\
                 TempImgPath    = None):
        
        self.BlockShape   = BlockShape
        H,W,S             = BlockShape
        self.TargetShape  = (H,W,S*CR)
        self.TarRate      = (1,1,CR)
        self.CR           = CR

        super().__init__(BlockShape  = BlockShape,\
                         TargetShape = self.TargetShape,\
                         Corp        = Corp,\
                         OverLap     = OverLap,
                         TempImgPath = TempImgPath)
        
    def ConfigMethod(self,MaskImg,\
                     ModelParaPath=None,\
                     ModelConfig=(64,),\
                     GenerateGuideMethod="GradDesc",\
                     Rc=None,Ra=None,\
                     CodeCam=True,\
                     AllpassCam=False,\
                     AllPassShift=None,\
                     Lamda          = 2.8,\
                     SwitchGroupMask= False,\
                     FSR            = False,\
                     FreeU_B        = None,\
                     PhaseChange    = 0.7,\
                     LineScanTime   = 10/1024,\
                     MaskHoldTime   = 3.381,\
                     Step           = 10,\
                    ):
        self.AllpassCam   = AllpassCam
        self.CodeCam      = CodeCam
        self.AllPassShift = AllPassShift
        self.Inv_Fnc_list = {}
        self.FSR          = FSR
        #----------------------------
        print("=======================================================")
        print("-------------------------------------------------------")
        if CodeCam   :print("Rc:",Rc)
        if AllpassCam:print("Ra :",Ra)
        print("Rolling_mask_coded :",CodeCam)
        print("Rolling_only_coded :",AllpassCam)
        if AllpassCam:print("Shift :",AllPassShift)
        #----------------------------  
        def For_Fnc_Base(InvPRcbCls_MainDevice,x,TempDevice,addnoise=None):
            res        = InvPRcbCls_MainDevice.forward(x,addnoise=addnoise,\
                                                        TempDevice=TempDevice).squeeze(0)
            return res
        def Inv_Fnc_Creater_Base(Current_DEVICE,Silence,CR,\
                                    ModelParaPath,MaskImg):
            T              = 1000
            GuideDiffusion = GuideDiffusion_cls(T=T,device=Current_DEVICE,Parallel=False)
            InvPRcbCls     = SCI_invprob(device=Current_DEVICE,\
                                    CR=CR,MaskImg=MaskImg,\
                                    CodeCam=CodeCam,\
                                    AllpassCam=AllpassCam,\
                                    Rc=Rc,Ra=Ra,\
                                    AllPassShift=AllPassShift,\
                                    SwitchGroupMask=SwitchGroupMask,\
                                    FractureSurfaceRepair=FSR,\
                                    LineScanTime=LineScanTime,\
                                    MaskHoldTime=MaskHoldTime)
            GuideDiffusion.SetGuider(GuideDiffusion_Guider_cls(InverseProblemCls = InvPRcbCls))
            GuideDiffusion.RegisterPara(ModelParaPath,dim=ModelConfig[0])
            GuideDiffusion.ActivatePara(b=FreeU_B)
            GuideDiffusion.ConfigSolveFnc(Step=Step,Lamda=Lamda,\
                                          PhaseChange=PhaseChange,\
                                          Silence=Silence)
            return GuideDiffusion

        InvPRcbCls_MainDevice = SCI_invprob(CR=self.CR,MaskImg=MaskImg,\
                                        device=self.MainDevice,\
                                        CodeCam=CodeCam,\
                                        AllpassCam=AllpassCam,\
                                        Rc=Rc,Ra=Ra,\
                                        AllPassShift=AllPassShift,\
                                        SwitchGroupMask=SwitchGroupMask,\
                                        FractureSurfaceRepair=FSR,\
                                        LineScanTime=LineScanTime,\
                                        MaskHoldTime=MaskHoldTime)
        
        For_Fnc              = lambda x,addnoise=False,TempDevice=None:\
                                                    For_Fnc_Base(InvPRcbCls_MainDevice,\
                                                                    x,TempDevice,addnoise)
        self.Inv_Fnc_Creater = lambda input_DEVICE,Silence=False:Inv_Fnc_Creater_Base(\
                                                                input_DEVICE,\
                                                                Silence,\
                                                                self.CR,\
                                                                ModelParaPath,\
                                                                MaskImg)
        Message              = "GuideDiffusion_%s_%s"%(GenerateGuideMethod,ModelConfig[0])

            
        return For_Fnc,Message
                    
    def Solve(self,Img):
        START_Time        = time.time()
        print("-------------------------------------------------------")
        print("Measurement info:")
        if self.AllpassCam and self.CodeCam:
            _,_,C  = Img.shape
            MeasP  = Img[...,:C//2]
            MeasA  = Img[...,C//2:]
            print("Shape:",MeasP.shape)
            print("MaskMeas|Mean:%1.3le"%MeasP.mean().item(),"|Var:%2.4lf"%MeasP.std().item())
            print("OnlyMeas|Mean:%1.3le"%MeasA.mean().item(),"|Var:%2.4lf"%MeasA.std().item())
            self.Solve__([MeasP,MeasA])
            Res    = self.Merge()
        else:
            print("Shape:",Img.shape)
            print("Meas   |Mean:%1.3le"%Img.mean().item(),"|Var:%2.4lf"%Img.std().item())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.Solve__([Img])
                Res = self.Merge()
        return Res,time.time()-START_Time

    def ResEval(self,y,r,o=None,bigflag=False):
        H,W,C    = y.shape
        RH,RW,RC = r.shape
        if not o is None:
            o        = o[:RH,:RW,:RC]
        APS      = self.AllPassShift
        if self.AllpassCam and self.CodeCam:
            if self.AllPassShift is None or self.AllPassShift == 0:
                ya    = y[:,:,C//2:]
                ya    = ya.view(H,W,C//2,1).repeat(1,1,1,self.CR)\
                        .view(H,W,(C//2)*self.CR)
                ya    = ya[:RH,:RW,:RC]
            else:
                ya    = torch.zeros((H,W,(C//2)*self.CR))
                sya   = y[...,C//2:-1].view(H,W,(C//2)-1,1)\
                                    .repeat(1,1,1,self.CR)\
                                    .view(H,W,((C//2)-1)*self.CR)
                ya[...,APS:APS+((C//2)-1)*self.CR] = sya
                ya    = ya[:RH,:RW,:RC]
            yc    = y[:,:,:C//2]
            yc    = yc.view(H,W,C//2,1).repeat(1,1,1,self.CR)\
                    .view(H,W,(C//2)*self.CR)
            yc    = yc[:RH,:RW,:RC]
            y     = torch.hstack((ya,yc))
        if not self.CodeCam:
            if self.AllPassShift is None or self.AllPassShift == 0:
                ya    = y.view(H,W,C,1).repeat(1,1,1,self.CR)\
                        .view(H,W,C*self.CR)
            else:
                ya    = torch.zeros((H,W,C*self.CR))
                sya   = y.view(H,W,C,1).repeat(1,1,1,self.CR)\
                          .view(H,W,(C)*self.CR)
                ya[...,:(C)*self.CR] = sya
            y     = ya[:RH,:RW,:RC]
        if not self.AllpassCam:
            y     = y.view(H,W,C,1).repeat(1,1,1,self.CR)\
                     .view(H,W,C*self.CR)
            y     = y[:RH,:RW,:RC]
        if not o is None:
            o     = (o - o.min())/(o.max()-o.min())
        y     = (y - y.min())/(y.max()-y.min())
        r     = (r - r.min())/(r.max()-r.min())
        if not o is None:
            Res   = torch.hstack((o,y,r)).numpy()
            psnr_ = PSNR(o,r)
            ssim_ = SSIM(o,r)
        else:
            if bigflag == True:
                Res = ((r*65535).numpy()).astype(np.uint16)
            else:
                Res   = torch.hstack((y,r)).numpy()
        if not o is None:
            return Res,psnr_,ssim_
        else:
            return Res