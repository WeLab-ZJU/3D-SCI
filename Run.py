from util               import *
from StitchManager      import Stitch_SCI
import torch
import os
import tifffile
import os
from omegaconf import OmegaConf
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def Run(Configpath):
    Config = OmegaConf.load(Configpath)

    # Path
    DataPath    = Config.Path.DataPath
    OutputPath  = Config.Path.OutputPath
    ModelPath   = Config.Path.ModelPath

    # ForwardModel
    Rolling_only= Config.ForwardModel.Rolling_only_flag
    Rolling_mask= Config.ForwardModel.Rolling_mask_flag
    CR          = Config.ForwardModel.Compressive_ratio
    Noise       = Config.ForwardModel.Noise
    Rc          = (Config.ForwardModel.PSF_rolling_mask_sigma_xy,Config.ForwardModel.PSF_rolling_mask_sigma_z)
    Ra          = (Config.ForwardModel.PSF_rolling_only_sigma_xy,Config.ForwardModel.PSF_rolling_mask_sigma_z)
    Shift       = Config.ForwardModel.Shift
    LineScanTime= Config.ForwardModel.LineScanTime
    MaskHoldTime= Config.ForwardModel.MaskHoldTime
    Org_shape   = Config.ForwardModel.Org_shape

    # Reconstruction
    ModelChannel= Config.Reconstruction.ModelChannel
    Iteration   = Config.Reconstruction.Iteration
    Lambda      = Config.Reconstruction.Lambda
    
    DeviceList    = ["cuda:%d"%d for d in range(Config.Reconstruction.GPUNum)]
    DeviceList    = list(map(torch.device,DeviceList))
    Parallel      = len(DeviceList)

    # Mask and Data
    MaskImg       = MaskLoad("Generate",Size=(Org_shape[0],Org_shape[1]),B=1,CR=CR*2)
    Orig          = LoadImg(DataPath)
    Orig          = Orig[:Org_shape[0],:Org_shape[1],:Org_shape[2]]
    Orig          = (Orig - Orig.mean())/Orig.std()

    # Framework Config
    Solver        = Stitch_SCI(CR         = CR,
                               BlockShape = (Org_shape[0],Org_shape[1],Org_shape[2]//CR),
                               Corp       = (4,4,1),
                               OverLap    = (4,4,1),
                               TempImgPath= "Temp")
    Solver.ConfigDevice(DeviceList)
    For_Fnc,Mes   = Solver.ConfigMethod(MaskImg,ModelPath,
                                        ModelConfig     = (ModelChannel,),
                                        Rc              = Rc,
                                        Ra              = Ra,
                                        CodeCam         = Rolling_mask,
                                        AllpassCam      = Rolling_only,
                                        AllPassShift    = Shift,
                                        Lamda           = Lambda,
                                        Step            = Iteration,
                                        LineScanTime    = LineScanTime,
                                        MaskHoldTime    = MaskHoldTime,
                                        SwitchGroupMask = True)
    # Reconstruction
    Meas            = For_Fnc(Orig,addnoise=Noise)
    Reco,TIME       = Solver.Solve(Meas.clone())
    Res,PSNR,SSIM = Solver.ResEval(Meas,Reco,Orig)

    # Save
    print("%s | P:%d | Mask:%s Only:%s Rc:%s Ra:%s Shift:%d Noise:%1.1le | PSNR:%.2lf SSIM:%.3lf | Time:%.1lf"%\
                     (Mes,Parallel,\
                      Rolling_mask,Rolling_only,\
                      "%.1lf"%Rc if  type(Rc) is float else str(Rc),\
                      "%.1lf"%Ra  if  type(Ra)  is float else str(Ra),\
                      Shift,\
                      Noise if not Noise is None else 0,\
                      PSNR,SSIM,TIME))
    tifffile.imwrite(OutputPath+"/%s_P-%d_Mask-%s_Only-%s_Rc-%s_Ra-%s_Shift-%d_Noise-%2.1le _PSNR-%.2lf_SSIM-%.3lf_Time-%.2lf.tif"%\
                     (Mes,Parallel,\
                      Rolling_mask,Rolling_only,\
                      "%.1lf"%Rc if  type(Rc) is float else str(Rc),\
                      "%.1lf"%Ra  if  type(Ra)  is float else str(Ra),\
                      Shift,\
                      Noise if not Noise is None else 0,\
                      PSNR,SSIM,TIME),Res.transpose(2,0,1))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run image reconstruction with specified config file')
    parser.add_argument('--config', type=str, default='Config.yaml',
                        help='Path to the configuration file (default: Config.yaml)')
    args = parser.parse_args()
    Run(args.config)
