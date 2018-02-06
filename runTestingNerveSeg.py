# from __future__ import print_function
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
from utils import *
from ganComponents import *
from nnBuildUnits import CrossEntropy2d
from nnBuildUnits import computeSampleAttentionWeight
from nnBuildUnits import adjust_learning_rate
import time
from Unet2d_pytorch import UNet
from Unet3d_pytorch import UNet3D

parser = argparse.ArgumentParser(description="PyTorch InfantSeg")

parser.add_argument("--isSegReg", action="store_true", help="is Seg and Reg?", default=False)
parser.add_argument("--isDiceLoss", action="store_true", help="is Dice Loss used?", default=True)
parser.add_argument("--isSoftmaxLoss", action="store_true", help="is Softmax Loss used?", default=True)
parser.add_argument("--isContourLoss", action="store_true", help="is Contour Loss used?", default=True)
parser.add_argument("--isResidualEnhancement", action="store_true", help="is residual learning operation enhanced?", default=False)
parser.add_argument("--isViewExpansion", action="store_true", help="is view expanded?", default=True)
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=True)
parser.add_argument("--isSpatialDropOut", action="store_true", help="is spatial dropout used?", default=False)
parser.add_argument("--isFocalLoss", action="store_true", help="is focal loss used?", default=False)
parser.add_argument("--isSampleImportanceFromAd", action="store_true", help="is sample importance from adversarial network used?", default=False)
parser.add_argument("--dropoutRate", type=float, default=0.25, help="Spatial Dropout Rate. Default=0.25")
parser.add_argument("--lambdaAD", type=float, default=0, help="loss coefficient for AD loss. Default=0")
parser.add_argument("--adImportance", type=float, default=0, help="Sample importance from AD network. Default=0")

parser.add_argument("--modelPath", default="/home/dongnie/Desktop/myPyTorch/pytorch-SRResNet/Segmentor_wce_wdice_contour_lrdcr_1127_200000.pt", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="preSub_wce_wdice_contour_lrdcr_1127_", type=str, help="prefix of the to-be-saved predicted filename")
parser.add_argument("--how2normalize", type=int, default=3, help="how to normalize the data")


def main():
    opt = parser.parse_args()
    print opt
    
    path_test = '/home/dongnie/warehouse/nerveSeg/data'
    
   
   
    netG = UNet3D(in_channel=1, n_classes=2)
     #netG.apply(weights_init)
    netG.cuda()
    netG.load_state_dict(torch.load(opt.modelPath))
    
    
    ids = [44,45,47]
    for ind in ids:
        start = time.time()
        mr_test_itk=sitk.ReadImage(os.path.join(path_test,'NB_P%03d.nii.gz'%ind))
        ct_test_itk=sitk.ReadImage(os.path.join(path_test,'NB_P%03d_FN.nii.gz'%ind))
        
        mrnp=sitk.GetArrayFromImage(mr_test_itk)
        mu=np.mean(mrnp)
    
        ctnp=sitk.GetArrayFromImage(ct_test_itk)
        
        #for training data in pelvicSeg
        if opt.how2normalize == 1:
            maxV, minV=np.percentile(mrnp, [99 ,1])
            print 'maxV,',maxV,' minV, ',minV
            mrnp=(mrnp-mu)/(maxV-minV)
            print 'unique value: ',np.unique(ctnp)
    
        #for training data in pelvicSeg
        elif opt.how2normalize == 2:
            maxV, minV = np.percentile(mrnp, [99 ,1])
            print 'maxV,',maxV,' minV, ',minV
            mrnp = (mrnp-mu)/(maxV-minV)
            print 'unique value: ',np.unique(ctnp)
        
        #for training data in pelvicSegRegH5
        elif opt.how2normalize== 3:
            std = np.std(mrnp)
            mrnp = (mrnp - mu)/std
            print 'maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp)
            
        elif opt.how2normalize== 4:
            maxV, minV = np.percentile(mrnp, [99.2 ,1])
            print 'maxV is: ',np.ndarray.max(mrnp)
            mrnp[np.where(mrnp>maxV)] = maxV
            print 'maxV is: ',np.ndarray.max(mrnp)
            mu=np.mean(mrnp)
            std = np.std(mrnp)
            mrimg = (mrnp - mu)/std
            print 'maxV,',np.ndarray.max(mrimg),' minV, ',np.ndarray.min(mrimg)
    
    #             full image version with average over the overlapping regions
    #             ct_estimated = testOneSubject(mrnp,ctnp,[3,168,112],[1,168,112],[1,8,8],netG,'Segmentor_model_%d.pt'%iter)
        
        # the attention regions
#         x1=80
#         x2=192
#         y1=35
#         y2=235
#         matFA = mrnp[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
#         matGT = ctnp[:,y1:y2,x1:x2]
    #                 volFA = sitk.GetImageFromArray(matFA)
    #                 sitk.WriteImage(volFA,'volFA'+'.nii.gz')
    #                 volGT = sitk.GetImageFromArray(matGT)
    #                 sitk.WriteImage(volGT,'volGT'+'.nii.gz')
        matFA = mrnp
        matGT = ctnp
        matOut,_ = testOneSubject(matFA,matGT,2,[16,64,64],[16,64,64],[8,20,20],netG,opt.modelPath)
        ct_estimated = np.zeros([ctnp.shape[0],ctnp.shape[1],ctnp.shape[2]])
#         print 'matOut shape: ',matOut.shape
#         ct_estimated[:,y1:y2,x1:x2] = matOut
    
        ct_estimated = np.rint(ct_estimated) 
        diceBladder = dice(ct_estimated,ctnp,1)

        
#         print 'pred: ',ct_estimated.dtype, ' shape: ',ct_estimated.shape
#         print 'gt: ',ctnp.dtype,' shape: ',ctnp.shape
        print 'sub%d'%ind,'dice1 = ',diceBladder
        volout = sitk.GetImageFromArray(ct_estimated)
        sitk.WriteImage(volout,opt.prefixPredictedFN+'sub{}'.format(ind)+'.nii.gz')
        print 'cost time is %.2f'%(time.time()-start)
        volgt = sitk.GetImageFromArray(ctnp)
        sitk.WriteImage(volgt,'gt_sub{}'.format(ind)+'.nii.gz')     
        
    
    
    
if __name__ == '__main__':
#     testGradients()     
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main() 