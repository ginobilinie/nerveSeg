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
from Unet2d_pytorch import UNet
from Unet3d_pytorch import UNet3D
from nnBuildUnits import CrossEntropy3d
import time

'''
This code is to do the baseline experiment using UNet for infant segmentation.
By Dong Nie
Nov. 2017
'''

# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--how2normalize", type=int, default=0, help="how to normalize the data")
parser.add_argument("--batchSize", type=int, default=10, help="training batch size")
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=5000, help="number of iterations to save the model")
parser.add_argument("--showTestPerformanceEvery", type=int, default=5000, help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decLREvery", type=int, default=40000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--prefixModelName", default="Segmentor_wce_lrdcr_1112_", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="preSub_wce_lrdcr_1112_", type=str, help="prefix of the to-be-saved predicted filename")

def main():    
    
    global opt, model 
    opt = parser.parse_args()
    print opt
        
#     prefixModelName = 'Segmentor_wce_lrdcr_1112_'
#     prefixPredictedFN = 'preSub_wce_lrdcr_1112_'
#     showTrainLossEvery = 100
#     lr = 1e-4
#     showTestPerformanceEvery = 5000
#     saveModelEvery = 5000
#     decLREvery = 40000
#     numofIters = 200000
#     how2normalize = 0
    
    
    
    #net=UNet()
    net = UNet3D(in_channel=3, n_classes=4)
    net.cuda()
    params = list(net.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())
    
    
    mu_mr = 0.0138117
    mu_t1 = 272.49
    mu_t2 = 49.42
     
    std_mr = 0.0578914
    std_t1 = 1036.12705933
    std_t2 = 193.835485614
 
    
    optimizer = optim.SGD(net.parameters(),lr=opt.lr)
    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
#     criterion = nn.NLLLoss2d()
    
    given_weight = torch.cuda.FloatTensor([1,4,4,4])
    
    criterion_3d = CrossEntropy3d(weight=given_weight)
    criterion_3d = criterion_3d.cuda()
    #inputs=Variable(torch.randn(1000,1,32,32)) #here should be tensor instead of variable
    #targets=Variable(torch.randn(1000,10,1,1)) #here should be tensor instead of variable
#     trainset=data_utils.TensorDataset(inputs, targets)
#     trainloader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#     inputs=torch.randn(1000,1,32,32)
#     targets=torch.LongTensor(1000)
    
    path_test ='/home/dongnie/Desktop/Caffes/data/month6_library_new/normals'
    path_patients_h5 = '/home/dongnie/Desktop/Caffes/data/month6_library_new/infantSegH5'
    path_patients_h5_test ='/home/dongnie/Desktop/Caffes/data/month6_library_new/infantSegH5Test'
#     batch_size=10
    data_generator = Generator_3D_patches(path_patients_h5,opt.batchSize,inputKey1='dataFA',inputKey2='dataT1',inputKey3='dataT2',outputKey='dataSeg')
    data_generator_test = Generator_3D_patches(path_patients_h5_test,opt.batchSize,inputKey1='dataFA',inputKey2='dataT1',inputKey3='dataT2',outputKey='dataSeg')

########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 

        # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    running_loss = 0.0
    start = time.time()
    for iter in range(opt.start_epoch,opt.numofIters+1):
        #print('iter %d'%iter)
        
        fa,t1,t2,labels = data_generator.next()
        
        fa = (fa - mu_mr)/std_mr
        t1 = (t1 - mu_t1)/std_t1
        t2 = (t2 - mu_t2)/std_t2
        
        fa = np.expand_dims(fa, axis=1)
        t1 = np.expand_dims(t1, axis=1)
        t2 = np.expand_dims(t2, axis=1)
        inputs = np.concatenate((fa,t1,t2),axis=1)
        inputs = np.squeeze(inputs)
#         print 'shape is ....',inputs.shape
        labels = np.squeeze(labels)
        labels = labels.astype(int)

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        #we should consider different data to train
        
        #wrap them into Variable
        inputs,labels = Variable(inputs),Variable(labels)
        
        
        ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))
        
#         print inputs.data.shape
        outputG = net(inputs) #here I am not sure whether we should use twice or not
        net.zero_grad()
        
        
        lossG_G = criterion_3d(outputG,torch.squeeze(labels)) 
                
        lossG_G.backward() #compute gradients
        
        #lossG_D.backward()

        
        #for other losses, we can define the loss function following the pytorch tutorial
        
        optimizer.step() #update network parameters

        #print('loss for generator is %f'%lossG.data[0])
        #print statistics
        running_loss = running_loss + lossG_G.data[0]
#         print 'running_loss is ',running_loss,' type: ',type(running_loss)
        
#         print type(outputD_fake.cpu().data[0].numpy())
        
        if iter%opt.showTrainLossEvery==0: #print every 2000 mini-batches
            print '************************************************'
            print 'time now is: ' + time.asctime(time.localtime(time.time()))
#             print 'running loss is ',running_loss
            print 'average running loss for generator between iter [%d, %d] is: %.3f'%(iter - 100 + 1,iter,running_loss/100)

            print 'lossG_G is %.2f respectively.'%(lossG_G.data[0])

            print 'cost time for iter [%d, %d] is %.2f'%(iter - 100 + 1,iter, time.time()-start)
            print '************************************************'
            running_loss = 0.0
            start = time.time()
        if iter%opt.saveModelEvery==0: #save the model
            torch.save(net.state_dict(), opt.prefixModelName+'%d.pt'%iter)
            print 'save model: '+opt.prefixModelName+'%d.pt'%iter
        if iter%opt.decLREvery==0:
            opt.lr = opt.lr*0.1
            adjust_learning_rate(optimizer, opt.lr)
                
        if iter%opt.showTestPerformanceEvery==0: #test one subject  
            # to test on the validation dataset in the format of h5 
            fa,t1,t2,labels = data_generator_test.next()
            fa = np.expand_dims(fa, axis=1)
            t1 = np.expand_dims(t1, axis=1)
            t2 = np.expand_dims(t2, axis=1)
            inputs = np.concatenate((fa,t1,t2),axis=1)
            inputs = np.squeeze(inputs)

            labels = np.squeeze(labels)
            labels = labels.astype(int)
            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs,labels = Variable(inputs),Variable(labels)
            outputG = net(inputs)
            lossG_G = criterion_3d(outputG,torch.squeeze(labels))
            
            print '.......come to validation stage: iter {}'.format(iter),'........'
            print 'lossG_G is %.2f.'%(lossG_G.data[0])


            mr_test_itk=sitk.ReadImage(os.path.join(path_test,'NORMAL01_cbq-FA.hdr'))
            t1_test_itk=sitk.ReadImage(os.path.join(path_test,'NORMAL01_cbq.hdr'))
            t2_test_itk=sitk.ReadImage(os.path.join(path_test,'NORMAL01_cbq-T2.hdr'))
            ct_test_itk=sitk.ReadImage(os.path.join(path_test,'NORMAL01-ls-corrected.hdr'))
            
            mrnp=sitk.GetArrayFromImage(mr_test_itk)
            t1np=sitk.GetArrayFromImage(t1_test_itk)
            t2np=sitk.GetArrayFromImage(t2_test_itk)
            
    
            ctnp=sitk.GetArrayFromImage(ct_test_itk)
            ctnp[ctnp>200]=3 #white matter
            ctnp[ctnp>100]=2 #gray matter
            ctnp[ctnp>4]=1 #csf
            
            ##### specific normalization #####
            mrnp = (mrnp - mu_mr)/std_mr
            t1np = (t1np - mu_t1)/std_t1
            t2np = (t2np - mu_t2)/std_t2
            
            mu = np.mean(mrnp)
            
            #for training data in pelvicSeg
            if opt.how2normalize == 1:
                maxV, minV=np.percentile(mrnp, [99 ,1])
                print 'maxV,',maxV,' minV, ',minV
                mrnp=(mrnp-mu)/(maxV-minV)
                print 'unique value: ',np.unique(ctnp)

            #for training data in pelvicSeg
            if opt.how2normalize == 2:
                maxV, minV = np.percentile(mrnp, [99 ,1])
                print 'maxV,',maxV,' minV, ',minV
                mrnp = (mrnp-mu)/(maxV-minV)
                print 'unique value: ',np.unique(ctnp)
            
            #for training data in pelvicSegRegH5
            if opt.how2normalize== 3:
                std = np.std(mrnp)
                mrnp = (mrnp - mu)/std
                print 'maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp)
    
#             full image version with average over the overlapping regions
#             ct_estimated = testOneSubject(mrnp,ctnp,[3,168,112],[1,168,112],[1,8,8],netG,'Segmentor_model_%d.pt'%iter)
            
#             sz = mrnp.shape
#             matFA = np.zeros(sz[0],3,sz[2],sz[3],sz[4])
            matFA = np.concatenate((np.expand_dims(mrnp,axis=0),np.expand_dims(t1np,axis=0),np.expand_dims(t2np,axis=0)),axis=0)
             #note, matFA and matFAOut same size 
            matGT = ctnp
#                 volFA = sitk.GetImageFromArray(matFA)
#                 sitk.WriteImage(volFA,'volFA'+'.nii.gz')
#                 volGT = sitk.GetImageFromArray(matGT)
#                 sitk.WriteImage(volGT,'volGT'+'.nii.gz')
#             print 'matFA shape: ',matFA.shape
            matOut,_ = testOneSubject(matFA,matGT,4,[32,32,32],[32,32,32],[32,32,32],net,opt.prefixModelName+'%d.pt'%iter)
            print 'matOut shape: ',matOut.shape
            ct_estimated = matOut

            ct_estimated = np.rint(ct_estimated) 
#             ct_estimated = denoiseImg(ct_estimated, kernel=np.ones((20,20,20)))   
            diceBladder = dice(ct_estimated,ctnp,1)
            diceProstate = dice(ct_estimated,ctnp,2)
            diceRectumm = dice(ct_estimated,ctnp,3)
            
            print 'pred: ',ct_estimated.dtype, ' shape: ',ct_estimated.shape
            print 'gt: ',ctnp.dtype,' shape: ',ct_estimated.shape
            print 'dice1 = ',diceBladder,' dice2= ',diceProstate,' dice3= ',diceRectumm
            volout = sitk.GetImageFromArray(ct_estimated)
            sitk.WriteImage(volout,opt.prefixPredictedFN+'{}'.format(iter)+'.nii.gz')    
#             netG.save_state_dict('Segmentor_model_%d.pt'%iter)
#             netD.save_state_dic('Discriminator_model_%d.pt'%iter)
        
    print('Finished Training')
    
if __name__ == '__main__':     
    main()
    
