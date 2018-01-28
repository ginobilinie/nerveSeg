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
from Unet2d_pytorch import UNet, Discriminator
from Unet3d_pytorch import UNet3D
from nnBuildUnits import CrossEntropy3d
import time

'''
This copy of code is to implement a baseline 3T/7T reconstruction network using a 23D-Unet.
By Dong Nie
June 2017
'''

# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=True)
parser.add_argument("--how2normalize", type=int, default=0, help="how to normalize the data")
parser.add_argument("--batchSize", type=int, default=10, help="training batch size")
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=5000, help="number of iterations to save the model")
parser.add_argument("--showTestPerformanceEvery", type=int, default=5000, help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decLREvery", type=int, default=100000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--prefixModelName", default="/shenlab/local/dongnie/3T7T/Regressor_1122_", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="preSub1_1122_", type=str, help="prefix of the to-be-saved predicted filename")

def main():    
    global opt, model 
    opt = parser.parse_args()
    print opt    
        
#     prefixModelName = 'Regressor_1112_'
#     prefixPredictedFN = 'preSub1_1112_'
#     showTrainLossEvery = 100
#     lr = 1e-4
#     showTestPerformanceEvery = 2000
#     saveModelEvery = 2000
#     decLREvery = 40000
#     numofIters = 200000
#     how2normalize = 0


    netD = Discriminator()
    netD.apply(weights_init)
    netD.cuda()
    
    optimizerD =optim.Adam(netD.parameters(),lr=1e-3)
    criterion_bce=nn.BCELoss()
    criterion_bce.cuda()
    
    #net=UNet()
    net = UNet(in_channel=5, n_classes=1)
#     net.apply(weights_init)
    net.cuda()
    params = list(net.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())
    
 
    
    optimizer = optim.Adam(net.parameters(),lr=opt.lr)
    criterion_L2 = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    #criterion = nn.CrossEntropyLoss()
#     criterion = nn.NLLLoss2d()
    
    given_weight = torch.cuda.FloatTensor([1,4,4,2])
    
    criterion_3d = CrossEntropy3d(weight=given_weight)
    
    criterion_3d = criterion_3d.cuda()
    criterion_L2 = criterion_L2.cuda()
    criterion_L1 = criterion_L1.cuda()
    
    #inputs=Variable(torch.randn(1000,1,32,32)) #here should be tensor instead of variable
    #targets=Variable(torch.randn(1000,10,1,1)) #here should be tensor instead of variable
#     trainset=data_utils.TensorDataset(inputs, targets)
#     trainloader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#     inputs=torch.randn(1000,1,32,32)
#     targets=torch.LongTensor(1000)
    
    path_test ='/shenlab/lab_stor5/dongnie/3T7T'
    path_patients_h5 = '/shenlab/lab_stor5/dongnie/3T7T/histH5Data_64to64'
    path_patients_h5_test ='/shenlab/lab_stor5/dongnie/3T7T/histH5DataTest_64to64'
#     batch_size=10
    data_generator = Generator_2D_slices(path_patients_h5,opt.batchSize,inputKey='data3T',outputKey='data7T')
    data_generator_test = Generator_2D_slices(path_patients_h5_test,opt.batchSize,inputKey='data3T',outputKey='data7T')

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
    running_loss = 0.0
    start = time.time()
    for iter in range(opt.start_epoch, opt.numofIters+1):
        #print('iter %d'%iter)
        
        inputs, labels = data_generator.next()
#         xx = np.transpose(inputs,(5,64,64))
        inputs = np.transpose(inputs,(0,3,1,2))
        inputs = np.squeeze(inputs) #5x64x64
#         print 'shape is ....',inputs.shape
        labels = np.squeeze(labels) #64x64
#         labels = labels.astype(int)

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        #we should consider different data to train
        
        #wrap them into Variable
        inputs,labels = Variable(inputs),Variable(labels)
        
        ## (1) update D network: maximize log(D(x)) + log(1 - D(G(z)))
        if opt.isAdLoss:
            outputG = net(inputs) #5x64x64->1*64x64
            
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
            outputD_real = netD(labels)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
                
            outputD_fake = netD(outputG)
            netD.zero_grad()
            batch_size = inputs.size(0)
            real_label = torch.ones(batch_size,1)
            real_label = real_label.cuda()
            #print(real_label.size())
            real_label = Variable(real_label)
            #print(outputD_real.size())
            loss_real = criterion_bce(outputD_real,real_label)
            loss_real.backward()
            #train with fake data
            fake_label = torch.zeros(batch_size,1)
    #         fake_label = torch.FloatTensor(batch_size)
    #         fake_label.data.resize_(batch_size).fill_(0)
            fake_label = fake_label.cuda()
            fake_label = Variable(fake_label)
            loss_fake = criterion_bce(outputD_fake,fake_label)
            loss_fake.backward()
            
            lossD = loss_real + loss_fake
#             print 'loss_real is ',loss_real.data[0],'loss_fake is ',loss_fake.data[0],'outputD_real is',outputD_real.data[0]
#             print('loss for discriminator is %f'%lossD.data[0])
            #update network parameters
            optimizerD.step()
        
        
        ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))
        
#         print inputs.data.shape
        outputG = net(inputs) #here I am not sure whether we should use twice or not
        net.zero_grad()   
        lossG_G = criterion_L1(outputG,torch.squeeze(labels)) 
        lossG_G.backward() #compute gradients
        
        if opt.isAdLoss:
            #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
            #angel of equation (note the max and min difference for generator and discriminator)
            outputG = net(inputs)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD = netD(outputG)
            lossG_D = criterion_bce(outputD,real_label) #note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_D.backward()
        
        #for other losses, we can define the loss function following the pytorch tutorial
        
        optimizer.step() #update network parameters

        #print('loss for generator is %f'%lossG.data[0])
        #print statistics
        running_loss = running_loss + lossG_G.data[0]

        
        if iter%opt.showTrainLossEvery==0: #print every 2000 mini-batches
            print '************************************************'
            print 'time now is: ' + time.asctime(time.localtime(time.time()))
#             print 'running loss is ',running_loss
            print 'average running loss for generator between iter [%d, %d] is: %.3f'%(iter - 100 + 1,iter,running_loss/100)
            
            print 'lossG_G is %.2f respectively.'%(lossG_G.data[0])
            if opt.isAdLoss:
                print 'loss_real is ',loss_real.data[0],'loss_fake is ',loss_fake.data[0],'outputD_real is',outputD_real.data[0]
                print('loss for discriminator is %f'%lossD.data[0])                
            
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
            inputs,labels = data_generator_test.next()

            inputs = np.transpose(inputs,(0,3,1,2))
            inputs = np.squeeze(inputs)

            labels = np.squeeze(labels)

            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs,labels = Variable(inputs),Variable(labels)
            outputG = net(inputs)
            lossG_G = criterion_L1(outputG,torch.squeeze(labels))
            
            print '.......come to validation stage: iter {}'.format(iter),'........'
            print 'lossG_G is %.2f.'%(lossG_G.data[0])


            mr_test_itk=sitk.ReadImage(os.path.join(path_test,'S1to1_3t.nii.gz'))
            ct_test_itk=sitk.ReadImage(os.path.join(path_test,'S1to1_7t.nii.gz'))
            
            mrnp=sitk.GetArrayFromImage(mr_test_itk)
            ctnp=sitk.GetArrayFromImage(ct_test_itk)
            
            ##### specific normalization #####
            mu = np.mean(mrnp)
            maxV, minV = np.percentile(mrnp, [99 ,25])
            #mrimg=mrimg
            mrnp = (mrnp-minV)/(maxV-minV)

            
            
            #for training data in pelvicSeg
            if opt.how2normalize == 1:
                maxV, minV = np.percentile(mrnp, [99 ,1])
                print 'maxV,',maxV,' minV, ',minV
                mrnp = (mrnp-mu)/(maxV-minV)
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
            matFA = mrnp
             #note, matFA and matFAOut same size 
            matGT = ctnp
#                 volFA = sitk.GetImageFromArray(matFA)
#                 sitk.WriteImage(volFA,'volFA'+'.nii.gz')
#                 volGT = sitk.GetImageFromArray(matGT)
#                 sitk.WriteImage(volGT,'volGT'+'.nii.gz')
#             print 'matFA shape: ',matFA.shape
            matOut = test_1_subject(matFA,matGT,[64,64,5],[64,64,1],[32,32,1],net,opt.prefixModelName+'%d.pt'%iter)
            print 'matOut shape: ',matOut.shape
            ct_estimated = matOut

#             ct_estimated = np.rint(ct_estimated) 
#             ct_estimated = denoiseImg(ct_estimated, kernel=np.ones((20,20,20)))   
#             diceBladder = dice(ct_estimated,ctnp,1)
#             diceProstate = dice(ct_estimated,ctnp,2)
#             diceRectumm = dice(ct_estimated,ctnp,3)
            itspsnr = psnr(ct_estimated, matGT)
            
            print 'pred: ',ct_estimated.dtype, ' shape: ',ct_estimated.shape
            print 'gt: ',ctnp.dtype,' shape: ',ct_estimated.shape
            print 'psnr = ',itspsnr
            volout = sitk.GetImageFromArray(ct_estimated)
            sitk.WriteImage(volout,opt.prefixPredictedFN+'{}'.format(iter)+'.nii.gz')    
#             netG.save_state_dict('Segmentor_model_%d.pt'%iter)
#             netD.save_state_dic('Discriminator_model_%d.pt'%iter)
        
    print('Finished Training')
    
if __name__ == '__main__':   
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'  
    main()
    
