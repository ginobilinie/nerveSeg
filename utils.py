import numpy as np
import os
import SimpleITK as sitk
import h5py
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import torch.nn.init
from torch.autograd import Variable

'''
Data generator, evluation for one subject and so on.
Dong Nie
Jan. 2017
'''

#Dong add keys here
def Generator_2D_slices(path_patients,batchsize,inputKey='dataMR',outputKey='dataCT'):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient))
            #dataMRptr=f['dataMR']
            dataMRptr=f[inputKey]
            dataMR=dataMRptr.value
            
            #dataCTptr=f['dataCT']
            dataCTptr=f[outputKey]
            dataCT=dataCTptr.value

            dataMR=np.squeeze(dataMR)
            dataCT=np.squeeze(dataCT)

            #print 'mr shape h5 ',dataMR.shape#B,H,W,C
            #print 'ct shape h5 ',dataCT.shape#B,H,W
            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0,dataMR.shape[0],to_add)
                X=np.zeros((dataMR.shape[0]+to_add,dataMR.shape[1],dataMR.shape[2],dataMR.shape[3]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)

            #X = np.expand_dims(X, axis=3)    
            X=X.astype(np.float32)
            y=np.expand_dims(y, axis=3)#B,H,W,C
            y=y.astype(np.float32)
            #y[np.where(y==5)]=0
            
            #shuffle the data, by dong
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            X=X[inds,...]
            y=y[inds,...]
            
            print 'X shape ', X.shape  
            print 'y shape ', y.shape                   
            for i_batch in xrange(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])



def Generator_3D_patches(path_patients,batchsize,inputKey1='dataMR', inputKey2=None, inputKey3=None, outputKey='dataSeg'):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
#     print 'keys: ',inputKey1,inputKey2,inputKey3,outputKey
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient))
#             print f.keys()
            dataMRptr=f[inputKey1]
            dataMR=dataMRptr.value
            
            if inputKey2!=None:
                dataT1Ptr = f[inputKey2]
                dataT1 = dataT1Ptr.value
                dataT1 = np.squeeze(dataT1)
            
            if inputKey3!=None:
                dataT2Ptr = f[inputKey3]
                dataT2 = dataT2Ptr.value
                dataT2 = np.squeeze(dataT2)
                
            #dataMR=np.squeeze(dataMR)
            
            dataCTptr=f[outputKey]
            dataCT=dataCTptr.value
            #dataCT=np.squeeze(dataCT)

            dataMR=np.squeeze(dataMR)
            dataCT=np.squeeze(dataCT)
            
            
            
            print 'mr shape h5 ',dataMR.shape

            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            
            if inputKey2!=None:
                dataT1 = dataT1[idx_rnd,...]
            
            if inputKey3!=None:
                dataT2 = dataT2[idx_rnd,...]
            
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0, dataMR.shape[0], to_add)
                X=np.zeros((dataMR.shape[0]+to_add, dataMR.shape[1], dataMR.shape[2], dataMR.shape[3]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                if inputKey2!=None:
                    X1=np.zeros((dataT1.shape[0]+to_add, dataT1.shape[1], dataT1.shape[2], dataT1.shape[3]))#dataMR
                    X1[:dataT1.shape[0],...]=dataT1
                    X1[dataT1.shape[0]:,...]=dataT1[inds_toadd]     
                    
                if inputKey3!=None:
                    X2=np.zeros((dataT2.shape[0]+to_add, dataT2.shape[1], dataT2.shape[2], dataT2.shape[3]))#dataMR
                    X2[:dataT2.shape[0],...]=dataT2
                    X2[dataT2.shape[0]:,...]=dataT2[inds_toadd] 
                
                y=np.zeros((dataCT.shape[0]+to_add, dataCT.shape[1], dataCT.shape[2], dataCT.shape[3]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X=np.copy(dataMR)    
                if inputKey2!=None:
                    X1 = np.copy(dataT1)
                if inputKey3!=None:
                    X2 = np.copy(dataT2)            
                y=np.copy(dataCT)

            X = np.expand_dims(X, axis=4)     
            X=X.astype(np.float32)
            
            if inputKey2!=None:
                X1 = np.expand_dims(X1, axis=4)     
                X1 = X1.astype(np.float32)
                    
            if inputKey3!=None:
                X2 = np.expand_dims(X2, axis=4)     
                X2 = X2.astype(np.float32)
            
            y=np.expand_dims(y, axis=4)
            y=y.astype(np.float32)
            
            print 'y shape ', y.shape
            print 'X shape ', X.shape
            
            if inputKey2==None:                 
                for i_batch in xrange(int(X.shape[0]/batchsize)):
                    yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])
            elif inputKey3!=None:
                for i_batch in xrange(int(X.shape[0]/batchsize)):
                    yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...], X1[i_batch*batchsize:(i_batch+1)*batchsize,...], X2[i_batch*batchsize:(i_batch+1)*batchsize,...], y[i_batch*batchsize:(i_batch+1)*batchsize,...])
            else:
                for i_batch in xrange(int(X.shape[0]/batchsize)):
                    yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...], X1[i_batch*batchsize:(i_batch+1)*batchsize,...], y[i_batch*batchsize:(i_batch+1)*batchsize,...])    
# custom weights initialization called on netG and netD
def weights_init(m):
    xavier=torch.nn.init.xavier_uniform
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
        #print m.weight.data
        #print m.bias.data
        xavier(m.weight.data)
        #xavier(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear')!=-1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


'''
    this function is used to compute the dice ratio
input:
    im1: gt
    im2 pred
    tid: the id for consideration
output:
    dcs
'''
def dice(im1, im2,tid):
    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

'''
    compute the peak signal noise ratio
Input:
    ct_generated: generated ct
    ct_GT: ground truth
Output:
    return psnr
'''
def psnr(ct_generated,ct_GT):
    print ct_generated.shape
    print ct_GT.shape

    mse=np.sqrt(np.mean((ct_generated-ct_GT)**2))
    print 'mse ',mse
    max_I=np.max([np.max(ct_generated),np.max(ct_GT)])
    print 'max_I ',max_I
    return 20.0*np.log10(max_I/mse)


'''
    Evaluate one patch using the latest pytorch model
input:
    patch_MR: a np array of shape [H,W,nchans]
    netG: the trained network
    modelPath: the path to the saved model (we donot use it here)
    isCla: if yes, we take it as a classification task, otherwise, it is a regression task
output:    
    patch_CT_pred: segmentation maps for the corresponding input patch 
'''
def evaluate(patch_MR, netG, modelPath, isCla=True):
    
    
        patch_MR = torch.from_numpy(patch_MR)

        patch_MR = patch_MR.unsqueeze(0)
        
#         patch_MR=np.expand_dims(patch_MR,axis=0)#[1,H,W,nchans]
        patch_MR = Variable(patch_MR).float().cuda()
#         netG = ResSegNet() #here is your network
#         netG.load_state_dict(torch.load(modelPath))
        netG.cuda()
        netG.eval()
#         print type(patch_MR), patch_MR.size()
        res = netG(patch_MR)
        
#         print res.size(),res.squeeze(0).size()
        if isCla:
            _, tmp = res.squeeze(0).max(0)
        else:
            tmp = res.squeeze(0)
        
        patchOut = tmp.data.cpu().numpy().squeeze()

        #imsave('mr32.png',np.squeeze(MR16_eval[0,:,:,2]))
        #imsave('ctpred.png',np.squeeze(patch_CT_pred[0,:,:,0]))
        #print 'mean of layer  ',np.mean(MR16_eval)
        #print 'min ct estimated ',np.min(patch_CT_pred)
        #print 'max ct estimated ',np.max(patch_CT_pred)
        #print 'mean of ctpatch estimated ',np.mean(patch_CT_pred)
        return patchOut


'''
Segmentation task:
    Receives an MR image and returns an segmentation label maps with the same size
    We use majority voting at the overlapping regions
input:
    MR_image: the raw input data (after preprocessing), HxWxD or ch*HxWxD
    CT_GT: the ground truth data
    NumOfClass: number of classes
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
output:
    matOut: the predicted segmentation map
'''
def testOneSubject(MR_image, CT_GT, NumOfClass, MR_patch_sz, CT_patch_sz, step, netG, modelPath):
    eps=1e-5
    
    matFA = MR_image
    matSeg = CT_GT
    
    print 'matFA, size: ',matFA.shape
    
    dFA = MR_patch_sz
    dSeg = CT_patch_sz
    
    ch = 1
    if len(matFA.shape)==4:
        [ch, row,col,leng] = matFA.shape
    else:
        [row,col,leng] = matFA.shape
        
    margin1 = (dFA[0]-dSeg[0])/2
    margin2 = (dFA[1]-dSeg[1])/2
    margin3 = (dFA[2]-dSeg[2])/2
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    
    #print 'matFA shape is ',matFA.shape
    matFAOut = np.zeros([ch, row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    #print 'matFAOut shape is ',matFAOut.shape
    matFAOut[0:ch, marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA

#     matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
#     matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension
# 
#     matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
#     matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension
# 
#     matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
#     matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,leng-marginD[2]:matFA.shape[2]]
    
    if margin1!=0:
        matFAOut[:,0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[:,row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[:,marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[:,marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[:,marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[:,marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    

#     dim1=np.arange(80,192)
#     dim2=np.arange(35,235)
#     x1=80
#     x2=192
#     y1=35
#     y2=235
# #     matFAOutScale = matFAOut[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
# #     matSegScale = matSeg[:,y1:y2,x1:x2]
    matFAOutScale = matFAOut
#     print 'matFAout shape: ',matFAOut.shape
    matSegScale = matSeg
    matOut = np.zeros((matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2],NumOfClass),dtype=np.int32)
    [row,col,leng] = matSegScale.shape
        
    cnt = 0
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA = matFAOutScale[:,i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                cnt = cnt + 1
                volFA = np.squeeze(volFA)
#                 print 'volFA shape:',volFA.shape
                temppremat = evaluate(volFA, netG, modelPath)
#                 volPre = sitk.GetImageFromArray(temppremat)
#                 sitk.WriteImage(volPre,'volPre_{}'.format(cnt)+'.nii.gz')
                
                if len(temppremat.shape)==2:
                    temppremat = np.expand_dims(temppremat,axis=0)
                for labelInd in range(NumOfClass): #note, start from 0
                    currLabelMat = np.where(temppremat==labelInd, 1, 0) # true, vote for 1, otherwise 0
                    #scio.savemat('volOut_%d'%cnt+'_label%d.mat'%labelInd,{'currLabelMat%d'%labelInd:currLabelMat})
                    matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd]+currLabelMat;
       
    #scio.savemat('matOut%s.mat'%fileID,{'matOut':matOut})
    matOut = matOut.argmax(axis=3)
    matOut = np.rint(matOut) #this is necessary to convert the data type to be accepted by NIFTI, otherwise will appear strange errors
#     print 'line 378: matOut shape: ',matOut.shape
#     matOut1 = matOut
#     matOut1=np.zeros([matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]])
#     matOut1[:,y1:y2,x1:x2]=matOut
    #matOut1=np.transpose(matOut1,(2,1,0))
    #matSeg=np.transpose(matSeg,(2,1,0))
    return matOut,matSeg


'''
Regression task:
    Receives an MR image and returns an predicted maps with the same size
    We use averaging at the overlapping regions
input:
    MR_image: the raw input data (after preprocessing), HxWxD or ch*HxWxD
    CT_GT: the ground truth data
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)
output:
    matOut: the predicted segmentation map
'''
def test_1_subject(MR_image,CT_GT,MR_patch_sz,CT_patch_sz,step, netG, modelPath):
        """
            receives an MR image and returns an estimated CT image of the same size
        """
        matFA = MR_image
        matSeg = CT_GT
        dFA = MR_patch_sz
        dSeg = CT_patch_sz

        eps = 1e-5
        [row,col,leng] = matFA.shape
        margin1 = int((dFA[0]-dSeg[0])/2)
        margin2 = int((dFA[1]-dSeg[1])/2)
        margin3 = int((dFA[2]-dSeg[2])/2)
        cubicCnt = 0
        marginD = [margin1,margin2,margin3]
        print 'matFA shape is ',matFA.shape
        matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
        print 'matFAOut shape is ',matFAOut.shape
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA

        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension

        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]] = matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]] = matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension

        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]] = matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]] = matFA[:,:,leng-marginD[2]:matFA.shape[2]]


        matOut = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))
        used = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
        #fid=open('trainxxx_list.txt','a');
#         print 'last i ',row-dSeg[0]
        for i in range(0,row-dSeg[0]+1,step[0]):
#             print 'i ',i
            for j in range(0,col-dSeg[1]+1,step[1]):
                for k in range(0,leng-dSeg[2]+1,step[2]):
                    volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                    #print 'volSeg shape is ',volSeg.shape
                    volFA = matFAOut[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                    #print 'volFA shape is ',volFA.shape
                    #mynet.blobs['dataMR'].data[0,0,...]=volFA
                    #mynet.forward()
                    #temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                    volFA = np.transpose(volFA,(2,0,1)) #5x64x64
                    temppremat = evaluate(volFA, netG, modelPath, isCla=False)
                    if len(temppremat.shape)==2:
                        temppremat = np.expand_dims(temppremat,axis=2)
                    #print 'patchout shape ',temppremat.shape
                    #temppremat=volSeg
                    matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat
                    used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1
        matOut = matOut/used
        return matOut



'''
    Sets the learning rate to the initial LR decayed by 10 every 10 epochs
'''
def adjust_learning_rate(optimizer, lr):
#     lr = opt.lr * (0.1 ** (epoch // opt.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  
    return lr
