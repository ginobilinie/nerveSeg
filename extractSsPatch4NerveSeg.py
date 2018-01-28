    
'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
for single-scale patches
We also extract patches for each organ separately which aims at jointly training
We crop segmentation maps, regression maps, and the contours.
Created on Oct. 20, 2016

We use adaptively steps to extract patches

Author: Dong Nie 
'''



import SimpleITK as sitk
import argparse
import datetime
from multiprocessing import Pool
import os
from os.path import isfile, join
from os import listdir
import h5py
import numpy as np
from skimage import feature

eps = 1e-5
d1 = 16 
d2 = 64
d3 = 64
dFA = [d1,d2,d3] # size of patches of input data
dSeg = [16,64,64] # size of pathes of label data
step1 = 1
step2 = 20
step3 = 20
step = [step1,step2,step3]
minIntensity = -0.82
    
    
# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--how2normalize", type=int, default=4, help="how to normalize the data")
parser.add_argument("--basePath", default="/home/dongnie/warehouse/nerveSeg/", type=str, help="base path for this project's data")
parser.add_argument("--dataFolder", default="data/", type=str, help="the name of the data source folder")
parser.add_argument("--labelFolder", default="labels/", type=str, help="the name of the label source folder")
parser.add_argument("--save2Folder", default="nerveH5/", type=str, help="the name of the save2folder")
parser.add_argument("--saveH5FN", default="train16x64x64_segregcontour_", type=str, help="the prefix of the save file(h5)")
parser.add_argument("--h5FilesListFN", default="trainPelvic16x64x64_segregcontour_list.txt", type=str, help="the prefix of the save file(h5)")
parser.add_argument("--adaptiveStepRatio", type=int, default=16, help="the adaptively step ratio to help decide proper step")

'''
   We obtian the 3D contours from the 3D images 
'''    
def obtain3DContours(mat,dim=0):
    cmat = np.zeros(mat.shape)
    if dim==0:
        for i in range(0,mat.shape[0]):
            cmat[i,:,:] = feature.canny(np.squeeze(mat[i,:,:])) #we should care about which value it really is
#         edges2 = feature.canny(mat[:,:,i], sigma=3)
    elif dim==1:
        for i in range(0,mat.shape[1]):
            cmat[:,i,:] = feature.canny(np.squeeze(mat[:,i,:])) #we should care about which value it really is
#         edges2 = feature.canny(mat[:,:,i], sigma=3)
    else:
        for i in range(0,mat.shape[2]):
            cmat[:,:,i] = feature.canny(np.squeeze(mat[:,:,i])) #we should care about which value it really is
#         edges2 = feature.canny(mat[:,:,i], sigma=3)
    return cmat
'''
This is useful to generate hdf5 database:
we generate regression/segmentation ground truth, as well as contour based ground truth
'''
def extractMsPatches4OneSubject(matFA,matSeg,fileID,d,step,rate):
  
    matContour = obtain3DContours(matSeg, dim=0)
    rate1=1.0/2
    rate2=1.0/4
    [row,col,leng] = matFA.shape
    
    step_col = np.rint(col/opt.adaptiveStepRatio)
    step[1] = step_col.astype(int)
    step_leng = np.rint(col/opt.adaptiveStepRatio)
    step[2] = step_leng.astype(int)
    cubicCnt = 0
    estNum = 20000
    trainFA = np.zeros([estNum,1, dFA[0],dFA[1],dFA[2]],np.float16)
    trainSeg = np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
    trainContour = np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
    trainFA2D = np.zeros([estNum, dFA[0],dFA[1],dFA[2]],np.float16)
#     trainBladder2D=np.zeros([estNum, dFA[0],dFA[1],dSeg[2]],np.float16)
    trainProstate = np.zeros([estNum, 1, dFA[0],dFA[1],dSeg[2]],np.float16)
#     trainRectum2D=np.zeros([estNum, dFA[0],dFA[1],dSeg[2]],np.float16)
    trainSeg2D=np.zeros([estNum,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
    trainContour2D = np.zeros([estNum,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)

    print 'trainFA shape, ',trainFA.shape
    #to padding for input
    margin1=(dFA[0]-dSeg[0])/2
    margin2=(dFA[1]-dSeg[1])/2
    margin3=(dFA[2]-dSeg[2])/2
    cubicCnt=0
    marginD=[margin1,margin2,margin3]
    print 'matFA shape is ',matFA.shape
    matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA
    matSegOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg
    #for mageFA, enlarge it by padding
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    #for matseg, enlarge it by padding
    if margin1!=0:
        matSegOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matSegOut[row+marginD[0]:matSegOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[matSeg.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matSegOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matSeg[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matSegOut[marginD[0]:row+marginD[0],col+marginD[1]:matSegOut.shape[1],marginD[2]:leng+marginD[2]]=matSeg[:,matSeg.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matSeg[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matSegOut.shape[2]]=matSeg[:,:,matSeg.shape[2]-1:leng-marginD[2]-1:-1]
    

        
    dsfactor = rate
    centerSlice=(dFA[2]-dSeg[2])/2
#     matSeg=matSeg.astype(np.int8)
    #actually, we can specify a bounding box along the 2nd and 3rd dimension, so we can make it easier 
    for i in range(0,row-dSeg[0],step[0]):
        for j in range(0,col-dSeg[1],step[1]):
            for k in range(0,leng-dSeg[2],step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] #note, we donot need matSegOut
                volContour = matContour[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]#note, we donot need matSegOut
                if np.sum(volSeg)<eps:
                    continue
                cubicCnt = cubicCnt+1
                #index at scale 1
                
                volFA = matFAOut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
              
                volSeg = volSeg.astype(int)
                volContour = volContour.astype(int)
                
#                 print type(volSeg)
#                 print type(volSeg[0,0,0])
                
                #print volSeg!=0
#                 volBladder = np.expand_dims(volFA[centerSlice,:,:],axis=0)
                volProstate = np.expand_dims(volFA[centerSlice,:,:],axis=0)
#                 volRectum = np.expand_dims(volFA[centerSlice,:,:],axis=0)
#                 print np.unique(volSeg)
                #print 'volBladder shape, ',volBladder.shape


                ### it seems the following codes doesn't work, so I use mask encoding as below
#                 volBladder[volSeg!=1]=0
#                 volProstate[volSeg!=2]=0
#                 volRectum[volSeg!=3]=0
#                 print np.unique(volBladder),np.unique(volProstate),np.unique(volRectum)

                ### The following codes work, I have verifed.
#                 maskBladder = np.zeros(volSeg.shape)
#                 maskBladder[volSeg==1] = 1
#                 volBladder = maskBladder*volBladder
                maskProstate = np.zeros(volSeg.shape)
                maskProstate[volSeg==1] = 1
                volProstate = maskProstate*volProstate
#                 maskRectum = np.zeros(volSeg.shape)
#                 maskRectum[volSeg==3] = 1
#                 volRectum = maskRectum*volRectum
#                 volOut=sitk.GetImageFromArray(volBladder)
#                 sitk.WriteImage(volOut,'volBladder_%d.nii.gz'%cubicCnt)
#                 volOut=sitk.GetImageFromArray(volProstate)
#                 sitk.WriteImage(volOut,'volProstate_%d.nii.gz'%cubicCnt)
#                 volOut=sitk.GetImageFromArray(volRectum)
#                 sitk.WriteImage(volOut,'volRectum_%d.nii.gz'%cubicCnt)
                trainFA[cubicCnt,0,:,:,:] = volFA #32*32*32
                trainSeg[cubicCnt,0,:,:,:] = volSeg#24*24*24
                trainContour[cubicCnt,0,:,:,:] = volContour#24*24*24

                trainFA2D[cubicCnt,:,:,:] = volFA #32*32*32
#                 trainBladder2D[cubicCnt,:,:,:] = volBladder
                trainProstate[cubicCnt,:,:,:] = volProstate
#                 trainRectum2D[cubicCnt,:,:,:] = volRectum
                
                trainSeg2D[cubicCnt,:,:,:] = volSeg#24*24*24
                trainContour2D[cubicCnt,:,:,:] = volContour


    trainFA = trainFA[0:cubicCnt,:,:,:,:]
    trainSeg = trainSeg[0:cubicCnt,:,:,:,:]
    trainContour = trainContour[0:cubicCnt,:,:,:,:]

    trainFA2D = trainFA2D[0:cubicCnt,:,:,:]
#     trainBladder2D = trainBladder2D[0:cubicCnt,:,:,:]
    trainProstate2D = trainProstate[0:cubicCnt,:,:,:]
#     trainRectum2D = trainRectum2D[0:cubicCnt,:,:,:]
    trainSeg2D = trainSeg2D[0:cubicCnt,:,:,:]
    trainContour2D = trainContour2D[0:cubicCnt,:,:,:]

    print opt.basePath+opt.save2Folder+opt.saveH5FN+'%s.h5'%fileID
    with h5py.File(opt.basePath+opt.save2Folder+opt.saveH5FN+'%s.h5'%fileID,'w') as f:
        f['dataMR'] = trainFA
        f['dataSeg'] = trainSeg
# #         f['dataContour'] = trainContour
#         f['dataMR2D'] = trainFA2D
# #         f['dataBladder2D'] = trainBladder2D
#         f['dataProstate2D'] = trainProstate2D
#         f['dataRectum2D'] = trainRectum2D
#         f['dataSeg2D'] = trainSeg2D
#         f['dataContour2D'] = trainContour2D
     
    with open(opt.h5FilesListFN,'a') as f:
#         f.write('/shenlab/lab_stor3/dongnie/pelvicSeg/mrs_data/train3x168x112_segreg_%s.h5\n'%fileID)
        f.write(opt.basePath+opt.save2Folder+opt.saveH5FN+'%s.h5\n'%fileID)

    return cubicCnt

#to remove zero slices along the 1st dimension
def stripNullSlices(tmpMR,mrimg,labelimg):
    startS=-1
    endS=-1
    for i in range(0,tmpMR.shape[0]):
        if np.sum(tmpMR[i,:,:])<eps:
            if startS==-1:
                continue
            else:
                endS=i-1
                break
        else:
            if startS==-1:
                startS=i
            else:
                continue
    if endS==-1: #means there is no null slices at the end
        endS=tmpMR.shape[0]-1
    return startS,endS
        
def main():
    global opt
    opt = parser.parse_args()

    print opt
    
    today = datetime.date.today()
    print today
    
#     path='/home/dongnie/warehouse/mrs_data/'
#     saveto='/home/dongnie/warehouse/mrs_data/'
#     path='/shenlab/lab_stor3/dongnie/pelvicSeg/mrs_data/'
#     saveto='/shenlab/lab_stor3/dongnie/pelvicSeg/mrs_data/'
   
    allfilenames = os.listdir(opt.basePath+opt.dataFolder)
#     print allfilenames
    allfilenames = filter(lambda x: '.nii.gz' in x, allfilenames)

    for i_file, filename in enumerate(allfilenames):
        datafilename = opt.basePath+opt.dataFolder+filename
        datafn=os.path.join(opt.basePath+opt.dataFolder, datafilename)
        
        labelfilename = filename[0:len(filename)-7]+'_FN.nii.gz'
        print labelfilename
        labelfn=os.path.join(opt.basePath+opt.labelFolder, labelfilename)
        imgOrg=sitk.ReadImage(datafn)
        mrimg=sitk.GetArrayFromImage(imgOrg)
        tmpMR=mrimg
           #mrimg=mrimg
        
        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg)
#         if ind<14:
#             labelimg=labelimg/10 
        print 'data shape: ',mrimg.shape,' label shape: ',labelimg.shape
        mrShape = mrimg.shape
        labelShape = labelimg.shape
#         print 'mrShape type: ',type(mrShape)
        print 'label unique value: ',np.unique(labelimg)
        if mrShape!=labelShape:
            print 'mri shape and label shape not matched'
            print datafilename
            continue
# #             if mrShape[0]
        rate=1
        print 'unique labels are ',np.unique(labelimg)
#         print 'it comes to sub',ind
        print 'shape of mrimg, ',mrimg.shape
#         startS,endS=stripNullSlices(tmpMR,mrimg,labelimg)
#         print 'start slice is,',startS, 'end Slice is', endS
#         mrimg=mrimg[startS:endS+1,:,:]
#         print 'shape of mrimg, ',mrimg.shape
       
        
#         maxV, minV=np.percentile(mrimg, [99 ,1])
#         std=np.std(mrimg)
#         #print 'maxV,',maxV,' minV, ',minV
#         #mrimg=(mrimg-mu)/(maxV-minV)
#         mrimg=(mrimg-mu)/std
        
        #for training data in pelvicSeg
        if opt.how2normalize == 1:
            mu=np.mean(mrimg)
            maxV, minV=np.percentile(mrimg, [99 ,1])
            print 'maxV,',maxV,' minV, ',minV
            mrimg=(mrimg-mu)/(maxV-minV)
            print 'unique value: ',np.unique(labelimg)
        
        #for training data in pelvicSeg
        elif opt.how2normalize == 2:
            mu=np.mean(mrimg)
            maxV, minV = np.percentile(mrimg, [99 ,1])
            print 'maxV,',maxV,' minV, ',minV
            mrimg = (mrimg-mu)/(maxV-minV)
            print 'unique value: ',np.unique(labelimg)
        
        #for training data in pelvicSegRegH5
        elif opt.how2normalize== 3:
            mu=np.mean(mrimg)
            std = np.std(mrimg)
            mrimg = (mrimg - mu)/std
            print 'maxV,',np.ndarray.max(mrimg),' minV, ',np.ndarray.min(mrimg)
            
                #for training data in pelvicSegRegH5
        elif opt.how2normalize== 4:
            maxV, minV = np.percentile(mrimg, [99.2 ,1])
            print 'maxV is: ',np.ndarray.max(mrimg)
            mrimg[np.where(mrimg>maxV)] = maxV
            print 'maxV is: ',np.ndarray.max(mrimg)
            mu=np.mean(mrimg)
            std = np.std(mrimg)
            mrimg = (mrimg - mu)/std
            print 'maxV,',np.ndarray.max(mrimg),' minV, ',np.ndarray.min(mrimg)
        
#         print 'maxV,',np.ndarray.max(mrimg),' minV, ',np.ndarray.min(mrimg)
#         labelimg=labelimg[startS:endS+1,:,:]
        
#         dim2_start = 35
#         dim2_end = 235
#         dim3_start = 80
#         dim3_end = 192
#         mrimg = mrimg[:,dim2_start:dim2_end,dim3_start:dim3_end] #attention region
#         labelimg = labelimg[:,dim2_start:dim2_end,dim3_start:dim3_end] #attention region
        
        
        fileID = filename[0:len(filename)-7]
        print 'fileID is: ', fileID
#         contourMatSeg = obtain3DContours(labelimg)
#         volOut = sitk.GetImageFromArray(contourMatSeg)
#         sitk.WriteImage(volOut,'contour_sub{}.nii.gz'.format(ind))
#         print 'contour saved'
        cubicCnt=extractMsPatches4OneSubject(mrimg,labelimg,fileID,dFA,step,rate)
        print '# of patches is ', cubicCnt
        
        tmpMat=mrimg
        tmpLabel=labelimg
        #reverse along the 1st dimension 
        mrimg=mrimg[tmpMat.shape[0]-1::-1,:,:]
        labelimg=labelimg[tmpLabel.shape[0]-1::-1,:,:]
        fileID = filename[0:len(filename)-7]+'_flip1'
        cubicCnt=extractMsPatches4OneSubject(mrimg,labelimg,fileID,dFA,step,rate)
#         #reverse along the 2nd dimension 
#         mrimg=mrimg[:,tmpMat.shape[1]-1::-1,:]
#         labelimg=labelimg[:,tmpLabel.shape[1]-1::-1,:]
#         fileID='%d_flip2'%ind
#         cubicCnt=extractMsPatches4OneSubject(mrimg,labelimg,fileID,dFA,step,rate)
#         #reverse along the 2nd dimension 
#         mrimg=mrimg[:,:,tmpMat.shape[2]-1::-1]
#         labelimg=labelimg[:,:,tmpLabel.shape[2]-1::-1]
#         fileID='%d_flip3'%ind
#         cubicCnt=extractMsPatches4OneSubject(mrimg,labelimg,fileID,dFA,step,rate)
if __name__ == '__main__':     
    main()
