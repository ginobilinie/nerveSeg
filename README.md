# nerveSeg with pytorch

This is a pytorch code for nerve segmentation by using a 2D-Unet or 3D-UNet. Actually, I also upload two other tasks: infant segmentation and 3T/7T reconstruction.

For the nerve-segmentation project: The main entrance is runNerveSeg.py

Steps to run the segmentation task:
1. use dicom2Nii.py to convert the given dicoms data to nii.gz. This is the most difficult step. You have to do it with great caution. First use this copy of code to deal with the inputs, then use it to deal with the labels (as data and labels are given in different folders). 
2. use extractSsPatch4NerveSeg.py to generate the hdf5 format data (we use 2D or 3D patch as a basic training unit, and we use hdf5 to store the data)
3. modify the runNerveSeg.py according to your own settings, such as data paths, hyper-parameter settings.
4. evaluate your method. Will upload later (I have to log into my own PC to download the code)

# nerveSeg with caffe

