#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:36:47 2017

@author: pmacias
"""
import os 
import subprocess
import numpy as np
import SimpleITK as sitk
import tempfile

import glob


  

_SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}

def is_exe(fpath):
    """
    check if the program is executable
    """
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

def check_executable(program):
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None

def find_executable(name, path, check_on_system = False):#cutre workaround
    print "Finding for "+name+" in "+path+"\n"
    #path = os.path.expanduser('~')
    for root, dirs, files in os.walk(path):
        if name in files:
            program = check_executable(os.path.join(root, name))
            if program is not None:
                return  program
    if check_on_system:
        raise Exception("There is not execuable "+name)
    else:
        new_seek_in = os.path.expanduser('~')#TODO is multiplatform?
        print ("It was impossible to find the function "+name+" on "
        +path+". It will be seeked on "+new_seek_in+"\n")
        return find_executable(name, os.path.abspath(os.path.expanduser('~')+os.sep), check_on_system=True) # Avoid infinite loop

def find_images(initialpath,imageFormats=['.mhd','.hdr']):
    images=[]
    for root, dirs, files in os.walk(initialpath):
        for name in files:
            for formats in imageFormats:
                if formats in name:
                    images.append(os.path.join(root, name))
    return images

def calling_external(exter_fun_path,inputs):
    for i in inputs:
        exter_fun_path=exter_fun_path+' '+str(i)
    process = subprocess.Popen(exter_fun_path, shell=True)
    process.wait()  


def name_output(name_func,inputimage,finalformat=None,pattern = None, save= True):
    """
    Save:
    True for the input same place
    False for the temp
    otherwise must be a route
    """
    
    file_name,ff = os.path.splitext(inputimage)
    file_name_real = file_name.split(os.sep)[-1] if pattern is None else pattern
    if isinstance(save, bool):
        file_name = file_name if save else os.path.join(tempfile.gettempdir(),file_name_real)
    elif os.path.isdir(save):
        print 'SAVE',file_name_real
        file_name = os.path.join(save,file_name_real)
    else:
        raise Exception(save+' is not a path for the refult of filter '+name_func)
    if finalformat is not None:
        ff = finalformat
    
    return  file_name+'_'+name_func+ff
    


def argus2SimpleITK(path):
  """
  Transforms Argus/IDL image types to SimpleITK usual
  """
  typemap = {"unsigned integer":'u2', "signed integer":'i2', "unsigned long":'u4',
 "signed long":'i4', "short float":'f4', "long float":'f8', "byte":np.uint8}
   
  hdrMap = {};
  #Get fields to open raw as image
  with open(path) as f:
    for line in f:
      splitted = line.lower().split(':=') #All to lower case in order to avoid problems
      if(len(splitted) > 1):
        hdrMap[splitted[0].strip()] = splitted[1].strip()


#  print hdrMap['number format']
  rawFile = os.path.join(os.path.dirname(path), hdrMap['name of data file'])
  if os.path.exists(rawFile)==False:
      rawFile=os.path.join(os.path.dirname(path),  os.path.splitext(hdrMap['name of data file'])[0].upper()+os.path.splitext(hdrMap['name of data file'])[1])
  #TODO sometimes matrixsize[3] doesnt exists. Check for NumberTotalImages within hdrMap
  if 'matrix size [3]' in hdrMap:
      dims = (int(hdrMap['matrix size [2]']), int(hdrMap['matrix size [1]']),int( hdrMap['matrix size [3]']))
  elif 'NumberTotalImages' in hdrMap:
      dims = (int(hdrMap['matrix size [2]']), int(hdrMap['matrix size [1]']),int( hdrMap['NumberTotalImages']))
   
  spacing = ( float(hdrMap['scaling factor (mm/pixel) [2]']), float( hdrMap['scaling factor (mm/pixel) [1]']), float( hdrMap['scaling factor (mm/pixel) [3]']) )
  offset = (float(hdrMap['offset [2]']), float(hdrMap['offset [1]']), float(hdrMap['offset [3]']))
  endian = hdrMap['imagedata byte order']
  endiannes = '>' if  endian == 'bigendian' else '<'
  
#  print "Dims: ",dims, "Spacing: " ,spacing, "Offset: ",offset
  dataType = np.int16;
  if(int(hdrMap['number of bytes per pixel']) == 1):
    dataType = np.dtype(typemap["byte"])
  else:
    dataType = endiannes + typemap[hdrMap['number format']]
    
    
  arrayImage = np.fromfile(rawFile, dtype = dataType)
  dims = list(reversed(dims))
  arrayImage = arrayImage.reshape(dims, order = 'C') 

  itkImage = sitk.GetImageFromArray(arrayImage)
  itkImage.SetOrigin(offset)
  itkImage.SetSpacing(spacing)
  
  
  

  return itkImage


def hdr_to_mhd(pathimages):
    images=find_images(pathimages,imageFormats=['.hdr'])
    for i in images:
        image=argus2SimpleITK(i)
        pathfinal=os.path.join(os.path.abspath(os.path.join(i, os.pardir)),os.path.basename(os.path.normpath(i)).split('.')[0]+'.mhd')
        sitk.WriteImage(image,pathfinal)

def get_3D_from_2D_itk_images(stack_2D_itk_images, ref_img_in_stack = 0 ):
    ref_img = stack_2D_itk_images[ref_img_in_stack]
    img_out = sitk.Image(ref_img.GetSize() + (len(stack_2D_itk_images),), ref_img.GetPixelID() )
    
    for i,im in enumerate(stack_2D_itk_images):
        im_vol = sitk.JoinSeries(im)
        img_out = sitk.Paste(img_out, im_vol, im_vol.GetSize(), destinationIndex=[0,0,i])
    return img_out

        
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def crop_to_boundind_box_lung_mask(orginal_image, mask):
    h,w,d = orginal_image.GetSize()
    box_filter = sitk.LabelShapeStatisticsImageFilter()
    
    box_filter.Execute(mask)
    x_lm_1,y_lm_1,z_lm_1,dx_lm,dy_lm,dz_lm = box_filter.GetBoundingBox(1)
    x_lm_2 = x_lm_1 + dx_lm; y_lm_2 = y_lm_1 + dy_lm; z_lm_2 = z_lm_1 + dz_lm
    
    crop_or = sitk.Crop(orginal_image, [x_lm_1,y_lm_1,z_lm_1], [h - x_lm_2, w - y_lm_2, d - z_lm_2 ])
    crop_mask = sitk.Crop(mask, [x_lm_1,y_lm_1,z_lm_1], [h - x_lm_2, w - y_lm_2, d - z_lm_2 ])
    
    return crop_or, crop_mask 
        

def resample_sitk_image_size(sitk_image, size=None, interpolator=None, fill_value=0):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    size : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1: #  8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not size:
        min_size = orig_size.min()
        new_size = [min_size]*num_dim
    else:
        new_size = [s for s in size]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(),\
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_spacing = orig_size*(orig_spacing/new_size)
    #new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_spacing = [s for s in new_spacing] #  SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()
    
    print('new size', new_size, 'new spacing', new_spacing)

    resampled_sitk_image = resample_filter.Execute(sitk_image, new_size,
                                                   sitk.Transform(), 
                                                   sitk_interpolator, 
                                                   orig_origin, new_spacing,
                                                   orig_direction, fill_value,
                                                   orig_pixelid)

    return resampled_sitk_image

if __name__ == "__main__":
    print 'Functions'
    ref_img = sitk.ReadImage('/tmp/im_itk.mhd')
  
    image = '/media/amunoz/PHE_Studies/LOTE_2-study_5007/Study-5007/406AOC/2.3WEEKS_26JUL14/5007_406AOC_3WEEKSW_Segmentation/5007_406AOC_3WEEKSW_Segmentation_invImage.mhd'
    image_mask = '/media/amunoz/PHE_Studies/LOTE_2-study_5007/Study-5007/406AOC/2.3WEEKS_26JUL14/5007_406AOC_3WEEKSW_Segmentation/5007_406AOC_3WEEKSW_Segmentation_imageLungMaskFill.mhd'
    
    images = glob.glob("/media/amunoz/PHE_Studies/LOTE_2-study_5007/Study-5007/*/*/*_Segmentation/*_invImage.mhd")
    
    for image in images:
        name = image.split(os.sep)[-1][:-26]
        im_itk = sitk.ReadImage(image)
        image_mask = image[:-12]+'imageLungMaskFill.mhd'
        print(image)
        print(image_mask)
        if not os.path.exists(image_mask):
            print('sus muertos', image_mask)
        print("")
        
        im_mask = sitk.ReadImage(image_mask) > 0
        #im_itk.CopyInformation(ref_img)
        im,im_msk = crop_to_boundind_box_lung_mask(im_itk, im_mask)
        sitk.WriteImage(resample_sitk_image_size(im, size=[128,128,64], interpolator='bspline'), os.path.join('/tmp/resized_batch_2_mhd/',name+'_img.mhd') )
        sitk.WriteImage(resample_sitk_image_size(im_msk, size=[128,128,64], interpolator='nearest'),  os.path.join('/tmp/resized_batch_2_mhd/',name+'_mask.mhd'))
    
#        SimpleITK.WriteImage(im,'/tmp/im_itk_crop.nii')
#        SimpleITK.WriteImage(im_msk,'/tmp/im_mask_itk_crop.nii')
    
    #    for i in range(len(images)):
#        im_subject = images[i].split(os.sep)[-1]
#        #mask_subject = masks[i].split(os.sep)[-2]
#        
#        #assert im_subject==mask_subject
#        
#        name = im_subject[:-4]
#        
#        im = sitk.ReadImage(images[i])
#        #msk = sitk.ReadImage(masks[i])
#        #im.SetSpacing([0.087622, 0.087622, 0.087622])
#        #msk.SetSpacing([0.087622, 0.087622, 0.087622])
#        print(name)
#        sitk.WriteImage(resample_sitk_image_size(im, size=[128,128,64], interpolator='bspline'), os.path.join('resized_cropped_monkeys',name+'.mhd') )
#        #sitk.WriteImage(resample_sitk_image_size(msk > 0, size=[128,128,64], interpolator='nearest'),  os.path.join('../iTrain_images_msk',name+'.mhd'))
#        print("")
#        
    

        
