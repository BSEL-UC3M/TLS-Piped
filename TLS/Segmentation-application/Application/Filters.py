#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:06:50 2017

@author: pmacias
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 09:24:51 2017

@author: biig, pmacias
"""
import numpy as np
import SimpleITK 
import os
import pandas as pd
from collections import Iterable

from scipy.stats import halfnorm, expon


from Segmentation import A_Filter, FILTER_TYPE, UTILITIES_DIR, External_Filter,Simple_ITK_Filter, Fiji_Filter, FIJI_DIR, FIJI_MACROS_DIR


class Apply_Median(External_Filter):
    '''
    Median filter
    Inputs:
        Name input volume
    Output:
        Name of the output volume
    '''
    def __init__(self, location_path = UTILITIES_DIR,save_img =(None,True), 
                 radius=1,  execute_if_exits = True):
        A_Filter.__init__(self, "apply_median", FILTER_TYPE.PRE_FILTER, 
                          location_path, built_in=False, save_img= save_img, execute_if_exits = execute_if_exits)
        self.radius = radius
    
    def set_params(self):
        self.params = [self.radius]
     

class Otsu_Threshold(External_Filter):
    '''
    Otsu Threshold    
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
    '''  
    def __init__(self, location_path = UTILITIES_DIR,save_img =(None,True)):
        A_Filter.__init__(self, 'hu_threshold', FILTER_TYPE.PRE_FILTER, location_path, built_in=False, save_img = save_img)
    
    def set_params(self):
        self.params = []
        

class Labelize(External_Filter):
    '''
    Labelize filter
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
    '''
    def __init__(self,location_path = UTILITIES_DIR,save_img =(None,True)):
        A_Filter.__init__(self, 'labelize', FILTER_TYPE.PRE_FILTER, location_path, built_in=False, save_img = save_img)
        
    def set_params(self):
        self.params = []
        


class Choose_Lungs(External_Filter):
    '''
    Choose lungs
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
    '''
    
    def __init__(self,location_path = UTILITIES_DIR,save_img =(None,True)):
        A_Filter.__init__(self, 'choose_lungsV2', FILTER_TYPE.PRE_FILTER, location_path, built_in=False, save_img = save_img)
    
    def set_params(self):
        self.params = []
    
class Choose_Lungs2(External_Filter):
    '''
    Choose lungs
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
    '''
    
    def __init__(self,label = 4, location_path = UTILITIES_DIR,save_img =(None,True)):
        A_Filter.__init__(self, 'choose_lungsV2', FILTER_TYPE.PRE_FILTER, location_path, built_in=False, save_img = save_img, params_order=False)
        self.label = label
    
    def set_params(self):
        self.params = [self.label]
    




class Extract_Trachea_FM(External_Filter):
    """
    Initial Position a tuple with the 3D position or a method to get it
    """
    MACAQUE_EXPECTED_PERIMETER = (11.5,33.5)
    MICE_EXPECTED_PERIMETER = (1.2,4)

    def __init__(self,location_path = UTILITIES_DIR, save_img = (None,True),intial_postion = None,
                 trachea_expected_perimeter = MACAQUE_EXPECTED_PERIMETER,time_Step=0.8, variable_thr =(-625, -300),fixed_thr=1.4):
        A_Filter.__init__(self, 'air_ext_fm', FILTER_TYPE.TRAQUEA_EXTRACTION, location_path, built_in=False, save_img = save_img)
        self.time_step = time_Step
        self.variable_thr = variable_thr
        self.fixed_threshold=fixed_thr
        self.trachea_expected_perimeter = trachea_expected_perimeter
        self.initial_postion = intial_postion
    
    
    def set_initial_position(self):
        pass
    
    def set_params(self):
        self.params  = [self.time_step, self.output_path_and_image.path, self.variable_thr[0], self.variable_thr[1], #TODO var_thr[1] is fixed_the and fixed_thr is prop_sigma
                        self.fixed_threshold, self.initial_postion[0], self.initial_postion[1], self.initial_postion[2]]

class Dilation(Simple_ITK_Filter):
    '''
    Dilation
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
    '''
    def __init__(self, radius = 1,kernel = SimpleITK.sitkBall, background = 0.0,
                 foreground = 1.0, boundaryToForeground = False,save_img =(None,True) ):
        A_Filter.__init__(self,'Dilation', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.radius = radius
        self.kernel = kernel
        self.background = background 
        self.foreground = foreground
        self.boundaryForegorund = boundaryToForeground
    
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.BinaryDilateImageFilter()
        self.simple_itk_filter.SetKernelRadius(self.radius)
        self.simple_itk_filter.SetKernelType(self.kernel)
    def set_params(self):
        self.params = [self.background, self.foreground, self.boundaryForegorund]

def find_trachea_init_ITK(image_input, trachea_start_slice = 1):
    """
    This functions needs a trachea at the "trachea_start_slice" slice of image_input
    """
    class Find_Position(External_Filter):
        
        def __init__(self,location_path = UTILITIES_DIR,save_img =(None,True), start_slice = trachea_start_slice):
            A_Filter.__init__(self, 'find_trachea', FILTER_TYPE.OTHERS, location_path, built_in=False, save_img = save_img)
            self.start_slice = start_slice
        
        def set_params(self):
            self.params = [self.start_slice]
            
    find_postion = Find_Position(start_slice=trachea_start_slice)
    find_postion.execute(image_input)
    with open(find_postion.output_path_and_image.path) as cords_file:
        return [int(coordinate) for coordinate in cords_file.readline().rstrip('\r\n').split(',')]
 
       
class Binarize(Simple_ITK_Filter):
    def __init__(self, lower_th = 1,upper_th= 255, inside_val = 255, outside_val = 0, save_img =(None,True)):
        A_Filter.__init__(self,'Binarize', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.lower_th = lower_th
        self.upper_th= upper_th
        self.inside_val = inside_val
        self.outside_val = outside_val
        
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.BinaryThresholdImageFilter()
        
    def set_params(self):
        self.params = [self.lower_th, self.upper_th, self.inside_val, self.outside_val]


class Mask(Simple_ITK_Filter):
    def __init__(self, mask_image, outside_val = 0, save_img =(None,True)):
        A_Filter.__init__(self,'Mask', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.mask_image = mask_image
        self.outside_val = outside_val
    
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.MaskImageFilter()
        
    def set_params(self):
        image = self.mask_image if isinstance(self.mask_image, SimpleITK.Image) else SimpleITK.ReadImage(self.mask_image)
        image.CopyInformation(self.input_path_and_image.image)
        self.params = [image, self.outside_val]

        
class Mask_Neg(Simple_ITK_Filter):
    def __init__(self, mask_image, save_img =(None,True)):
        A_Filter.__init__(self,'Mask', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.mask_image = mask_image 
        
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.MaskNegatedImageFilter()
        
    def set_params(self):
        image = self.mask_image if isinstance(self.mask_image, SimpleITK.Image) else SimpleITK.ReadImage(self.mask_image)
        image.CopyInformation(self.input_path_and_image.image)
        self.params = [image]


class File_Holes(Simple_ITK_Filter):
    def __init__(self, radius = 1, iterations = 20, foregorund = 255, background = 0, threads  = 4, majority_thr = 1, save_img =(None,True)):
        A_Filter.__init__(self, 'Filling', FILTER_TYPE.POST_FILTER, built_in=True, save_img=save_img )
        self.radius = radius
        self.background = background
        self.foregorund = foregorund
        self.iterations = iterations
        self.threads = threads
        self.majority_thr = majority_thr
        
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.VotingBinaryIterativeHoleFillingImageFilter()
        self.simple_itk_filter.SetNumberOfThreads(self.threads)
        
    def set_params(self):
        self.params = [self.radius, self.iterations, self.majority_thr, self.foregorund, self.background]

class Erode(Simple_ITK_Filter):
    def __init__(self, radius = 1, kernel = SimpleITK.sitkBall, backgorund = 0.0, foregorund = 255.0, boundary_foreground = False, save_img =(None,True)):
        A_Filter.__init__(self, 'Erode', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.radius = radius
        self.kernel = kernel
        self.backgorund = backgorund
        self.foregorund = foregorund
        self.boundary_foreground = boundary_foreground 
        
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.BinaryErodeImageFilter()
        self.simple_itk_filter.SetKernelRadius(self.radius)
        self.simple_itk_filter.SetKernelType(self.kernel)
    def set_params(self):
        self.params = [self.backgorund, self.foregorund, self.boundary_foreground]
       

class Hu_Threshold(A_Filter):
    '''
    Hu Threshold
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
''' 
    def __init__(self, iniTh=0.0,thrPlusOne = -600, huDiff=0.01, ct_HUnits_error=290, outside_val = 0.0, inside_val =255.0, save_img =(None,True)):
        A_Filter.__init__(self, 'Hu_trheshold', FILTER_TYPE.PRE_FILTER, built_in=True, save_img=save_img)
        self.iniTh = iniTh
        self.huDiff = huDiff
        self.errFac = ct_HUnits_error
        self.inside_val = inside_val
        self.outside_val = outside_val
        self.thrPlusOne = thrPlusOne

    
    def execute(self,inputimage, output = None):
        self.to_interface(inputimage)
        img = SimpleITK.GetArrayFromImage(self.input_path_and_image.image)
        img[img < -1024] = -1024
        def getHuMeans(image, thr):
            #Returns the means under and upper a thr of an array
            return ( np.mean(image[image < thr]), np.mean(image[image >= thr]) )

        while(np.abs(self.iniTh - self.thrPlusOne) > self.huDiff):
            self.iniTh = self.thrPlusOne
            uNuBMeans = getHuMeans(img, self.thrPlusOne)
            self.thrPlusOne = (uNuBMeans[0] + uNuBMeans[1]) / 2.0
            print self.iniTh,self.thrPlusOne,uNuBMeans[0],uNuBMeans[1]
        
        self.thrPlusOne += self.errFac
        print self.iniTh,self.thrPlusOne,uNuBMeans[0],uNuBMeans[1]
    
        if self.thrPlusOne < 0:    
            img[img >= self.thrPlusOne] = self.outside_val    
            img[img < self.thrPlusOne] = self.inside_val
        else:
            img[img <= self.thrPlusOne] = self.inside_val
            img[img > self.thrPlusOne] = self.outside_val    
        
        self.output_path_and_image.image = SimpleITK.GetImageFromArray(img)
        self.output_path_and_image.image.CopyInformation(self.input_path_and_image.image)
        self.output_path_and_image.path = self.output_path_and_image.path if output is None else output
        SimpleITK.WriteImage(self.output_path_and_image.image,self.output_path_and_image.path)
        return self



class SRM(Fiji_Filter):
    def __init__(self, location_path= FIJI_DIR, macro_path = os.path.join(FIJI_MACROS_DIR, 'SRM.ijm'),
                 save_img = (None, True), q = 25, three_dim = True, averages = False):
        A_Filter.__init__(self, 'ImageJ-linux64', FILTER_TYPE.OTHERS, location_path, built_in=False, save_img = save_img)
        self.q = q
        self.three_dim = three_dim
        self.averages = averages
        self.macro_path = macro_path
    
    def str_three_dim(self, three_dim = None):
        if three_dim is not None:
            self.three_dim = three_dim
        if isinstance(self.three_dim, bool):
            return '3d' if self.three_dim is True else ""
        return self.three_dim
    
    def str_averages(self, averages = None):
        if averages is not None:
            self.averages = averages
        if isinstance(self.averages, bool):
            return 'showaverages' if self.averages is True else ""
        return self.averages
    
    def set_params(self):
        p1 = 'image_path='
        p2 = 'q='
        p3 = 'averages='
        p4 = 'threeD='
        p5 = 'image_out_path='
        self.params = [self.macro_path,
                       "'"+p1+'"'+self.input_path_and_image.path+'"'+"," +p2+str(self.q)+","+p3+
                       '"'+self.str_averages()+'"'+","+p4+'"'+self.str_three_dim()+'"'+","+p5+'"'+self.output_path_and_image.path+'"'+"'" ]
            

class Keep_N_Objects(A_Filter):
    """
    Input should be a label image
    order 0 for descending order
    works fine up to 2^16 labels
    """
    SIZE = 'Physical Size'
    FERET_DIAMETER = 'Feret Diameter'
    PERIMETER = 'Perimeter'
    ELONGATION = 'Elongation'
    SPHERICAL_DIAMETER = 'Spherical Diameter' 
    SPHERICAL_RADIUS = 'Spherical Radius'
    FLATNESS = 'Flatness'
    PIXELS = 'Number of Pixels'
    PIXELS_ON_BORDER = 'Number of pixels on border'
    PERIMETER_ON_BORDER = 'Perimeter on border'
    PERIMETER_ON_BORDER_RATIO = 'Perimeter on border ratio'
    ROUNDNESS = 'Roundness'
    CENTROID = 'Centroid'
    CENTOID_ON_PIXELS = 'Centroid on pixels dimensions'
    
    
    def __init__(self, n_objects = 3, feature = SIZE, order = 0, bck_val = 0, save_img =(None,True) ):
        A_Filter.__init__(self, 'Keep-Objects', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.n_objects = n_objects
        self.feature = feature
        self.order = order
        self.bck_val = bck_val
        
    def execute(self,inputimage, output = None):
        self.to_interface(inputimage)
        shape_stats = SimpleITK.LabelShapeStatisticsImageFilter()
        FEATS = {Keep_N_Objects.SIZE:shape_stats.GetPhysicalSize,
                 Keep_N_Objects.ELONGATION:shape_stats.GetElongation,
                 Keep_N_Objects.SPHERICAL_DIAMETER:shape_stats.GetEquivalentSphericalPerimeter,
                 Keep_N_Objects.SPHERICAL_RADIUS:shape_stats.GetEquivalentSphericalRadius,
                 Keep_N_Objects.FERET_DIAMETER:shape_stats.GetFeretDiameter,
                 Keep_N_Objects.FLATNESS:shape_stats.GetFlatness, 
                 Keep_N_Objects.PIXELS:shape_stats.GetNumberOfPixels, 
                 Keep_N_Objects.PIXELS_ON_BORDER:shape_stats.GetNumberOfPixelsOnBorder,
                 Keep_N_Objects.PERIMETER:shape_stats.GetPerimeter, 
                 Keep_N_Objects.PERIMETER_ON_BORDER:shape_stats.GetPerimeterOnBorder,
                 Keep_N_Objects.PERIMETER_ON_BORDER_RATIO:shape_stats.GetPerimeterOnBorderRatio,
                 Keep_N_Objects.ROUNDNESS:shape_stats.GetRoundness}
        shape_stats.Execute(self.input_path_and_image.image, self.bck_val,
                            self.feature == Keep_N_Objects.FERET_DIAMETER, self.feature == Keep_N_Objects.PERIMETER)
        measures = np.array([FEATS[self.feature](l) for l in shape_stats.GetLabels() ])
        sorted_selec_labels =  np.array(shape_stats.GetLabels())[measures.argsort()[:self.n_objects]] if self.order != 0 else np.array(shape_stats.GetLabels())[measures.argsort()[-self.n_objects:]]
        
        type_im = SimpleITK.sitkUInt8 if np.max(sorted_selec_labels)/256 < 1 else SimpleITK.sitkUInt16
        print type_im
        self.output_path_and_image.image = SimpleITK.Image(self.input_path_and_image.image.GetSize(),type_im)
        self.output_path_and_image.image.CopyInformation(self.input_path_and_image.image)
        print sorted_selec_labels
        for l in sorted_selec_labels:
            masked = self.input_path_and_image.image == l if type_im == SimpleITK.sitkUInt8 else SimpleITK.Cast(self.input_path_and_image.image == l, SimpleITK.sitkUInt16)
            self.output_path_and_image.image += masked*l
            
        self.output_path_and_image.path = self.output_path_and_image.path if output is None else output
        SimpleITK.WriteImage(self.output_path_and_image.image,self.output_path_and_image.path)
         
        return self

class Trim_Stomach(A_Filter):
    def __init__(self, ct_image_filter ,save_img =(None,True) ):
        A_Filter.__init__(self, 'Trim_stomach', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.ct_image_filter = ct_image_filter
            
    def execute(self,inputimage, output = None):
        print('TRIMMING',self.input_path_and_image.path)
        self.to_interface(inputimage)
        stomach_msk = trim_stomach(self.ct_image_filter.output_path_and_image.path, self.input_path_and_image.image)
        self.output_path_and_image.image = SimpleITK.Mask(self.input_path_and_image.image, stomach_msk == 0)
        self.output_path_and_image.path = self.output_path_and_image.path if output is None else output
        SimpleITK.WriteImage(self.output_path_and_image.image,self.output_path_and_image.path)
        return self
      

  


def trim_stomach(image_real_bad_lungs, image_mask_bad_lungs ):
    """
    ITK images as input
    """   
    
    
    shape_stats = SimpleITK.LabelShapeStatisticsImageFilter()
    image_real_bad_lungs = SimpleITK.ReadImage(image_real_bad_lungs) if isinstance(image_real_bad_lungs, str) else image_real_bad_lungs
    
    print('image_mask_bad_lungs',image_mask_bad_lungs)
    image_mask_bad_lungs.CopyInformation(image_real_bad_lungs)
    masked_bad_lungs = SimpleITK.Mask(image_real_bad_lungs, image_mask_bad_lungs > 0, outsideValue=5000)
    
    SimpleITK.WriteImage(masked_bad_lungs, '/tmp/masked_trim_sth_lungs.mhd')
    srm_filt = SRM(q=10, averages=True).execute('/tmp/masked_trim_sth_lungs.mhd')
    srm_image = SimpleITK.ReadImage(srm_filt.output_path_and_image.path) #> 4
    averages = np.unique(SimpleITK.GetArrayFromImage(srm_image))
    
    found_trachea = False
    i = 0
    while found_trachea == False and i < len(averages):
      shape_stats.Execute(srm_image == averages[i])
      bbox = shape_stats.GetBoundingBox(1)
      print('average',averages[i],bbox)
      if bbox[2] == 0 and bbox[5] > 10:
        found_trachea = True
      else:
        i += 1
    
    
    stomach_label = 1
    if i > 0 and found_trachea:
      stomach_label = i - 1
    
    print('STOMACH', stomach_label)
    stomach = srm_image == averages[stomach_label]
    stomach.CopyInformation(image_real_bad_lungs)
    return stomach
    #SimpleITK.WriteImage(srm_image == averages[stomach_label], os.path.join('/media/pmacias/DATA2/stommachs/', 'result_'+str(int(averages[i]))+name))
        
def isolated_object_info(itk_2D_binary_image):
    shape_stats = SimpleITK.LabelShapeStatisticsImageFilter()
    shape_stats.ComputePerimeterOn()
    shape_stats.Execute(SimpleITK.ConnectedComponent(itk_2D_binary_image))
    if shape_stats.GetNumberOfLabels() == 1:
        label = shape_stats.GetLabels()[0]
        centroid = shape_stats.GetCentroid(label)
        return {Keep_N_Objects.ROUNDNESS:shape_stats.GetRoundness(label),
                Keep_N_Objects.CENTROID:centroid,
                Keep_N_Objects.CENTOID_ON_PIXELS:itk_2D_binary_image.TransformPhysicalPointToIndex(centroid),
                Keep_N_Objects.PERIMETER: shape_stats.GetPerimeter(label)}
    else: 
        return None
    
def naive_prob_trachea_init(dic_isolated, centroid, subject = Extract_Trachea_FM.MACAQUE_EXPECTED_PERIMETER): #Centorid could be a 2D point or an image to compute it
    from scipy.spatial.distance import euclidean
    return naive_trachea_init(dic_isolated[Keep_N_Objects.ROUNDNESS],
                       euclidean(centroid, dic_isolated[Keep_N_Objects.CENTROID])/20, np.abs(np.mean(subject) - dic_isolated[Keep_N_Objects.PERIMETER])/ np.abs(subject[1] - subject[0] ) )
    
def get_init_trachea_naive(itk_3D_binary_image, animal_model = Extract_Trachea_FM.MACAQUE_EXPECTED_PERIMETER):
    shape_stats = SimpleITK.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(itk_3D_binary_image)
    centroid = shape_stats.GetCentroid(shape_stats.GetLabels()[0])[:2] #Avoid Z dimension for now
    p = -1; more_likely = None; z = -1
    for i in range(itk_3D_binary_image.GetSize()[2]):
        feats_info = isolated_object_info(itk_3D_binary_image[:,:,i])
        if feats_info is not None:
            p_aux = naive_prob_trachea_init(feats_info, centroid, subject = animal_model)
            if p_aux > p:
                more_likely = feats_info
                p = p_aux
                z = i
    print p
    return list(more_likely[Keep_N_Objects.CENTOID_ON_PIXELS])+[z] if more_likely is not None else None
            
        
    

def naive_trachea_init(sphericity, distance, perimeter):
    rv = halfnorm()
    rv_d = expon(scale = 1/halfnorm.pdf(0)) #roudness is more significant
    sphericity = sphericity if sphericity < 1 else 1.0
    return  rv_d.pdf(1 - sphericity) * rv.pdf(perimeter) * rv.pdf(distance)            





def volumeVSslice(image_vol):
  image_vol = image_vol > 0
  stats = SimpleITK.LabelShapeStatisticsImageFilter()
  stats.ComputePerimeterOn()
  stats.ComputeFeretDiameterOn()
  d = {}
  for slc in range(image_vol.GetSize()[-1]):
    feat = 0
    stats.Execute(image_vol[:,:,slc])
    if stats.GetNumberOfLabels() > 0:
      feat = stats.GetPerimeter(1)
    d[slc] = feat
  return d
    
      

def split_blobs(mask, smooth = 1):
  dt = SimpleITK.DanielssonDistanceMapImageFilter()
  dt.SetUseImageSpacing(True)
  distim = dt.Execute(mask == 0)
  
  distimS = SimpleITK.SmoothingRecursiveGaussian(distim, sigma=smooth, normalizeAcrossScale=True) 
  distim = distimS * SimpleITK.Cast(distim > 0, SimpleITK.sitkFloat32)
  
  peakF = SimpleITK.RegionalMaximaImageFilter()
  peakF.SetForegroundValue(1)
  peakF.FullyConnectedOn()
  peaks  = peakF.Execute(distim)
  
  markers = SimpleITK.ConnectedComponent(peaks, fullyConnected=True)
  
  WS = SimpleITK.MorphologicalWatershedFromMarkers(-1 * distim, markerImage=markers, markWatershedLine=True, fullyConnected=True)
  SimpleITK.WriteImage(WS, '/tmp/ws.mhd')
  
        
if __name__ == "__main__":
    import glob
    print 'Filter.py' 
    summ = pd.read_csv('/home/pmacias/Descargas/Batch6-Sheet1.csv')
    sub_summ  = summ[summ.Stomach_Included == 'Y'][summ.Lungs_Correctly_Chosen == 'Y']
    ids = sub_summ.ID.values
    main_path = '/media/amunoz/PHE_Studies/Lote_6_study_5449/MHDimgs/Results/'
    #medians = glob.glob('/home/pmacias/Projects/MonkeysTuberculosis/TLS-Piped/TLS/Segmentation-application/Results_-200/*/*median.mhd')
    for idi in ids[0:1]:
      m = glob.glob(os.path.join(main_path,'*'+idi,'*apply_median.mhd' ) )[0]
      mask = glob.glob(os.path.join(main_path,'*'+idi,'*choose_lungsV2.mhd' ) )[0]
      print(m)
      print(mask)
      print('-------------------------------------')
      mask_2_path = '/media/amunoz/PHE_Studies/Lote_6_study_5449/MHDimgs/Results/Seg_Prev_Res_Wk0_11V_251117_3/Wk0_11V_251117_3_choose_lungsV2.mhd'
      image_2_path = '/media/amunoz/PHE_Studies/Lote_6_study_5449/MHDimgs/Results/Seg_Prev_Res_Wk0_11V_251117_3/Wk0_11V_251117_3_apply_median.mhd'
      m = image_2_path
      mask = mask_2_path
      #mask_itk = SimpleITK.ReadImage(mask_2_path)
      #mask_itk.SetSpacing( [0.255859, 0.255859, 0.625])
      #pos = get_init_trachea_naive(mask_itk, animal_model=Extract_Trachea_FM.MACAQUE_EXPECTED_PERIMETER)
      #print('POSITION', pos)
      Trim_Stomach(SimpleITK.ReadImage(m)).execute(mask)
      #trim_stomach(m,mask)
      
      #trim_stomach()
    #srm = SRM(save_img=('/tmp/srm_filter.mhd', False),q=25)
    #srm.execute("/tmp/flipped.mhd")
    
    #macaques
    #images = glob.glob('/media/pmacias/DATA2/amunoz/LOTE_2-study_5007/Study-5007/*/*/*/*_Segmentation_imageLungMask.mhd')
    #Mice
    
    #images = glob.glob('/media/pmacias/DATA2/amunoz/GSK/CropImagesAnalyzeFlipped/*/*_imageLungMask.mhd')
    
    #image_mask_bad_lungs = SimpleITK.ReadImage("/home/pmacias/Projects/MonkeysTuberculosis/TLS-Piped/TLS/Segmentation-application/Results/Seg_Prev_Res_i2/i2_choose_lungsV2.mhd")
    #image_real_bad_lungs = SimpleITK.ReadImage('/home/pmacias/Projects/MonkeysTuberculosis/TLS-Piped/TLS/Segmentation-application/Results/Seg_Prev_Res_i2/i2_apply_median.mhd')
    

    
    #Labelize().execute(mask_n > 0)
    #Keep_N_Objects(feature=)
    
#    otsu = SimpleITK.OtsuMultipleThresholds(masked_bad_lungs, numberOfThresholds = 2)
#    SimpleITK.WriteImage(otsu , '/tmp/Otsu.mhd')
#    image_mask_bad_lungs = SimpleITK.Mask(image_mask_bad_lungs, otsu > 0, outsideValue=0)
#    SimpleITK.WriteImage(image_mask_bad_lungs > 0 , '/tmp/image_mask_bad_lungs.mhd')
    
    
    
    
#    print('LABELS',shape_stats.GetNumberOfLabels())
#    print('SIZE', shape_stats.GetPhysicalSize(1))
#    er = Erode(radius=5, foregorund=1).execute(image_bad_lungs > 0)
#    print(er.output_path_and_image.path)
    #lab = Labelize().execute('/tmp/image_mask_bad_lungs.mhd')
#    
#    images = glob.glob('/media/pmacias/DATA2/amunoz/GSK/GSK_animals_new/TE0789/*/Segmentation/*_imageLungMask.mhd')
#    for im in images:
#        n = im.split('/')[-3]
#        print n
#        try:
#            img = SimpleITK.ReadImage(im)
#        except:
#            continue
#        img.SetSpacing([0.087622, 0.087622,0.087622])
#        img = SimpleITK.Flip(img, [False,False,True])
#        pos = get_init_trachea_naive(img, animal_model=Extract_Trachea_FM.MICE_EXPECTED_PERIMETER)
#        #pos = get_init_trachea_naive(img, animal_model=Extract_Trachea_FM.MACAQUE_EXPECTED_PERIMETER)
#        SimpleITK.WriteImage(img[:,:,pos[-1]], '/tmp/'+n+'.mhd')
#        print pos
