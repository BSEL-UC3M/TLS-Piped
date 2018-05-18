#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:38:44 2017

@author: pmacias
"""

import os
from Segmentation import RESULTS_FOLDER, Pipe
from Filters import Apply_Median,Choose_Lungs, Otsu_Threshold, Labelize,Extract_Trachea_FM, Hu_Threshold
from Filters import Binarize, Mask_Neg, Dilation,find_trachea_init_ITK, File_Holes, Erode, Mask
from UsefullFunctions import find_images, argus2SimpleITK

#TODO Evolve this with "REFACTORING" in order to include automatically each 
#  new filter within the module
SAVE_OPTIONS_DEF = {"Median":True , "Otsu":True, 'Labelize': True,
                    'Choose':True, 'Trachea':True}


def config_files(input_image_path, results_dir):
    image_dir, image_file = os.path.split(input_image_path)
    image_name,extension = os.path.splitext(image_file)
    results_dir = results_dir if results_dir is not None else image_dir
    inter_rslts_fld = os.path.join(results_dir, RESULTS_FOLDER+"_"+ image_name) #intermediate results
    
    if not os.path.exists(inter_rslts_fld):
        os.makedirs(inter_rslts_fld)
    image_trick = os.path.join(inter_rslts_fld,image_file)
    inter_rslts_fld = os.path.join(inter_rslts_fld,image_name)
    
    return results_dir, image_trick, image_file
    
def mouse_lung_segmentation(input_image_path,results_dir = None):
    results_dir, image_trick, image_file = config_files(input_image_path, results_dir)
        
    lungs_chooser = Choose_Lungs(save_img=(image_trick, True)) #Initialize out to allow pipeline bifurcations
                    
    lungs_chooser_pipeline = [Apply_Median(save_img=(image_trick, True)),
                              Hu_Threshold(ct_HUnits_error=275.0,inside_val = 1.0, save_img=(image_trick, True)),
                              Labelize(save_img=(image_trick, True)),
                              lungs_chooser,
                              Binarize()]
                              
    lungs_chooser_workflow = Pipe('Workflow for extracting mice lungs', lungs_chooser_pipeline)
    lungs_chooser_workflow.execute(input_image_path)
    
    trachea_extraction_pipeline = [ Extract_Trachea_FM(intial_postion = find_trachea_init_ITK(lungs_chooser_workflow.output_path_and_image.path), time_Step = 0.5, 
                                                       fixed_thr=1.0, trachea_expected_perimeter= Extract_Trachea_FM.MICE_EXPECTED_PERIMETER,save_img=(image_trick,True)),
                                  Binarize(),Dilation(radius=2, foreground = 255) ]
    
    trachea_extraction_workflow = Pipe('Workflow for extracting the mice airways', trachea_extraction_pipeline)
    trachea_extraction_workflow.execute(lungs_chooser_workflow.filters_list[0])
    
    filling_pipe = [Mask_Neg(trachea_extraction_workflow.filters_list[-1].output_path_and_image.image,save_img=(image_trick, True)),
                    File_Holes(radius=(2,2,2), threads=7), Erode(save_img=(os.path.join(results_dir,os.path.splitext(image_file)[0]+"_lungs_mask.mhd"), False))]
    filling_workflow = Pipe('Filling Holes', filling_pipe)
    filling_workflow.execute(lungs_chooser_workflow.filters_list[-1])
    
    Mask(filling_workflow.output_path_and_image.image, save_img=(os.path.join(results_dir,os.path.splitext(image_file)[0]+"_segmented_lungs.mhd"),False)).execute(input_image_path)


def pamplona_lungs_segmentation(input_image_path, results_dir = None):
    results_dir, image_trick, image_file = config_files(input_image_path, results_dir)
        
    lungs_chooser = Choose_Lungs(save_img=(image_trick, True)) #Initialize out to allow pipeline bifurcations
    
    
    lungs_chooser_pipeline = [Apply_Median(save_img=(image_trick, True), radius=4 ),
                              Hu_Threshold(thrPlusOne=-1,ct_HUnits_error=30,inside_val = 1.0, save_img=(image_trick, True)),
                              Labelize(save_img=(image_trick, True)),
                              lungs_chooser,
                              Binarize()]

                                                        
    lungs_chooser_workflow = Pipe('Workflow for extracting pamplona lungs', lungs_chooser_pipeline)
    lungs_chooser_workflow.execute(input_image_path)
    
    trachea_extraction_pipeline = [ Extract_Trachea_FM(intial_postion = find_trachea_init_ITK(lungs_chooser_workflow.output_path_and_image.path),
                                                      trachea_expected_perimeter= Extract_Trachea_FM.MICE_EXPECTED_PERIMETER,save_img=(image_trick,True),
                                                      variable_thr= (-625,100), time_Step=0.8, fixed_thr=0.8 ),
                                                      Binarize(), Dilation(radius=3, foreground = 255) ]
    
    trachea_extraction_workflow = Pipe('Workflow for extracting the mice airways', trachea_extraction_pipeline)
    trachea_extraction_workflow.execute(lungs_chooser_workflow.filters_list[0])
    
    
    filling_pipe = [Mask_Neg(trachea_extraction_workflow.filters_list[-1].output_path_and_image.image,save_img=(image_trick, True)),
                File_Holes(radius=(4,4,4), threads=7, iterations=35),
                Erode(radius=2, save_img=(os.path.join(results_dir,os.path.splitext(image_file)[0]+"_lungs_mask.mhd"), False))]
                    
    filling_workflow = Pipe('Filling Holes', filling_pipe)
    filling_workflow.execute(lungs_chooser_workflow.filters_list[-1])
    
    Mask(filling_workflow.output_path_and_image.image, 
         save_img=(os.path.join(results_dir,os.path.splitext(image_file)[0]+"_segmented_lungs.mhd"),False)).execute(input_image_path)    
 
    

if __name__ == "__main__":
    import glob
    images = glob.glob('../../../CT_pulmones/*.mhd')
    results = 'Results'
    for image in images:
        print image
        pamplona_lungs_segmentation(image, results_dir=results)
    
