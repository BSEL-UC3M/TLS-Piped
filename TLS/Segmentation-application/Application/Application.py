#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:38:44 2017

@author: pmacias
"""

import os
from Segmentation import RESULTS_FOLDER, Pipe
from Filters import Apply_Median,Choose_Lungs, Otsu_Threshold, Labelize,Extract_Trachea_FM, Hu_Threshold, Trim_Stomach,Choose_Lungs2
from Filters import Binarize, Mask_Neg, Dilation,find_trachea_init_ITK, File_Holes, Erode, Mask, get_init_trachea_naive
from UsefullFunctions import find_images, argus2SimpleITK
from file_utils import check_ids
import SimpleITK
import pandas as pd
import glob



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
    
    #trachea_extraction_pipeline = [ Extract_Trachea_FM(intial_postion = find_trachea_init_ITK(lungs_chooser_workflow.output_path_and_image.path), time_Step = 0.5, 
     #                                                  fixed_thr=1.0, trachea_expected_perimeter= Extract_Trachea_FM.MICE_EXPECTED_PERIMETER,save_img=(image_trick,True)),
      #                            Binarize(),Dilation(radius=2, foreground = 255) ]
    
    trachea_extraction_pipeline = [ Extract_Trachea_FM(intial_postion = get_init_trachea_naive(lungs_chooser_workflow.output_path_and_image.image, animal_model=Extract_Trachea_FM.MICE_EXPECTED_PERIMETER), time_Step = 0.5, 
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
    
    
    lungs_chooser_pipeline = [Apply_Median(save_img=(image_trick, True), radius=1 ),
                              Hu_Threshold(thrPlusOne=-1,ct_HUnits_error=30,inside_val = 1.0, save_img=(image_trick, True)),
                              Labelize(save_img=(image_trick, True)),
                              lungs_chooser,
                              Binarize()]

                                                        
    lungs_chooser_workflow = Pipe('Workflow for extracting pamplona lungs', lungs_chooser_pipeline)
    lungs_chooser_workflow.execute(input_image_path)
    
    ##Alternative trachea extraction###
    #trachea_extraction_pipeline = [ Extract_Trachea_FM(intial_postion = find_trachea_init_ITK(lungs_chooser_workflow.output_path_and_image.path),
     #                                                 trachea_expected_perimeter= Extract_Trachea_FM.MICE_EXPECTED_PERIMETER,save_img=(image_trick,True),
      #                                                variable_thr= (-625,100), time_Step=0.8, fixed_thr=0.8 ),
       #                                               Binarize(), Dilation(radius=3, foreground = 255) ]
    
    trachea_extraction_pipeline = [ Extract_Trachea_FM(intial_postion = get_init_trachea_naive(lungs_chooser_workflow.output_path_and_image.image, animal_model=Extract_Trachea_FM.MICE_EXPECTED_PERIMETER),
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
 

def macaque_lungs_segmentation(input_image_path, results_dir = None):
    results_dir, image_trick, image_file = config_files(input_image_path, results_dir)
        
    lungs_chooser = Choose_Lungs(save_img=(image_trick, True)) #Initialize out to allow pipeline bifurcations
    lungs_chooser_pipeline = [Apply_Median(radius= 0,save_img=(image_trick, True)),
                              Otsu_Threshold(save_img=(image_trick, True)),
                              Labelize(save_img=(image_trick, True)),
                              lungs_chooser,
                              Binarize()]
    
    
#    lungs_chooser_pipeline = [Apply_Median(save_img=(image_trick, True)),
#                              Hu_Threshold(thrPlusOne=-1,ct_HUnits_error=200,inside_val = 1.0, save_img=(image_trick, True)),
#                              Labelize(save_img=(image_trick, True)),
#                              lungs_chooser,
#                              Binarize()]
                                                        
    lungs_chooser_workflow = Pipe('Workflow for extracting macaques lungs', lungs_chooser_pipeline)
    lungs_chooser_workflow.execute(input_image_path)
    
    trachea_extraction_pipeline = [ Extract_Trachea_FM(intial_postion = find_trachea_init_ITK(lungs_chooser_workflow.output_path_and_image.path), 
                                                       trachea_expected_perimeter= Extract_Trachea_FM.MACAQUE_EXPECTED_PERIMETER,save_img=(image_trick,True)),
                                  Binarize(),Dilation(radius=2, foreground = 255) ]
    
    trachea_extraction_workflow = Pipe('Workflow for extracting the mice airways', trachea_extraction_pipeline)
    trachea_extraction_workflow.execute(lungs_chooser_workflow.filters_list[0])
    
    filling_pipe = [Mask_Neg(trachea_extraction_workflow.filters_list[-1].output_path_and_image.image,save_img=(image_trick, True)),
                    File_Holes(radius=(2,2,2), threads=7), Erode(), 
                    Mask_Neg(trachea_extraction_workflow.filters_list[-1].output_path_and_image.image,save_img=(os.path.join(results_dir,os.path.splitext(image_file)[0]+"_lungs_mask.mhd"), False))]
    filling_workflow = Pipe('Filling Holes', filling_pipe)
    filling_workflow.execute(lungs_chooser_workflow.filters_list[-1])
    
    Mask(filling_workflow.output_path_and_image.image, save_img=(os.path.join(results_dir,os.path.splitext(image_file)[0]+"_segmented_lungs.mhd"),False)).execute(input_image_path)

def macaque_lungs_segmentation2(input_image_path, results_dir = None):
    results_dir, image_trick, image_file = config_files(input_image_path, results_dir)
        
    #lungs_chooser = Choose_Lungs(save_img=(image_trick, True)) #Initialize out to allow pipeline bifurcations
    #lungs_chooser_pipeline = [Apply_Median(radius= 0,save_img=(image_trick, True)),
     #                         Otsu_Threshold(save_img=(image_trick, True)),
      #                        Labelize(save_img=(image_trick, True)),
       #                       lungs_chooser,
        #                      Binarize()]
    
    lungs_chooser = Choose_Lungs(save_img=(image_trick, True)) #Initialize out to allow pipeline bifurcations
    apply_median = Apply_Median(save_img=(image_trick, True), execute_if_exits = False)
    lungs_chooser_pipeline = [apply_median,
                              Otsu_Threshold(save_img=(image_trick, True)),
                              Labelize(save_img=(image_trick, True)),
                              lungs_chooser,
                              Trim_Stomach(apply_median,save_img=(image_trick, True)),
                              Binarize()]           
    lungs_chooser_workflow = Pipe('Workflow for extracting macaques lungs', lungs_chooser_pipeline)
    lungs_chooser_workflow.execute(input_image_path)
       
    trachea_extraction_pipeline = [ Extract_Trachea_FM(intial_postion = find_trachea_init_ITK(lungs_chooser_workflow.output_path_and_image.path), 
                                                       trachea_expected_perimeter= Extract_Trachea_FM.MACAQUE_EXPECTED_PERIMETER,save_img=(image_trick,True)),
                                  Binarize(),Dilation(radius=2, foreground = 255) ]
    
    trachea_extraction_workflow = Pipe('Workflow for extracting the mice airways', trachea_extraction_pipeline)
    trachea_extraction_workflow.execute(lungs_chooser_workflow.filters_list[0])
    
    filling_pipe = [Mask_Neg(trachea_extraction_workflow.filters_list[-1].output_path_and_image.image,save_img=(image_trick, True)),
                    File_Holes(radius=(2,2,2), threads=7), Erode(), 
                    Mask_Neg(trachea_extraction_workflow.filters_list[-1].output_path_and_image.image,save_img=(os.path.join(results_dir,os.path.splitext(image_file)[0]+"_lungs_mask.mhd"), False))]
    filling_workflow = Pipe('Filling Holes', filling_pipe)
    filling_workflow.execute(lungs_chooser_workflow.filters_list[-1])
    
    Mask(filling_workflow.output_path_and_image.image, save_img=(os.path.join(results_dir,os.path.splitext(image_file)[0]+"_segmented_lungs.mhd"),False)).execute(input_image_path)
    

def macaque_lungs_segmentation3(input_image_path, results_dir = None):
    results_dir, image_trick, image_file = config_files(input_image_path, results_dir)
        
    lungs_chooser = Choose_Lungs2(label = 2,save_img=(image_trick, True)) #Initialize out to allow pipeline bifurcations
    apply_median = Apply_Median(save_img=(image_trick, True), execute_if_exits = False)
    lungs_chooser_pipeline = [apply_median,
                              Otsu_Threshold(save_img=(image_trick, True)),
                              Labelize(save_img=(image_trick, True)),
                              lungs_chooser,
                              Trim_Stomach(apply_median,save_img=(image_trick, True)),
                              #Labelize(save_img=(image_trick, True)),
                              #lungs_chooser,
                              #Trim_Stomach(apply_median,save_img=(image_trick, True)),
                              Binarize()]           
    lungs_chooser_workflow = Pipe('Workflow for extracting macaques lungs', lungs_chooser_pipeline)
    lungs_chooser_workflow.execute(input_image_path)
       
    trachea_extraction_pipeline = [ Extract_Trachea_FM(intial_postion = find_trachea_init_ITK(lungs_chooser_workflow.output_path_and_image.path),
                                                       trachea_expected_perimeter= Extract_Trachea_FM.MACAQUE_EXPECTED_PERIMETER,save_img=(image_trick,True)),
                                  Binarize(),Dilation(radius=2, foreground = 255) ]
    
    trachea_extraction_workflow = Pipe('Workflow for extracting the mice airways', trachea_extraction_pipeline)
    trachea_extraction_workflow.execute(lungs_chooser_workflow.filters_list[0])
    
    filling_pipe = [Mask_Neg(trachea_extraction_workflow.filters_list[-1].output_path_and_image.image,save_img=(image_trick, True)),
                    File_Holes(radius=(2,2,2), threads=7), Erode(), 
                    Mask_Neg(trachea_extraction_workflow.filters_list[-1].output_path_and_image.image,save_img=(os.path.join(results_dir,os.path.splitext(image_file)[0]+"_lungs_mask.mhd"), False))]
    filling_workflow = Pipe('Filling Holes', filling_pipe)
    filling_workflow.execute(lungs_chooser_workflow.filters_list[-1])
    
    Mask(filling_workflow.output_path_and_image.image, save_img=(os.path.join(results_dir,os.path.splitext(image_file)[0]+"_segmented_lungs.mhd"),False)).execute(input_image_path)
    


def remove_tmps():
  files = glob.glob('/tmp/1*')
  for f in files:
    os.remove(f)
    

if __name__ == "__main__":
    #images = glob.glob('/home/pmacias/Projects/MonkeysTuberculosis/TLS-Piped/B6_tests/*.mhd')
    #main_path = '/media/amunoz/PHE_Studies/Lote_6_study_5449/MHDimgs/Results/'
    #images = glob.glob('/media/amunoz/PHE_Studies/Lote_6_study_5449/MHDimgs/*.mhd')
    #images = glob.glob('/home/pmacias/Projects/MonkeysTuberculosis/TLS-Piped/TLS/Segmentation-application/Results_-200/*/*median.mhd')
    #images = glob.glob('/media/pmacias/DATA2/amunoz/GSK/GSK_animals_new/TE0789/*/Segmentation/*_invImage.mhd')
    
    
    #summ = pd.read_csv('/home/pmacias/Descargas/Batch6-Sheet1.csv')
    #sub_summ  = summ[summ.Stomach_Included == 'Y'][summ.Lungs_Correctly_Chosen == 'Y']
    #ids = sub_summ.ID.values
    ids = check_ids('/home/pmacias/Descargas/Batch6-Sheet1.csv', '/media/pmacias/DATA2/Results_PHE_B6/')
    main_path = '/media/amunoz/PHE_Studies/Lote_6_study_5449/MHDimgs/'
    results = '/media/pmacias/DATA2/Results_PHE_B6'
    #results = '/media/amunoz/PHE_Studies/Lote_6_study_5449/MHDimgs/Results/'
    fails = []
    #print('IMAGESLEN', len(images))
    #results = 'Results_Test_new_feat'
    ids = ['Wk16_15V_24Th_March_2018_3']
    n_ids = len(ids)
    for i,idi in enumerate(ids):
        remove_tmps()
        image = os.path.join(main_path,idi+'.mhd')
        print(idi)
        print (image)
        #image = SimpleITK.Flip(SimpleITK.ReadImage(image), [False,True,False])
        try:
          macaque_lungs_segmentation3(image, results_dir = results)
        except:
          fails.append({'Name':idi, 'path':image})
          continue
        print('FALTAN----->', n_ids - i + 1,'    ------------' )
        #pamplona_lungs_segmentation(image, results_dir=results) #Pamplona Elastasa pipeline
    
    
    print('FAILS',fails)
    df = pd.DataFrame(fails)
    df.to_csv(os.path.join(results,'failsLacks.csv'))
        
    
