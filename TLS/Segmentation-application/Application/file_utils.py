#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 12:38:07 2019

@author: pmacias
"""

import shutil
import pandas as pd
import os
import glob

import SimpleITK

main_path = '/media/amunoz/PHE_Studies/Lote_6_study_5449/MHDimgs/'
suffix = '_segmented_lungs.mhd'

def complete_seg(masks_folder):
  masks = glob.glob(os.path.join(masks_folder,'*lungs_mask.mhd'))
  print(masks)
  for mask in masks:
    #print ('MASK',mask)
    image_dir, image_file = os.path.split(mask)
    sub = image_file.split('_lungs_mask.mhd')[0]
    result_image = os.path.join(masks_folder, sub+suffix)
    if not os.path.exists(result_image):
      image_original = glob.glob(os.path.join(main_path,sub+'.mhd') )[0]
      
      print(mask)
      print(image_original)
      print('--------------------------------------------------------')
      mask_itk = SimpleITK.ReadImage(mask)
      image_itk = SimpleITK.ReadImage(image_original)
      mask_itk.CopyInformation(image_itk)
      masked_image = SimpleITK.Mask(image_itk, mask_itk > 0)
      SimpleITK.WriteImage(masked_image, result_image)
      
def copy_if_not_exits(src_folder, dest_folder):
  images = glob.glob(os.path.join(src_folder,'*'+suffix))
  
  for image in images:
    image_dir, image_file = os.path.split(image)
    image_name,extension = os.path.splitext(image_file)
    coincidence = glob.glob(os.path.join(dest_folder, image_file))
    n_coincidence = len(coincidence)
    if n_coincidence == 0:
      print('no coincidence for', image_file, image_name)
      folder_name = 'Seg_Prev_Res_'+image_name.split(os.path.splitext(suffix)[0] )[0]
      folder_to_copy = os.path.join(image_dir, folder_name)
      print(folder_to_copy,os.path.exists(folder_to_copy))
      final_destination = os.path.join(dest_folder,folder_name)
      if os.path.exists(final_destination):
        shutil.rmtree(final_destination)
      shutil.copytree(folder_to_copy, final_destination)
      print('Image', image)
      files = glob.glob(image.split(suffix)[0]+'*')
      for f in files:
        print(f)
        shutil.copy(f, os.path.join(dest_folder,os.path.split(f)[1]))
    elif n_coincidence  == 1:
      print('coincidence')
      print(image)
      print(coincidence[0])
    else:
      print('WTF',image)
    print('------------')
    
def check_ids(ids_file, images_folder):
  summ = pd.read_csv(ids_file)
  ids = summ.ID.values
  lack = 0
  coincidences = 0
  no_ways = 0
  lacks_list = []
  for idi in ids:
    #print(idi)
    coincidence = glob.glob(os.path.join(images_folder,idi+suffix))
    n_coincidence = len(coincidence)
    if n_coincidence == 0:
      lack+=1
      lacks_list.append(idi)
    elif n_coincidence == 1:
      coincidences += 1
    else:
      no_ways += 1
  print('coincidences',coincidences, 'lacks',lack, 'no_ways',no_ways)
  print(lacks_list)
  return lacks_list

def check_duplicates(folder):
  images = glob.glob(folder+'*'+suffix)
  cons = 0
  
  for image in images:
    image_dir, image_file = os.path.split(image)
    image_name,extension = os.path.splitext(image_file)
    sub = image_file.split(suffix)[0][:-1]
    coincideces = glob.glob(folder+sub+'*segmented_lungs.mhd')
    if len(coincideces) > 1:
      print(sub)
      print(coincideces)
      print('----------------------')
      cons += 1
    
  print('TOTAL',cons)
  
      
      
      
      
      
  
  
    

if __name__ == "__main__":
#   copy_if_not_exits('/media/pmacias/DATA2/RESULTS_DAVID/Results_PHE_B6/', '/media/pmacias/DATA2/Results_PHE_B6/')
#   complete_seg('/media/pmacias/DATA2/RESULTS_DAVID/Results_PHE_B6/')
#    check_ids('/home/pmacias/Descargas/Batch6-Sheet1.csv', '/media/pmacias/DATA2/Results_PHE_B6/')
    check_duplicates('/media/pmacias/DATA2/Results_PHE_B6/')
#    
#    summ = pd.read_csv('/home/pmacias/Descargas/Batch6-Sheet1.csv')
#    sub_summ  = summ[summ.Stomach_Included == 'N'][summ.Lungs_Correctly_Chosen == 'Y'][summ.Finish == 'Y']
#    ids = sub_summ.ID.values
#    main_path = '/media/amunoz/PHE_Studies/Lote_6_study_5449/MHDimgs/Results'
#    results = '/media/pmacias/DATA2/Results_PHE_B6'
#    
#    for idi in ids:
#      to_copy = glob.glob(os.path.join(main_path,idi+'*'))
#      print(to_copy)
#      for f in to_copy:
#        print f
#        image_dir, image_file = os.path.split(f)
#        shutil.copy(f,os.path.join(results, image_file))
#      