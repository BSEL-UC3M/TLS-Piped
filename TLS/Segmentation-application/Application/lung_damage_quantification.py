#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:05:18 2017

@author: biig, pmacias
"""


import sys
import numpy as np
import SimpleITK
import os
import xlrd
import pandas as pd
import subprocess
import logging

separator=os.sep


'''EXTRACTING VOLUMES'''            

imageFormats=['.mhd','.hdr']
QuantificationFolder='Quantification'
NameCSV='LungDamage'
NameColumn="Name"
TreatmentColumn="Treatment"
WeekColumn="Week"

def quantify(pathR,pathMask,pathOriginal,VolumeToMeasure='Vol_Rel_Sick',thresholds = (-1024,-500,-100, sys.maxint),periodTimes=None):
    
    
    pathCSV=extracting_volumes_gsk(pathMask,pathOriginal,thresholds)
    logging.info('.csv file created in %s',pathCSV)
    if periodTimes!=None:
        string=''
        for p in periodTimes:
            string=string+' '+str(p)
        subprocess.check_output('Rscript --vanilla '+pathR+' '+pathCSV+' '+NameColumn+' '+TreatmentColumn+' '+WeekColumn+' '+VolumeToMeasure+' '+string,shell=True)
    else:
        subprocess.check_output('Rscript --vanilla '+pathR+' '+pathCSV+' '+NameColumn+' '+TreatmentColumn+' '+WeekColumn+' '+VolumeToMeasure,shell=True)
    
    logging.info('Waterfalls created and saved in %s',os.path.abspath(os.path.join(pathMask, os.pardir)))
    return (os.path.abspath(os.path.join(pathMask, os.pardir)))

def extracting_volumes_gsk(pathMask,pathOriginal,thresholds = (-1024,-500,-100, sys.maxint)):
        
    [volOriginal,rootOriginal]=reading_volumes(pathOriginal)
    [volMask,rootMask]=reading_volumes(pathMask)
    pathFinal=os.path.join(os.path.abspath(os.path.join(pathMask, os.pardir)),QuantificationFolder)
    
    
    List=[]
    quantified=0
    for original,mask,rootO,rootM in zip(volOriginal,volMask,rootOriginal,rootMask):
        quantified=quantified+1
        nameOriginal=original.split('_')[0]
        treatmentOriginal=original.split('_')[2]
        weekOriginal=(original.split('_')[1]).split('.')[0]
        logging.info('Quantified %s out of %s (%s in process)',str(quantified),str(len(volOriginal)),nameOriginal)
#        print '\rQuantified %d out of %d (%s in process)' %(quantified,len(volOriginal),nameOriginal),
#        sys.stdout.flush()
        originalimage=SimpleITK.ReadImage(rootO+separator+original)
        maskimage=SimpleITK.ReadImage(rootM+separator+mask)
        dictionary1={NameColumn:nameOriginal,TreatmentColumn:treatmentOriginal,WeekColumn:weekOriginal}
        dictionary2=get_Lung_Damage(originalimage, maskimage, thresholds)
        dictionary1.update(dictionary2)
        List.append(dictionary1)
        
    logging.info('Quantified %s out of %s (Quantification finished)',str(quantified+1),str(len(volOriginal)))
#    print '\rQuantified %d out of %d (Finished)      ' %(quantified,len(volOriginal)),
#    sys.stdout.flush()
    listdic=pd.DataFrame(List)
    if os.path.exists(pathFinal)==False:
        os.makedirs(pathFinal)
    listdic.to_csv(pathFinal+separator+NameCSV+'.csv')
    return (pathFinal+separator+NameCSV+'.csv')

def extracting_volumes_macaques(pathMask,pathOriginal,pathExcell,pathFinal,excelcolumns=['ID', 'Week','Study Group'],moreinformation={"Study":5407,"Lote":4},thresholds = (-1024,-500,-100, sys.maxint)):
    
    [volOriginal,rootOriginal]=reading_volumes(pathOriginal)
    [volMask,rootMask]=reading_volumes(pathMask)
    
    ColumnCharact=read_excell(pathExcell,excelcolumns)
    names=ColumnCharact.get(excelcolumns[0])
    weeks=ColumnCharact.get(excelcolumns[1])
    groups=ColumnCharact.get(excelcolumns[2])
    Dead=dead_subjects(names,weeks,groups)
    
    typeOfGroups=[]
    for g in set(groups):
        typeOfGroups.append(str(g))
        typeOfGroups.sort()
        
    List=[] 
    quantified=0
    for group in typeOfGroups:
        for dead in Dead:
            if Dead.get(dead)[1]==group:
                logging.info('Quantified %s out of %s (%s in process)',str(quantified),str(len(volOriginal)),dead)
#                print '\rQuantified %d out of %d (%s in process)' %(quantified,len(volOriginal),dead),
#                sys.stdout.flush()
                dictionary1={"Name":dead,"Group":group,"Week":Dead.get(dead)[0]}
                dictionary1.update(moreinformation)
                dictionary2={Lungs_Fields.VOL_HEAL_PIX:'NA',   Lungs_Fields.VOL_HEAL:'NA', Lungs_Fields.VOL_SOFT_PIX:'NA',
                Lungs_Fields.VOL_SOFT:'NA', Lungs_Fields.VOL_HARD_PIX:'NA',         Lungs_Fields.VOL_HARD:'NA',
                Lungs_Fields.VOL_SICK_PIX:'NA',      Lungs_Fields.VOL_SICK:'NA',    Lungs_Fields.VOL_REL_SOFT:'NA',
                Lungs_Fields.VOL_REL_HARD:'NA', Lungs_Fields.VOL_REL_SICK:'NA'}
                dictionary1.update(dictionary2)
                List.append(dictionary1)
                
        for original,mask,rootO,rootM in zip(volOriginal,volMask,rootOriginal,rootMask):
            nameOriginal=original.split('_')[0]
            weekOriginal=(original.split('_')[1]).split('W')[0]
            groupOriginal=(original.split('_')[2]).split('.')[0]
            logging.info('Quantified %s out of %s (%s in process)',str(quantified),str(len(volOriginal)),nameOriginal)
#            print '\rQuantified %d out of %d (%s in process)' %(quantified,len(volOriginal),nameOriginal),
#            sys.stdout.flush()
            if groupOriginal==group:
                quantified=quantified+1
                originalimage=SimpleITK.ReadImage(rootO+separator+original)
                maskimage=SimpleITK.ReadImage(rootM+separator+mask)
                dictionary1={NameColumn:nameOriginal,TreatmentColumn:groupOriginal,WeekColumn:weekOriginal}
                dictionary1.update(moreinformation)
                dictionary2=get_Lung_Damage(originalimage, maskimage, thresholds)
                dictionary1.update(dictionary2)
                List.append(dictionary1)
                
    logging.info('Quantified %s out of %s (Quantification finished)',str(quantified+1),str(len(volOriginal)))
#    print '\rQuantified %d out of %d (Finished)      ' %(quantified,len(volOriginal)),
#    sys.stdout.flush()
    listdic=pd.DataFrame(List)
    if os.path.exists(pathFinal+separator+QuantificationFolder)==False:
        os.makedirs(pathFinal+separator+QuantificationFolder)
    listdic.to_csv(pathFinal+separator+QuantificationFolder+separator+NameCSV+'.csv')
    
  
class Lungs_Fields:
    VOL_HEAL_PIX = "Vol_Healthy_Pix"; VOL_HEAL = "Vol_Healthy"; VOL_SOFT_PIX = "Vol_Soft_Pix"; VOL_SOFT = "Vol_Soft";
    VOL_HARD_PIX = "Vol_Hard_Pix"; VOL_HARD = "Vol_Hard"; VOL_SICK_PIX = "Vol_Sick_Pix"; VOL_SICK = "Vol_Sick";
    VOL_REL_SOFT = "Vol_Rel_Soft"; VOL_REL_HARD = "Vol_Rel_Hard"; VOL_REL_SICK = "Vol_Rel_Sick"
           
 

def get_Lung_Damage(volOriginal, volMask, thresholds = (-1024,-500,-100, sys.maxint), vol_out_value = -1024, verbose = 0):
  """
  The function returns the volumes defined by Chen el al.(2014) by manual setting of the thresholds
  between regions.
  
  Parameters
  ----------
  volOriginal : SimpleITK image type
                The original chest volume to quantify
  volMask : SimpleITK image type
            Mask of the lungs for volOriginal
  thresholds : iterable-like.
               thresholds[0] < Healthy Tissue <= thresholds[1]\n
               thresholds[1] < Soft Tissue <= thresholds[2]\n
               thresholds[2] < Hard Tissue <= thresholds[3]\n
  Returns
  ----------
  Volumes defined in :class:"~Lungs_Fields"
  
  """
  healthLowTh = thresholds[0]
  healthUppTh = thresholds[1]
  softTissTh = thresholds[2]
  hardTissTh = thresholds[3]
  
  spacing = volOriginal.GetSpacing();
  voxelVol = spacing[0]*spacing[1]*spacing[2]

  volMask.CopyInformation(volOriginal)
    
  volMasked = SimpleITK.Mask(volOriginal, volMask,vol_out_value)
    
  volMaskedArray = SimpleITK.GetArrayFromImage(volMasked).ravel()
    
  low = volMaskedArray > healthLowTh
  high = volMaskedArray <= healthUppTh
  volHealthy = np.sum(low*high);
    
  low = volMaskedArray > healthUppTh
  high = volMaskedArray <= softTissTh
  volSoft = np.sum(low*high);
    
  low = volMaskedArray > softTissTh
  high = volMaskedArray <= hardTissTh
  volHard = np.sum(low*high);
    
  volSick = volSoft + volHard
  Vol_rel_soft = float(volSoft)/volSick
  Vol_rel_hard = float(volHard)/volSick
  Vol_rel_sick = float(volSick)/(volSick+volHealthy)
  
  if verbose > 0:
      print Lungs_Fields.VOL_SICK, volSick,
      Lungs_Fields.VOL_REL_SOFT, Vol_rel_soft,
      Lungs_Fields.VOL_REL_HARD, Vol_rel_hard,
      Lungs_Fields.VOL_REL_SICK, Vol_rel_sick        
    
  return  {Lungs_Fields.VOL_HEAL_PIX:volHealthy,   Lungs_Fields.VOL_HEAL:volHealthy*voxelVol, Lungs_Fields.VOL_SOFT_PIX:volSoft,
           Lungs_Fields.VOL_SOFT:volSoft*voxelVol, Lungs_Fields.VOL_HARD_PIX:volHard,         Lungs_Fields.VOL_HARD:volHard*voxelVol,
           Lungs_Fields.VOL_SICK_PIX:volSick,      Lungs_Fields.VOL_SICK:volSick*voxelVol,    Lungs_Fields.VOL_REL_SOFT:Vol_rel_soft,
           Lungs_Fields.VOL_REL_HARD:Vol_rel_hard, Lungs_Fields.VOL_REL_SICK:Vol_rel_sick}   




def rewriting_volumes(path,AssignedGroups,initialformat='.mhd'):
    for root, dirs, files in os.walk(path):
        for name in files:
            if initialformat in name:
                for key in AssignedGroups.keys():
                    if key in name:
                        image=SimpleITK.ReadImage(root+separator+name)
                        finalname=root+separator+name.split(initialformat)[0]+'_'+AssignedGroups.get(key)+initialformat
                        SimpleITK.WriteImage(image,finalname)
                        os.remove(root+separator+name)
                        os.remove(root+separator+name.split(initialformat)[0]+'.raw')
    
def reading_volumes(path):
    volumes=[]
    root_vol=[]
    for root, dirs, files in os.walk(path):
        for name in files:
            for formats in imageFormats:
                if formats in name:
                    volumes.append(name)
                    root_vol.append(root)
    volumes.sort()
    root_vol.sort()       
    return volumes,root_vol            
                

def read_excell(pathexcell,characteristics):
    
    excell=xlrd.open_workbook(pathexcell)        
    xl_sheet = excell.sheet_by_index(0)
    columns=xl_sheet.row(0)
    Columns=[]

    for idx, cell_obj in enumerate(columns):
        Columns.append(cell_obj.value)
        
    ColumnCharact={}
    for characteristic in characteristics:
        count=0
        for c in Columns:
            if c==characteristic:
                column=xl_sheet.col(count)
                values=[]
                for idx, cell_obj in enumerate(column):
                    values.append(cell_obj.value)
                ColumnCharact.update({characteristic:values[1:]})
            count=count+1
    return ColumnCharact




def assign_characteristic(names,groups):
    typeOfNames=set(names)
    AssignedGroups={}
    for typename in typeOfNames:
        count=0
        for name in names:
            if typename==name:
                AssignedGroups.update({str(name):str(groups[count])})
            count=count+1
    return AssignedGroups




def dead_subjects(names,weeks,groups):
    typeOfNames=set(names)
    typeOfWeeks=set(weeks)
    Dead={}
    for name in typeOfNames:
        for week in typeOfWeeks:
            found=0
            for n,g in zip(names,groups):
                if n==name:
                    group=g
            for n,w in zip(names,weeks):
                if n==name and w==week:
                    found=found+1
            if found==0:
                Dead.update({str(name):[str(int(week)),str(group)]})
    return Dead

def orderByGroup(volOriginal,volMask,rootOriginal,rootMask,groups):
    typeOfGroups=set(groups)
    orderedVolOriginal=[]
    orderedVolMask=[]
    orderedRootOriginal=[]
    orderedRootMask=[]
    for group in typeOfGroups:
        for original,mask,rootO,rootM in zip(volOriginal,volMask,rootOriginal,rootMask):
            if group in original:
                orderedVolOriginal.append(original)
                orderedVolMask.append(mask)
                orderedRootOriginal.append(rootO)
                orderedRootMask.append(rootM)
    return orderedVolOriginal,orderedVolMask,orderedRootOriginal,orderedRootMask
            

if __name__ == "__main__":
    print 'Super highly naive  quantification'
    
            