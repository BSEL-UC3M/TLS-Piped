#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:35:56 2017

@author: pmacias
"""

import subprocess
import SimpleITK 
import csv
import os
from tracheaInitialPositionNewV2_Blanca import tracheaInitialPosition_V2
from Segmentation import Extracting


'''
--------------------- EXTRACTING TRACHEA ---------------------------------------
'''
finalName='Trachea_Extracted'
maskName='Mask_Trachea'

    
class Extract_Trachea_air_extracter(Extracting):

    def __init__(self,tracheaExpectedPerimeter=(11.5,33.5),timeStep='0.8',variablethreshold='-625 -300',fixedthreshold='1.4',kernel_dilation=2):
        self.timeStep=timeStep
        self.variablethreshold=variablethreshold
        self.fixedthreshold=fixedthreshold
        self.tracheaExpectedPerimeter=tracheaExpectedPerimeter
        self.kernel=kernel_dilation
        
    def execute(self,path_mask_lungs,path_originalimage):
        [coordinates,mask_modified,original_modified]=tracheaInitialPosition_V2([path_mask_lungs],[path_originalimage],self.tracheaExpectedPerimeter)
        if coordinates[0]=='0' and coordinates[1]=='0' and coordinates[2]=='0':
            find=find_trachea()
            coordinates=find.execute(path_mask_lungs)
        output_air_extracter=name_output('air_extracter',original_modified)
        calling_external('air_ext_fm',[original_modified,self.timeStep,output_air_extracter,self.variablethreshold,self.fixedthreshold,coordinates[0],coordinates[1],coordinates[2]])     
        mask_modified=SimpleITK.ReadImage(mask_modified)
        output_air_extracter=SimpleITK.ReadImage(output_air_extracter)
        mask_modified=binary(mask_modified)
        mask_trachea=binary(output_air_extracter)
        mask_trachea_dilated=dilationV2(mask_trachea,self.kernel)
        finalmask=subtract_masks(mask_modified,mask_trachea_dilated)
        finalmaskname=name_output(finalName,original_modified)
        mask_trachea_name=name_output(maskName,original_modified)
        SimpleITK.WriteImage(finalmask,finalmaskname)
        SimpleITK.WriteImage(mask_trachea_dilated,mask_trachea_name)
        return finalmaskname,mask_trachea_name

class Extract_Trachea_Tree_Auto(Extracting):

    def __init__(self,tracheaExpectedPerimeter=(11.5,26.5),timeStep='0.8',kernel_dilation=2):
        self.timeStep=timeStep
        self.tracheaExpectedPerimeter=tracheaExpectedPerimeter
        self.kernel=kernel_dilation
        
    def execute(self,path_mask_lungs,path_originalimage):
        [coordinates,mask_modified,original_modified]=tracheaInitialPosition_V2([path_mask_lungs],[path_originalimage],self.tracheaExpectedPerimeter)
        if coordinates[0]=='0' and coordinates[1]=='0' and coordinates[2]=='0':
            find=find_trachea()
            coordinates=find.execute(path_mask_lungs)
        output_tree_auto=name_output('Tree_Auto',original_modified)
        calling_external('Tree_Auto_v2',[original_modified,self.timeStep,output_tree_auto,coordinates[0],coordinates[1],coordinates[2]])     
        mask_modified=SimpleITK.ReadImage(mask_modified)
        output_tree_auto=SimpleITK.ReadImage(output_tree_auto)
        mask_modified=binary(mask_modified)
        mask_trachea=binary(output_tree_auto)
        mask_trachea_dilated=dilationV2(mask_trachea,self.kernel)
        finalmask=subtract_masks(mask_modified,mask_trachea_dilated)
        finalmaskname=name_output(finalName,original_modified)
        mask_trachea_name=name_output(maskName,original_modified)
        SimpleITK.WriteImage(finalmask,finalmaskname)
        SimpleITK.WriteImage(mask_trachea_dilated,mask_trachea_name)
        return finalmaskname,mask_trachea_name



'''CALLING EXTERNAL FUNCTION TERMINAL'''   

def calling_external(name_func,inputs):
    for root, dirs, files in os.walk(os.path.expanduser('~')):
        if name_func in files:
            path=os.path.join(root, name_func)
    for i in inputs:
        path=path+' '+i
    process = subprocess.Popen(path, shell=True)
    process.wait()   

def name_output(name_func,inputimage,finalformat=None):
    
    if finalformat==None:
        finalformat=inputimage.split('.')[1]
    return  inputimage.split('.')[0]+'_'+name_func+'.'+finalformat 


    
'''Find trachea
    Inputs:
        Name input volume
    Outputs:
        Coordinates of the certer of the trachea
'''
class find_trachea():
    
    def execute(self,inputimage):
        output_find_trachea=name_output('coordinates_trachea',inputimage,'.csv')
        calling_external('find_trachea',[inputimage,'1',output_find_trachea])                                   
        opencsv=open(output_find_trachea)
        coordinates=csv.reader(opencsv)
        for row in coordinates:
            self.coordinates=row
        return self.coordinates


            
'''Air extracter
    Inputs:im
        Name input volume
        timeStep, default='0.8'
        variablethreshold, default='-625 -300'
        fixedthreshold, default ='1.4'
        Coordinates of the center of the trachea
    Outputs:
        Name of the output volume
'''
class air_extracter():
    
    
    def __init__(self,timeStep='0.8',variablethreshold='-625 -300',fixedthreshold='1.4'):
        self.timeStep=timeStep
        self.variablethreshold=variablethreshold
        self.fixedthreshold=fixedthreshold
    
    def execute(self,inputimage,xcoordinate,ycoordinate,zcoordinate):       
        output_air_extracter=name_output('air_extracter',inputimage)
        calling_external('air_ext_fm',[inputimage,self.timeStep,output_air_extracter,self.variablethreshold,self.fixedthreshold,xcoordinate,ycoordinate,zcoordinate])     
        return  output_air_extracter
        
'''Tree_Auto
    Inputs:im
        Name input volume
    Outputs:
        Name of the output volume
'''

class Tree_Auto():

    
    def __init__(self,timeStep=0.8):
        self.timeStep=timeStep
    
    def execute(self,inputimage,xcoordinate,ycoordinate,zcoordinate):       
        self.timeStep=str(self.timeStep)
        output_Tree_Auto=name_output('Tree_Auto',inputimage)
        calling_external('Tree_Auto_v2',[inputimage,self.timeStep,output_Tree_Auto,xcoordinate,ycoordinate,zcoordinate])     
        return  output_Tree_Auto
    
    '''Binary images
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
    '''
def binary(inputimage):
        binary=SimpleITK.BinaryThresholdImageFilter()
        binary.SetLowerThreshold(1)
        binary.SetUpperThreshold(255)
        binary.SetInsideValue(255)
        binary.SetOutsideValue(0)
        binaryimage=binary.Execute(inputimage)
        return binaryimage 
'''Subtracting masks
    Inputs:
        Name input volume 1
        Name input volume 2
    Outputs:
        Name of the output volume
'''
def subtract_masks(inputimage1,inputimage2):
        inputimage1.SetSpacing([1.0,1.0,1.0])
        inputimage2.SetSpacing([1.0,1.0,1.0])
        inputimage1.SetOrigin([0.0, 0.0, 0.0])
        inputimage2.SetOrigin([0.0, 0.0, 0.0])
        subtract=SimpleITK.SubtractImageFilter()
        Mask=subtract.Execute(inputimage1,inputimage2)
        binary=SimpleITK.BinaryThresholdImageFilter()
        binary.SetLowerThreshold(254)
        binary.SetUpperThreshold(255)
        binary.SetInsideValue(255)
        binary.SetOutsideValue(0)
        Maskbin=binary.Execute(Mask)
        return Maskbin   
    
def dilationV2(inputimage,kernel=2):
        dilation=SimpleITK.BinaryDilateImageFilter()
        dilation.SetKernelRadius(kernel)
        dilation.SetForegroundValue(255)
        dilatedimage=dilation.Execute(inputimage)
        return dilatedimage   
    
