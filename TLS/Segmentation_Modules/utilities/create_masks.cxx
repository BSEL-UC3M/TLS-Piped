#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

#include "itkMaskRegionImageFilter.h"
#include <itkBinaryErodeImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkSubtractImageFilter.h>
//#include "itkFillHolesImageFilter.h"
//#include "itkBinaryFillholeImageFilter.h"
#include <itkSliceBySliceImageFilter.h>

#include <stdio.h>
#include <iostream>
#include <string>
#include <sys/stat.h>

bool FileExists(std::string );

int main( int argc, char * argv[] )
{

  if( argc < 2 )
    {
    std::cout << "Usage: ";
    std::cout << argv[0] << " inputImageFile" << std::endl;
    return EXIT_SUCCESS;
    }
    
  if (not strcmp(argv[1],"--help")) {
    std::cout << "Usage: ";
    std::cout << argv[0] << " inputImageFile" << std::endl;
    return EXIT_SUCCESS;
  }

  char * infilename  = argv[1];
  
  typedef signed short PixelType;
  typedef unsigned char MaskPixelType;
  const unsigned int Dimension=3;

  typedef itk::Image< PixelType, 3 >                       	ImageType;
  typedef itk::Image< MaskPixelType, 3 >                    MaskImageType;
  typedef itk::Image< MaskPixelType, 2 >                    SliceImageType;

	
//	typedef ImageType::IndexType                                     	IndexType;
//  typedef ImageType::SizeType                                      	SizeType;
//  typedef ImageType::RegionType                                    	OutputImageRegionType;
 
  typedef itk::ImageFileReader< ImageType >                      	ReaderType;
	ReaderType::Pointer reader =                                    ReaderType::New();
  typedef itk::ImageFileWriter< MaskImageType >                   MaskWriterType;
	MaskWriterType::Pointer mask_writer =                           MaskWriterType::New();

  //Read image	
	reader->SetFileName( infilename );
	reader->Update();
  ImageType::Pointer inputImage =	reader->GetOutput();
  // Create output image	
	
	MaskImageType::Pointer maskImage = MaskImageType::New();
	maskImage->SetRegions( inputImage->GetRequestedRegion() );
	maskImage->CopyInformation( inputImage );
  maskImage->Allocate();
  
  
  maskImage->SetSpacing(inputImage->GetSpacing());  
  //std::cout << "Spacing " << maskImage->GetSpacing() << std::endl;
  //Convert to a binary mask (255)
  typedef itk::ImageRegionConstIterator< ImageType > ConstIteratorType;
  typedef itk::ImageRegionIterator< MaskImageType>       IteratorType;

  ConstIteratorType inputIt(   inputImage, inputImage->GetRequestedRegion() );
  IteratorType      maskIt(  maskImage, inputImage->GetRequestedRegion() );
  for ( inputIt.GoToBegin(), maskIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++maskIt)
  {
    if (inputIt.Get() != 0)
      maskIt.Set( 255 );
    else
      maskIt.Set( 0 );
  }

  mask_writer->SetFileName("mask.dcm");
  mask_writer->SetInput(maskImage);
  mask_writer->Update();
  
  //Split up this mask in

  typedef itk::MaskRegionImageFilter< MaskImageType, MaskImageType > ROIFilterType;
  ROIFilterType::Pointer extractor = ROIFilterType::New();
  MaskImageType::RegionType inputRegion = maskImage->GetLargestPossibleRegion();
  MaskImageType::SizeType size = inputRegion.GetSize();
  MaskImageType::IndexType start = inputRegion.GetIndex();
  MaskImageType::RegionType desiredRegion;

  extractor->SetInput( maskImage );
  const unsigned int max_x = size[0];
  const unsigned int max_y = size[1];
  const unsigned int max_z = size[2];
  
  
  //std::cout << "Maximum number of slices: " << max_z;
  size[2] = (int) max_z / 3.0;  

  //upper
  start[2] = 0;
    
  desiredRegion.SetSize( size );
  desiredRegion.SetIndex( start );
  extractor->SetRegionOfInterest( desiredRegion );
  //std::cout << extractor;    

  if (not FileExists("up_mask.dcm")) {
      std::cout << "Generating up_mask.dcm..." << std::endl;
      mask_writer->SetFileName("up_mask.dcm");
      mask_writer->SetInput( extractor->GetOutput() );
      mask_writer->Update();
   }

  //middle
  start[2] = size[2];
  
  desiredRegion.SetSize( size );
  desiredRegion.SetIndex( start );
  extractor->SetRegionOfInterest( desiredRegion );
  //std::cout << extractor;
  if (not FileExists("middle_mask.dcm")) {
      std::cout << "Generating middle_mask.dcm..." << std::endl;
      mask_writer->SetFileName("middle_mask.dcm");
      mask_writer->SetInput( extractor->GetOutput() );
      mask_writer->Update();
  }

  //lower
  start[2] = 2*size[2];
  
  desiredRegion.SetSize( size );
  desiredRegion.SetIndex( start );
  extractor->SetRegionOfInterest( desiredRegion );
  //std::cout << extractor;    
  if (not FileExists("lower_mask.dcm")) {
    std::cout << "Generating lower_mask.dcm..." << std::endl;
    mask_writer->SetFileName("lower_mask.dcm");
    mask_writer->SetInput( extractor->GetOutput() );
    mask_writer->Update();
  }
  //left

  start[2] = 0;
  size[0] = (int) max_x / 2; size[1] = max_y; size[2] = max_z;
   
  desiredRegion.SetSize( size );
  desiredRegion.SetIndex( start );
  extractor->SetRegionOfInterest( desiredRegion );
  //std::cout << extractor;
  if (not FileExists("left_mask.dcm")) {
      std::cout << "Generating left_mask.dcm..." << std::endl;
      mask_writer->SetFileName("left_mask.dcm");
      mask_writer->SetInput( extractor->GetOutput() );
      mask_writer->Update();
  }
  //right
  start[0] = (int) max_x / 2;
  size[0] = (int) max_x / 2; size[1] = max_y; size[2] = max_z;
   
  desiredRegion.SetSize( size );
  desiredRegion.SetIndex( start );
  extractor->SetRegionOfInterest( desiredRegion );
  //std::cout << extractor;    
  if (not FileExists("right_mask.dcm")) {
      std::cout << "Generating right_mask.dcm..." << std::endl;
      mask_writer->SetFileName("right_mask.dcm");
      mask_writer->SetInput( extractor->GetOutput() );
      mask_writer->Update();
  }
  

  /* Peel and core:
     1) first erode the original mask image
     2) fill holes in the eroded image
     3) make the difference between the original and the eroded one. This is the peel image.
     4) Now take the original mask image and subtract the peel. This is the core.
  */

  //1) Hole filling - DISABLE DUE TO ERRORS IN COMPILATION (itkBinaryFillholeImageFilter no in standard itk)
  /*
  
  typedef itk::BinaryFillholeImageFilter< SliceImageType >    FillHoles;
  FillHoles::Pointer filler =  FillHoles::New();
  filler->SetForegroundValue(255);

  typedef itk::SliceBySliceImageFilter< MaskImageType,MaskImageType >    SliceBySliceType;
  SliceBySliceType::Pointer slice_by_slice = SliceBySliceType::New();
  slice_by_slice->SetFilter(filler);
  slice_by_slice->SetInput( maskImage );
*/
  //mask_writer->SetFileName("no_holes.dcm");
  //mask_writer->SetInput( slice_by_slice->GetOutput() );
  //mask_writer->Update();
  
  //2) erosion
  typedef itk::BinaryBallStructuringElement<unsigned short, Dimension> kernelType;
  
  // Declare the type for the morphology Filter
  typedef itk::BinaryErodeImageFilter<MaskImageType, MaskImageType, kernelType> eroderType;

  // Create the filter
  eroderType::Pointer eroder = eroderType::New();

  // Create the structuring element
  kernelType ball;
  kernelType::SizeType ballSize;

//  eroder->SetInput( slice_by_slice->GetOutput() );
  eroder->SetInput( maskImage );
  
  ball.SetRadius( 10 );
  ball.CreateStructuringElement();
  eroder->SetKernel( ball );

  //mask_writer->SetFileName("eroded.dcm");
  //mask_writer->SetInput( eroder->GetOutput() );
  //mask_writer->Update();
  
  //3) Peel_Mask = Original_Mask - HoleFilled_Mask
  typedef itk::SubtractImageFilter< MaskImageType, MaskImageType > SubtractFilterType;
  SubtractFilterType::Pointer subtractor = SubtractFilterType::New();
  subtractor->SetInput1( maskImage );
  subtractor->SetInput2( eroder->GetOutput() );
  std::cout << "Generating peel_mask.dcm..." << std::endl;
  mask_writer->SetFileName("peel_mask.dcm");
  mask_writer->SetInput( subtractor->GetOutput() );
  mask_writer->Update();
 
  MaskImageType::Pointer peel_img = subtractor->GetOutput();

  //4) Core_Mask = Original_Mask - Peel_Mask
  subtractor->SetInput2( peel_img );
  std::cout << "Generating core_mask.dcm..." << std::endl;
  mask_writer->SetFileName("core_mask.dcm");
  mask_writer->SetInput( subtractor->GetOutput() );
  mask_writer->Update();

}

bool FileExists(std::string strFilename) {
  struct stat stFileInfo;
  bool blnReturn;
  int intStat;

  // Attempt to get the file attributes
  intStat = stat(strFilename.c_str(),&stFileInfo);
  if(intStat == 0) {
    // We were able to get the file attributes
    // so the file obviously exists.
    blnReturn = true;
  } else {
    // not able to get the file attributes.
    // This may mean that we don't have permission to
    // access the folder which contains this file. If you
    // need to do that level of checking, lookup the
    // return values of stat which will give you
    // more details on why stat failed.
    blnReturn = false;
  }
  
  return(blnReturn);
}

