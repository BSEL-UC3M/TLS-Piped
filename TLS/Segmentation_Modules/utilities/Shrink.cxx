
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <iostream>

#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkAffineTransform.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;
using namespace itk;
int main( int argc, char * argv[] )
{

  char * inFilename;
  char * outFilename;
  float scale = 0;
  int x_size = 0;
  int y_size = 0;
  
  bool x_y_size = false;  
  if ((argc < 4))
  { 
    cerr << "Usage: " << endl;
    cerr << "  · " << argv[0] << "  inputImageFile outputImageFile scale_factor [x_size y_size]" << endl;
    return EXIT_FAILURE;    
  } else if (argc == 6) {
  
      inFilename  = argv[1];
      outFilename = argv[2];
      scale = atof(argv[3]);
      x_size = atoi(argv[4]);
      y_size = atoi(argv[5]);
      x_y_size = true;
    
  } else {
      inFilename  = argv[1];
      outFilename = argv[2];
      scale = atof(argv[3]);
  }
  

  typedef signed short                                        PixelType;
  typedef Image<PixelType, 2>                                 ImageType;
  typedef ImageFileReader<ImageType>                          ReaderType;
  typedef ImageFileWriter<ImageType>                          WriterType;
  typedef ResampleImageFilter<ImageType,ImageType>            ReSampleType;
  typedef itk::AffineTransform<double, 2>                     TransformType;
  typedef itk::NearestNeighborInterpolateImageFunction<
      ImageType, double >  InterpolatorType;
  
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inFilename);
  reader->Update();
  
  double inSpacing[2];
  inSpacing[0] = reader->GetOutput()->GetSpacing()[0]*scale;
  inSpacing[1] = reader->GetOutput()->GetSpacing()[1]*scale;
  double origin[2];
  origin[0] = 0.0;
  origin[1] = 0.0;
  ImageType::SizeType inSize;
  inSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  ImageType::SizeType outSize;
  if (x_y_size) {
    outSize[0] = x_size;
    outSize[1] = y_size;
  } else {
    outSize[0] = inSize[0]/scale;
    outSize[1] = inSize[1]/scale;
  }
  
  ReSampleType::Pointer resampler1 = ReSampleType::New();
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  resampler1->SetInterpolator(interpolator);
  resampler1->SetDefaultPixelValue(0);
  resampler1->SetOutputSpacing(inSpacing);
  resampler1->SetOutputOrigin(origin);
  resampler1->SetSize(outSize);
  resampler1->SetInput(reader->GetOutput());
  resampler1->Update();
  
  
  /*
  string inDirString = "";
  string prefix = "shrinked-";
  inDirString += inFilename;
  string inPath = inDirString.substr(0,inDirString.find_last_of("/")+1);
  string inFileString = inDirString.substr(inDirString.find_last_of("/")+1);
  
  
  string outDString =  prefix + inFilename;
  */
  
  WriterType::Pointer dWriter = WriterType::New();
  dWriter->SetInput(resampler1->GetOutput());
  //dWriter->SetFileName(outDString.c_str());
  dWriter->SetFileName(outFilename);
  dWriter->Update();
  
  return EXIT_SUCCESS;
}

