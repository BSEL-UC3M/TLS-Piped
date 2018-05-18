#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
//#include "itkHuThresholdImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "itkOtsuThresholdImageFilter.h"

#include "dicom_utilities.h"


int main( int argc, char * argv[] )
{


  if (argc < 3){
    std::cout << "Usage: " << argv[0] << " InputFileName OutputFileName" << std::endl;
    return EXIT_SUCCESS;
  }
  
  if ( (strcmp(argv[1],"--help") == 0) || ( (strcmp(argv[1],"-h") == 0) ) ){
    std::cout << "Usage: " << argv[0] << " InputFileName OutputFileName" << std::endl;
    return EXIT_SUCCESS;
  }


  typedef signed short PixelType;
  const unsigned int Dimension=3;

  typedef itk::Image< PixelType, Dimension >                       	ImageType;

  typedef itk::ImageFileReader< ImageType >                        	ReaderType;
  ReaderType::Pointer reader =                                     	ReaderType::New();
  typedef itk::ImageFileWriter< ImageType >                        	WriterType;

  typedef itk::ThresholdImageFilter<ImageType>   ThresholdType;
  typedef itk::OtsuThresholdImageFilter<ImageType, ImageType>   OtsuThresholdType;

  char * infilename  = argv[1];
  char * outfilename  = argv[2];

  reader->SetFileName( infilename );

  try {
	reader->Update();
  } catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught: can't read serie" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }
/*
  float     m_HuDiff = 0.01;
  HuThresholdType::Pointer huthreshold =               HuThresholdType::New();
  huthreshold->SetDiff(m_HuDiff);
  huthreshold->SetInput(reader->GetOutput());
  huthreshold->Update();
*/

/*
 * Some images are not limited to -1024 HU. To avoid problems with the Otsu threshold,
 * let's set all pixels < -1024 to -1024
*/

  int lowerTh = -1024;
  ThresholdType::Pointer th = ThresholdType::New();
  th->SetInput(reader->GetOutput());
  th->ThresholdBelow(lowerTh);
  th->SetOutsideValue(lowerTh);

  WriterType::Pointer writer = WriterType::New();

/*
 * Now we can run the Otsu Threshold
*/

  OtsuThresholdType::Pointer otsu = OtsuThresholdType::New();
  otsu->SetInput(th->GetOutput());
  otsu->SetOutsideValue( 0 );
  otsu->SetInsideValue(  1  );

//We need to update the filter before copying dicom data
  otsu->Update(); 
  copy_dicom_data(reader->GetOutput(), otsu->GetOutput());

   writer->SetInput(otsu->GetOutput());
  writer->SetFileName(outfilename);
  writer->Update();

  return EXIT_SUCCESS;

}
