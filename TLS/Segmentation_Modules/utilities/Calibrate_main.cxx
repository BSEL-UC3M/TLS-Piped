
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_Median
#endif

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include "itkImageRegionIterator.h"

int main( int argc, char ** argv )
{

  if( argc < 7 )
    {
    std::cout << "Usage: " << argv[0] << " inputImageFile outputImageFile airMean airSigma bloodMean bloodSigma " << std::endl;
    return EXIT_FAILURE;
    }

  typedef signed short PixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< PixelType, Dimension >  ImageType;


  typedef itk::ImageFileReader < ImageType >  ReaderType;
  typedef itk::ImageFileWriter < ImageType >  WriterType;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  reader->SetFileName( argv[1] );
  writer->SetFileName( argv[2] );
  

  // perform normalization to Hounsfield Units by linear equation y=ax+b
  // we first need to compute a and b values
  float a=0, b=0;
  float HU_air_desidered = -1000;
  float HU_blood_desidered = 50;
  float HU_air_real = atof(argv[3]);
  float HU_air_sigma = atof(argv[4]);
  float HU_blood_real = atof(argv[5]);
  float HU_blood_sigma = atof(argv[6]);
  
  a = (HU_air_desidered - HU_blood_desidered) / (HU_air_real - HU_blood_real);
  b = HU_blood_desidered - a*HU_blood_real;
  
  std::cout << "The linear transformation utilized for calibration will be y=ax+b with: "<< std::endl;
  std::cout << " a: " << a << std::endl;  
  std::cout << " b: " << b << std::endl;
  
// we now apply the equation to the image using an iterator

  reader->Update();

  ImageType::Pointer HU_image = reader->GetOutput();
  //HU_image->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  //HU_image->Allocate();

  typedef itk::ImageRegionIterator<ImageType>   IteratorType;
  IteratorType it = IteratorType(HU_image, HU_image->GetLargestPossibleRegion());

  for(it.GoToBegin();!it.IsAtEnd();++it)
    {
    it.Set( floor( a*it.Get() + b ) ); 
    }

  writer->SetInput( HU_image );

  try
    {
    writer->Update();

    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught!" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }


  return EXIT_SUCCESS;
}
