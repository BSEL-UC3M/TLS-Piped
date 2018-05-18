#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include <itkMaskImageFilter.h>
#include <itkMaskNegatedImageFilter.h>
#include "dicom_utilities.h"

int main( int argc, char ** argv )
{
  // Verify the number of parameters in the command line
  
  
  if( argc < 4 )
    {
    std::cout << "Usage: ";
    std::cout << argv[0] << " inputImageFile maskImage outputImageFile [negated]" << std::endl;
    return EXIT_SUCCESS;
    }

  bool negated = false;
  if (argc == 5) {
    negated = atoi(argv[4]);
  }

// We start by defining the PixelType and ImageType

  typedef unsigned short MaskPixelType;
  typedef signed short PixelType;
  typedef signed short OutputPixelType;
  
  const unsigned int Dimension = 3;

  typedef itk::Image< PixelType, Dimension >		ImageType;
  typedef itk::Image< MaskPixelType, Dimension >		MaskImageType;
  typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;

  typedef itk::ImageFileReader < ImageType >  ReaderType;
  typedef itk::ImageFileReader < MaskImageType >  MaskReaderType;
  typedef itk::ImageFileWriter < OutputImageType >  WriterType;

  ReaderType::Pointer reader = ReaderType::New();
  MaskReaderType::Pointer maskReader = MaskReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  
  reader->SetFileName( argv[1] );
  maskReader->SetFileName( argv[2] );
  
  try
    {
    reader->Update();
    maskReader->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught!" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }


  if (negated) {

    typedef itk::MaskNegatedImageFilter< ImageType, MaskImageType, OutputImageType> MaskFilterType;
    MaskFilterType::Pointer mask = MaskFilterType::New();
    mask->SetInput1(reader->GetOutput());  
    mask->SetInput2(maskReader->GetOutput());  
    mask->Update();
    writer->SetInput( mask->GetOutput() );
    copy_dicom_data(reader->GetOutput(), mask->GetOutput());
    
  } else {

    typedef itk::MaskImageFilter< ImageType, MaskImageType, OutputImageType> MaskFilterType;
    MaskFilterType::Pointer mask = MaskFilterType::New();
    mask->SetInput1(reader->GetOutput());  
    mask->SetInput2(maskReader->GetOutput());  
    mask->Update();
    writer->SetInput( mask->GetOutput() );
    copy_dicom_data(reader->GetOutput(), mask->GetOutput());

  }
  
  

  writer->SetFileName( argv[3] );

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
