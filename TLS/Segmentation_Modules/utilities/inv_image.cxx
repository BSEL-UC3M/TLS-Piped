
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include "itkInvImageFilter.h"
#include "dicom_utilities.h"

int main( int argc, char ** argv )
{
  // Verify the number of parameters in the command line
  
  
  if( argc < 3 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile outputImageFile" << std::endl;
    return EXIT_FAILURE;
    }


// We start by defining the PixelType and ImageType

  typedef signed short PixelType;
  typedef signed short OutputPixelType;
  
  const unsigned int Dimension = 3;

  typedef itk::Image< PixelType, Dimension >		ImageType;
  typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;

  typedef itk::ImageFileReader < ImageType >  ReaderType;
  typedef itk::ImageFileWriter < OutputImageType >  WriterType;
  typedef itk::InvImageFilter< ImageType, OutputImageType> InvertFilterType;

  ReaderType::Pointer reader = ReaderType::New();
  InvertFilterType::Pointer inv = InvertFilterType::New();
  WriterType::Pointer writer = WriterType::New();
  
  reader->SetFileName( argv[1] );
    
  try
    {
    reader->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught!" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  inv->SetInput(reader->GetOutput());
  inv->Update();
  
  copy_dicom_data(reader->GetOutput(), inv->GetOutput());

  writer->SetInput( inv->GetOutput() );
  writer->SetFileName( argv[2] );

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
