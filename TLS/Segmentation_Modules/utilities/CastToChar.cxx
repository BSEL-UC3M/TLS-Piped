#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkCastImageFilter.h"

int main( int argc, char ** argv )
{

 typedef signed short PixelType;
 typedef char OutputPixelType;
 const unsigned int Dimension = 3;


  typedef itk::Image< PixelType, Dimension >		ImageType;
  typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;


// The image type is used as a template parameter to instantiate
// the reader and writer.

  typedef itk::ImageFileReader < ImageType >  ReaderType;
  typedef itk::CastImageFilter < ImageType, OutputImageType >  CasterType;
  typedef itk::ImageFileWriter < OutputImageType >  WriterType;
  
  if( argc < 3 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile outputImageFile " << std::endl;
    return EXIT_FAILURE;
    }  
  
  ReaderType::Pointer reader = ReaderType::New();
  CasterType::Pointer caster = CasterType::New();
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

  caster->SetInput( reader->GetOutput() );
  writer->SetFileName( argv[2] );
  
   writer->SetInput( caster->GetOutput() );

//  Finally, execution of the pipeline can be triggered by invoking the
//  Update() method in the writer. This call must be placed in a try/catch
//  block since exceptions be potentially be thrown in the process of reading
//  or writing the images

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
// Software Guide : EndCodeSnippet

  return EXIT_SUCCESS;
}
