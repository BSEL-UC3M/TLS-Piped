
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryBallStructuringElement.h" 

int main( int argc, char * argv[] )
{
  if( argc != 4 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  inputImageFile  " << " kernel outputImageFile" << std::endl;
    return EXIT_FAILURE;
    }


  const unsigned int Dimension = 3;
  const unsigned int foreground = 1;
  
  typedef unsigned char   InputPixelType;
  typedef unsigned char   OutputPixelType;

  typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
  typedef itk::Image< OutputPixelType, Dimension >   OutputImageType;

  typedef itk::ImageFileReader< InputImageType  >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  typedef itk::BinaryBallStructuringElement< 
                      InputPixelType,
                      Dimension  >             StructuringElementType;
 
  typedef itk::BinaryErodeImageFilter<
                            InputImageType, 
                            OutputImageType,
                            StructuringElementType >  ErodeFilterType;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
 
  ErodeFilterType::Pointer  binaryErode  = ErodeFilterType::New();
 
  StructuringElementType  structuringElement;

  structuringElement.SetRadius( atoi(argv[2]) );  

  structuringElement.CreateStructuringElement();

  binaryErode->SetKernel(  structuringElement );

  reader->SetFileName( argv[1] );
  writer->SetFileName(  argv[3] );

  binaryErode->SetInput( reader->GetOutput() );
  
  binaryErode->SetErodeValue( foreground );

  writer->SetInput( binaryErode->GetOutput() );
  writer->Update();

  return EXIT_SUCCESS;
}

