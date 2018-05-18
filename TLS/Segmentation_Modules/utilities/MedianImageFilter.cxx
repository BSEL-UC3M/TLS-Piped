#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif



#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"





#include "itkMedianImageFilter.h"



int main( int argc, char * argv[] )
{
  if( argc < 4 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  inputImageFile   outputImageFile radius" << std::endl;
    return EXIT_FAILURE;
    }


  typedef   unsigned char  InputPixelType;
  typedef   unsigned char  OutputPixelType;

  typedef itk::Image< InputPixelType,  3 >   InputImageType;
  typedef itk::Image< OutputPixelType, 3 >   OutputImageType;



  typedef itk::ImageFileReader< InputImageType  >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  reader->SetFileName( argv[1] );
  writer->SetFileName( argv[2] );

  
  typedef itk::MedianImageFilter<
               InputImageType, OutputImageType >  FilterType;

  FilterType::Pointer filter = FilterType::New();
  
  InputImageType::SizeType indexRadius;
  
  int radius = atoi( argv[2]) ;
  
  indexRadius[0] = radius; // radius along x
  indexRadius[1] = radius; // radius along y
  indexRadius[1] = radius; // radius along y

  filter->SetRadius( indexRadius );
 
  filter->SetInput( reader->GetOutput() );
  writer->SetInput( filter->GetOutput() );
  writer->Update();
 

  return EXIT_SUCCESS;
}

