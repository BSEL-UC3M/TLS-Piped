#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif


#include "itkScalarImageToHistogramGenerator.h"
#include "itkImage.h"

#include "itkImageFileReader.h"

int main( int argc, char * argv [] )
{

  if( argc < 2 )
    {
    std::cerr << "Missing command line arguments" << std::endl;
    std::cerr << "Usage :  hist  inputImageFileName hmin hMAX" << std::endl;
    return -1;
    }


  typedef signed int       PixelType;
  const unsigned int          Dimension = 3;

  typedef itk::Image<PixelType, Dimension > ImageType;

  int nobs = 256;
  float hmin = atof(argv[2]);
  float hMAX = atof(argv[3]);
  float bin_width = (hMAX - hmin)/nobs;

  std::cout << "Calculating hist with " << nobs << " bins " << " of width " << bin_width << std::endl;

  typedef itk::ImageFileReader< ImageType > ReaderType;

  ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName( argv[1] );

  try
    {
    reader->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << "Problem reading image file : " << argv[1] << std::endl;
    std::cerr << excp << std::endl;
    return -1;
    }

  typedef itk::Statistics::ScalarImageToHistogramGenerator< 
                                 ImageType >   HistogramGeneratorType;

  HistogramGeneratorType::Pointer histogramGenerator =
                                        HistogramGeneratorType::New();

  histogramGenerator->SetInput(  reader->GetOutput() );

  histogramGenerator->SetNumberOfBins( nobs );
  histogramGenerator->SetMarginalScale( 10.0 );

  histogramGenerator->SetHistogramMin(  hmin );
  histogramGenerator->SetHistogramMax( hMAX );

  histogramGenerator->Compute();

  typedef HistogramGeneratorType::HistogramType  HistogramType;

  const HistogramType * histogram = histogramGenerator->GetOutput();

  const unsigned int histogramSize = histogram->Size();

  std::cout << "Histogram size " << histogramSize << std::endl;

  HistogramType::ConstIterator itr = histogram->Begin();
  HistogramType::ConstIterator end = histogram->End();

  unsigned int binNumber = 0;
  while( itr != end )
    {
    std::cout << hmin + binNumber*bin_width << ", " << itr.GetFrequency() << std::endl;
    ++itr;
    ++binNumber;
    }

  return 0;
  
}




