#include <iostream>
#include <fstream>
using namespace std;
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include "itkExtractImageFilter.h"

#include "itkImageRegionConstIterator.h"

typedef signed short PixelType;
const unsigned int Dimension = 3;

typedef itk::Image< PixelType, Dimension >		ImageType;
typedef itk::Image< PixelType, 2 >				ImageType_2D;
typedef itk::ExtractImageFilter< ImageType, ImageType_2D > ExtractFilterType;


void find1(ExtractFilterType::Pointer extracter,double* tc) {
  // define iterator
  typedef itk::ImageRegionConstIterator< ImageType_2D > IteratorType;
  IteratorType sIt = IteratorType( extracter->GetOutput(), extracter->GetOutput()->GetLargestPossibleRegion() );

  double cnt = 0;


  tc[0] = 0;
  tc[1] = 0;

  for(sIt.GoToBegin(); !sIt.IsAtEnd(); ++sIt)
	  if(sIt.Get()!=0) {

        tc[0] = tc[0] + sIt.GetIndex()[0];
        tc[1] = tc[1] + sIt.GetIndex()[1];

        cnt++;

        std::cout << tc[0] << " " << tc[1] << " " << tc[2] << " " << cnt << std::endl;
     }

   tc[0] = round(tc[0]/cnt);
   tc[1] = round(tc[1]/cnt);


}

void find2() {


}


int main( int argc, char ** argv )
{
  // Verify the number of parameters in the command line

  int sliceStart = 0;

  if( argc < 3 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile [startFromSlice] outputSeedFile" << std::endl;
    return EXIT_FAILURE;
    }

  if( argc == 4 )
      sliceStart = atoi( argv[2] );

//start by defining the PixelType and ImageType

  //typedef signed short PixelType;




// The image type is used as a template parameter to instantiate
// the reader and writer.

  typedef itk::ImageFileReader < ImageType >  ReaderType;


  ReaderType::Pointer reader = ReaderType::New();

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


// Extract slice and grow trachea around the given seed to have initial wavefront

  ExtractFilterType::Pointer extracter = ExtractFilterType::New();

  ImageType::SizeType imageSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  ImageType::RegionType extractRegion;
  extractRegion.SetIndex( 0, 0);
  extractRegion.SetIndex( 1, 0);
  extractRegion.SetIndex( 2, sliceStart );
  extractRegion.SetSize( 0, imageSize[0]);
  extractRegion.SetSize( 1, imageSize[1]);
  extractRegion.SetSize( 2, 0);
  extracter->SetInput( reader->GetOutput() );
  extracter->SetExtractionRegion( extractRegion );
  extracter->Update();

  /*typedef itk::ImageFileWriter < ImageType_2D >  WriterType;

  WriterType::Pointer writer = WriterType::New();

  writer->SetInput(extracter->GetOutput());
  writer->SetFileName( "trachea.tiff" );
  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught!" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }*/


  double tc[3] = {0};
  find1(extracter,tc);
  std::cout << tc[0] << "," << tc[1] << "," << sliceStart << std::endl;
  //std::cout << "La madre del cordero..."<< std::endl;
  ofstream myfile;
  myfile.open (argv[3]);
  //myfile << "x" << "," << "y" << "," << "z" << std::endl;
  myfile << tc[0] << "," << tc[1] << "," << sliceStart << std::endl;
  myfile.close();
  return EXIT_SUCCESS;
}
