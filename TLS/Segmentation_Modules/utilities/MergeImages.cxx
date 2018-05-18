#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include "itkMergeRegionImageFilter.h"

int main( int argc, char ** argv )
{
  // Verify the number of parameters in the command line
  
  
  if( argc < 5 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " originalImageFile mergeImageFile ix iy iz sx sy sz prob_cutoff outputImageFile" << std::endl;
    return EXIT_FAILURE;
    }

  char * original_file_name = argv[1];
  char * merge_file_name = argv[2];
  int ix = atoi(argv[3]); int iy = atoi(argv[4]);int iz = atoi(argv[5]);
  int sx = atoi(argv[6]); int sy = atoi(argv[7]);int sz = atoi(argv[8]);
  float prob_threshold = atof(argv[9]);
  char * out_file_name = argv[10];
  
// We start by defining the PixelType and ImageType

  typedef float PixelType;
  typedef float OutputPixelType;
  
  const unsigned int Dimension = 3;

  typedef itk::Image< PixelType, Dimension >		ImageType;
  typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;

  typedef itk::ImageFileReader < ImageType >  ReaderType;
  typedef itk::ImageFileWriter < OutputImageType >  WriterType;
  typedef itk::MergeRegionImageFilter< ImageType, OutputImageType> MergeFilterType;
  
  ImageType::IndexType index;
  ImageType::SizeType size;
  ImageType::RegionType mergeRegion;

  ReaderType::Pointer reader1 = ReaderType::New();
  ReaderType::Pointer reader2 = ReaderType::New();
  MergeFilterType::Pointer merger = MergeFilterType::New();
  WriterType::Pointer writer = WriterType::New();
  

  index[0] = ix; index[1] = iy;  index[2] = iz;
  size[0] = sx; size[1] = sy;  size[2] = sz;
  mergeRegion.SetIndex(index);
  mergeRegion.SetSize(size);
  
  reader1->SetFileName( original_file_name );
    
  try
    {
    reader1->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "Error reading " << original_file_name << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  ImageType::Pointer original_image = reader1->GetOutput();

  reader2->SetFileName( merge_file_name );
    
  try
    {
    reader2->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "Error reading " << merge_file_name << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }
  
  ImageType::Pointer merge_image = reader2->GetOutput();
  
  merger->SetInput(original_image);
  merger->SetMergeImage(merge_image);
  merger->SetMergeRegion(mergeRegion);
  merger->SetProbThreshold(prob_threshold);
  
 
  writer->SetInput( merger->GetOutput() );
  writer->SetFileName( out_file_name );

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
