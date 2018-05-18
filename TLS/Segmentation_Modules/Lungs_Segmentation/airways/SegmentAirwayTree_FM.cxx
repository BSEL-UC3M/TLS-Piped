
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_Median
#endif

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include "itkExtractImageFilter.h"

#include "itkImageRegionConstIterator.h"

#include "itkTreeSegmentationImageFilter.h"
#include "dicom_utilities.h"


int main( int argc, char ** argv )
{
  float v_th = -625;
  float f_th = -300;
  float sigma = 1.4;
  int x_c = 0;
  int y_c = 0;
  int z_c = 0;

  //just srt the main paramenters
  if( argc < 4 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile timeStep outputImageFile [variable_threshold] [fixed_threshold] [sigma] x_c y_c z_c" << std::endl;
    return EXIT_FAILURE;
    }

    v_th = atof(argv[4]);
    f_th = atof(argv[5]);
    sigma = atof(argv[6]);

    x_c = atoi(argv[7]);
    y_c = atoi(argv[8]);
    z_c = atoi(argv[9]);
   //std::cerr << "Args readed" << std::endl;


  typedef signed short PixelType;
  typedef signed short OutputPixelType;
  const unsigned int Dimension = 3;


  typedef itk::Image< PixelType, Dimension >		ImageType;
  typedef itk::Image< PixelType, 2 >				ImageType_2D;
  typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;


  typedef itk::ImageFileReader < ImageType >  ReaderType;
  typedef itk::ImageFileWriter < OutputImageType >  WriterType;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  reader->SetFileName( argv[1] );
  try
    {
  std::cerr << "Reader update " << std::endl;
    reader->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught!" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  writer->SetFileName( argv[3] );

// Extract slice and grow trachea around the given seed to have initial wavefront
  typedef itk::ExtractImageFilter< ImageType, ImageType_2D > ExtractFilterType;
  ExtractFilterType::Pointer extracter = ExtractFilterType::New();
   std::cerr << "Extracter " << std::endl;
  ImageType::SizeType imageSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  ImageType::RegionType extractRegion;
  extractRegion.SetIndex( 0, 0);
  extractRegion.SetIndex( 1, 0);
  extractRegion.SetIndex( 2, 0);
  extractRegion.SetSize( 0, imageSize[0]);
  extractRegion.SetSize( 1, imageSize[1]);
  extractRegion.SetSize( 2, 0);
  extracter->SetInput( reader->GetOutput() );
  extracter->SetExtractionRegion( extractRegion );
  std::cerr << "Segmenter" << std::endl;
  ImageType_2D::IndexType seed;

  // define airway tree segmentation filter
  typedef itk::TreeSegmentationImageFilter<ImageType, ImageType> SegmentFilterType;
  SegmentFilterType::Pointer segmenter = SegmentFilterType::New();
  segmenter->SetInput( reader->GetOutput() );
  segmenter->SetPropagationSigma(sigma);
  segmenter->Setvar_threshold(v_th);
  segmenter->Setfix_threshold(f_th);


  typedef itk::ImageRegionConstIterator< ImageType_2D > IteratorType;
  ImageType::IndexType seed_3D;
  seed_3D[0] = x_c; seed_3D[1] = y_c;  seed_3D[2] = z_c;

  segmenter->SetSeed(seed_3D);

  segmenter->SetTimeStep(atof(argv[2]));
  segmenter->SetBetaRadius(2);
  segmenter->Update();
  copy_dicom_data(reader->GetOutput(), segmenter->GetOutput());
  writer->SetInput( segmenter->GetOutput() );


  try
    {
  std::cerr << "writer " << std::endl;
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
