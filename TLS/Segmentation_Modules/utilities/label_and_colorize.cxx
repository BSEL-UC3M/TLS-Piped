#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkShapeLabelObject.h"
#include "itkLabelMap.h"

#include "itkBinaryImageToLabelMapFilter.h"
#include "itkLabelMapOverlayImageFilter.h"
#include "itkShapeRelabelLabelMapFilter.h"
#include "itkRGBPixel.h"


int main( int argc, char * argv[] )
{

  if (argc < 3){
    std::cout << "Usage: " << argv[0] << " InputFileName OutputFileName" << std::endl;
    return EXIT_SUCCESS;
  }
  
  if ( (strcmp(argv[1],"--help") == 0) || ( (strcmp(argv[1],"-h") == 0) ) ){
    std::cout << "Usage: " << argv[0] << " InputFileName OutputFileName" << std::endl;
    return EXIT_SUCCESS;
  }

  typedef signed short PixelType;
  const unsigned int Dimension=3;

  typedef itk::Image< PixelType, Dimension >                       	ImageType;
  typedef itk::Image< itk::RGBPixel<PixelType>, Dimension >             	RGBImageType;

  typedef itk::ImageFileReader< ImageType >                        	ReaderType;
  ReaderType::Pointer reader =                                     	ReaderType::New();
  typedef itk::ImageFileWriter< RGBImageType >                      WriterType;

  char * infilename  = argv[1];
  char * outfilename  = argv[2];

  reader->SetFileName( infilename );

  try {
	reader->Update();
  } catch( itk::ExceptionObject & err ) {
    std::cerr << "ExceptionObject caught: can't read serie" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  typedef unsigned long LabelType;
  typedef itk::ShapeLabelObject< LabelType, 3 >  LabelObjectType;
  typedef itk::LabelMap< LabelObjectType > LabelCollectionType;  
  typedef itk::BinaryImageToLabelMapFilter< ImageType, LabelCollectionType > ConverterType;
  typedef itk::LabelMapOverlayImageFilter< LabelCollectionType, ImageType, RGBImageType > OverlayType;
  typedef itk::ShapeRelabelLabelMapFilter<LabelCollectionType> RelabelType;


  ConverterType::Pointer converter = ConverterType::New();
  converter->SetInput( reader->GetOutput() );
  converter->SetInputForegroundValue( 1 );//Any nonzero value should be read as True
  converter->SetFullyConnected( 0 );
  converter->SetNumberOfThreads( 1 );
  converter->Update();

  LabelCollectionType::Pointer collection = converter->GetOutput();
  RelabelType::Pointer relabel = RelabelType::New();
  OverlayType::Pointer overlay = OverlayType::New();
 
  relabel->SetInput(collection);
  relabel->SetAttribute("NumberOfPixels");

  overlay->SetInput(relabel->GetOutput());
  overlay->SetFeatureImage(reader->GetOutput());
 
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput( overlay->GetOutput() );
  writer->SetFileName( outfilename );
  writer->Update();
  

  return EXIT_SUCCESS;

}
