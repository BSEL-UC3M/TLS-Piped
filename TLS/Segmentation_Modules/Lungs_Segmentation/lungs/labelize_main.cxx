#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkShapeLabelObject.h"
#include "itkLabelMap.h"

#include "itkBinaryImageToLabelMapFilter.h"
#include "itkShapeLabelMapFilter.h"
#include "itkBinaryShapeKeepNObjectsImageFilter.h"
#include "dicom_utilities.h"

int main( int argc, char * argv[] )
{
  typedef signed short PixelType;
  const unsigned int Dimension=3;

  typedef itk::Image< PixelType, Dimension >                       	ImageType;

  typedef itk::ImageFileReader< ImageType >                        	ReaderType;
  ReaderType::Pointer reader =                                     	ReaderType::New();
  typedef itk::ImageFileWriter< ImageType >                        	WriterType;

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
  typedef itk::ShapeLabelMapFilter< LabelCollectionType > ShapeFilterType;
  typedef itk::ShapeKeepNObjectsLabelMapFilter< LabelCollectionType > KeepNObjectsType;

  ConverterType::Pointer converter = ConverterType::New();
  converter->SetInput( reader->GetOutput() );
  converter->SetInputForegroundValue( 1 );//Any nonzero value should be read as True
  converter->SetFullyConnected( 0 );//Less restrictive
  std::cout << converter->GetNumberOfThreads() << " used threads" << std::endl;
  //converter->SetNumberOfThreads( 1 );
  converter->Update();

  ShapeFilterType::Pointer shape = ShapeFilterType::New();
  shape->SetInput( converter->GetOutput() );
  shape->Update();
    
  KeepNObjectsType::Pointer opening = KeepNObjectsType::New();
  opening->SetInput( shape->GetOutput() );
  opening->SetNumberOfObjects( 3 );
  opening->SetReverseOrdering( false );
  opening->SetAttribute( "NumberOfPixels" );
  //opening->SetNumberOfThreads( 1 );

  LabelCollectionType::Pointer collection = opening->GetOutput();
  typedef itk::LabelMapToLabelImageFilter< LabelCollectionType,ImageType > MapType;
  MapType::Pointer mapper = MapType::New();
  mapper->SetInput( collection );
  mapper->Update();

  copy_dicom_data(reader->GetOutput(), mapper->GetOutput());
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput( mapper->GetOutput() );
  writer->SetFileName( outfilename );
  writer->Update();
  


  std::cout << "  Considering the " << collection->GetNumberOfLabelObjects() << " biggest objects of the image..." << std::endl;

  return EXIT_SUCCESS;

}
