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

  if (argc < 4){
    std::cout << "Usage: " << argv[0] << " InputFileName OutputFileName label_id" << std::endl;
    return EXIT_SUCCESS;
  }
  
  if ( (strcmp(argv[1],"--help") == 0) || ( (strcmp(argv[1],"-h") == 0) ) ){
    std::cout << "Usage: " << argv[0] << " InputFileName OutputFileName label_id" << std::endl;
    return EXIT_SUCCESS;
  }

  typedef signed short PixelType;
  const unsigned int Dimension=3;

  typedef itk::Image< PixelType, Dimension >                       	ImageType;

  typedef itk::ImageFileReader< ImageType >                        	ReaderType;
  ReaderType::Pointer reader =                                     	ReaderType::New();
  typedef itk::ImageFileWriter< ImageType >                        	WriterType;

  char * infilename  = argv[1];
  char * outfilename  = argv[2];
  unsigned int label_id  = (unsigned int) atoi(argv[3]);

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

  ConverterType::Pointer converter = ConverterType::New();
  converter->SetInput( reader->GetOutput() );
  converter->SetInputForegroundValue( 1 );//Any nonzero value should be read as True
  converter->SetFullyConnected( 0 );
  converter->SetNumberOfThreads( 1 );
  converter->Update();

  std::vector<unsigned long> labelsToRemove;

  std::cout << "There are originally " << converter->GetOutput()->GetNumberOfLabelObjects() << " objects." << std::endl;

	// Note: do NOT remove the labels inside the loop!
	// Instead, mark all of the labels to be removed first and then remove them all together.
	for(unsigned int i = 0; i < converter->GetOutput()->GetNumberOfLabelObjects(); i++)
	  {

	  LabelType label = converter->GetOutput()->GetNthLabelObject(i)->GetLabel();

	  // Mark label to be removed
	  if(label != label_id)
		labelsToRemove.push_back(label);

	  }

	std::cout << "Removing " << labelsToRemove.size() << " objects." << std::endl;
	// Remove all regions that were marked for removal.
	for(unsigned int i = 0; i < labelsToRemove.size(); ++i)
	  {
	  converter->GetOutput()->RemoveLabel(labelsToRemove[i]);
	  }

	std::cout << "There are " << converter->GetOutput()->GetNumberOfLabelObjects()
			  << " objects remaining." << std::endl;

  LabelCollectionType::Pointer collection = converter->GetOutput();

  typedef itk::LabelMapToLabelImageFilter< LabelCollectionType,ImageType > MapType;
  MapType::Pointer mapper = MapType::New();
  mapper->SetInput( collection );
  mapper->Update();

  copy_dicom_data(reader->GetOutput(), mapper->GetOutput());
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput( mapper->GetOutput() );
  writer->SetFileName( outfilename );
  writer->Update();

  return EXIT_SUCCESS;

}
