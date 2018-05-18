#include "itkImage.h"
#include "itkImageFileReader.h"

int main( int argc, char * argv[] )
{


  if (argc < 2){
    std::cout << "Usage: " << argv[0] << " InputFileName" << std::endl;
    return EXIT_SUCCESS;
  }
  
  if ( (strcmp(argv[1],"--help") == 0) || ( (strcmp(argv[1],"-h") == 0) ) ){
    std::cout << "Usage: " << argv[0] << " InputFileName" << std::endl;
    return EXIT_SUCCESS;
  }


  typedef signed short PixelType;
  const unsigned int Dimension=3;

  typedef itk::Image< PixelType, Dimension >                       	ImageType;

  typedef itk::ImageFileReader< ImageType >                        	ReaderType;
  ReaderType::Pointer reader =                                     	ReaderType::New();
 
  char * infilename  = argv[1];

  reader->SetFileName( infilename );

  try {
	reader->Update();
  } catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught: can't read serie" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << "Image orientation is: " << reader->GetOutput()->GetDirection() <<  std::endl;

  return EXIT_SUCCESS;

}
