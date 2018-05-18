
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include "itkImageRegion.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "dicom_utilities.h"


int main( int argc, char ** argv ) {

//  typedef char PixelType;
//  typedef char OutputPixelType;
  typedef unsigned short PixelType;
  typedef unsigned short OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< PixelType, Dimension >		ImageType;
  typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;
  typedef OutputImageType::RegionType OutputImageRegionType;
  
  typedef ImageType::ConstPointer   InputImageConstPointer;


// The image type is used as a template parameter to instantiate
// the reader and writer.

  typedef itk::ImageFileReader < ImageType >  ReaderType;
  typedef itk::ImageFileWriter < OutputImageType >  WriterType;
  
  // Iterator Typedefs for this routine
  typedef itk::ImageRegionConstIterator< ImageType >       InputIterator;
  typedef itk::ImageRegionIterator< OutputImageType >      OutputIterator;


  
  if( argc < 4 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile outputImageFile newVal" << std::endl;
    return EXIT_FAILURE;
    }  
  
  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  
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

  InputImageConstPointer  inputPtr  = reader->GetOutput();
  
  // How big is the input image?
  
  ImageType::SizeType size = inputPtr->GetRequestedRegion().GetSize();
  ImageType::IndexType startIndex = inputPtr->GetRequestedRegion().GetIndex();

  std::cout << "size, startIndex: " << size << ", " << startIndex << std::endl;
  
  OutputImageRegionType Region;
  Region.SetSize(size);
  Region.SetIndex(startIndex);
  OutputImageType::Pointer imagen           =  OutputImageType::New();
  
  imagen->SetRegions( Region );
  imagen->Allocate();
  
 
  InputIterator  in_it  = InputIterator (inputPtr, inputPtr->GetRequestedRegion());
  
  OutputIterator out_it =  OutputIterator(imagen, Region);

  int newVal = atoi(argv[3]);

  std::cout << "Substitution value is " << argv[3] << " -> " << newVal << std::endl;
  long c = 0;
  
  for ( in_it.GoToBegin(), out_it.GoToBegin(); !in_it.IsAtEnd(), !out_it.IsAtEnd(); ++in_it, ++out_it ) {
    //std::cout << in_it.Get() << " ";
    if (in_it.Get() != 0) {
      out_it.Set(newVal);
      c++;
    }
  }
  std::cout << "Substituted " << c << " pixels" << std::endl;

  copy_dicom_data(reader->GetOutput(), imagen);
  
   writer->SetFileName( argv[2] );
   writer->SetInput( imagen );

//  Finally, execution of the pipeline can be triggered by invoking the
//  Update() method in the writer. This call must be placed in a try/catch
//  block since exceptions be potentially be thrown in the process of reading
//  or writing the images

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
// Software Guide : EndCodeSnippet

  return EXIT_SUCCESS;
}






