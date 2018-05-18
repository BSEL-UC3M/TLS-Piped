#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "itkImage.h"
#include "itkImageFileReader.h"

#include "itkLabelImageToShapeLabelMapFilter.h"
#include "itkShapeLabelObject.h"
#include "itkLabelMap.h"

#include <stdio.h>
#include <string>

int main( int argc, char * argv[] )
{
 
  char * infilename  = argv[1];
  
  typedef signed short PixelType;
  const unsigned int Dimension=3;

  typedef itk::Image< PixelType, Dimension >                       	ImageType;
	
  typedef ImageType::IndexType                                     	IndexType;
  typedef ImageType::SizeType                                      	SizeType;
 
  typedef itk::ImageFileReader< ImageType >                      	ReaderType;
  ReaderType::Pointer reader =                                     	ReaderType::New();
 
  reader->SetFileName( infilename );

  try {
    reader->Update();
  } catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught: can't read serie" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }
  
  ImageType::Pointer inputPtr = reader->GetOutput();
  
  typedef unsigned long LabelType;
  typedef itk::ShapeLabelObject< LabelType, Dimension >  LabelObjectType;
  typedef itk::LabelMap< LabelObjectType > LabelCollectionType;  
  typedef itk::LabelImageToShapeLabelMapFilter< ImageType, LabelCollectionType > ConverterType;

  ConverterType::Pointer converter = ConverterType::New();
  converter->SetInput( reader->GetOutput() );
  converter->SetBackgroundValue( 0 );
  converter->SetNumberOfThreads( 1 );
  converter->Update();

  LabelCollectionType::Pointer collection = converter->GetOutput();


  for(unsigned int i = 0; i < collection->GetNumberOfLabelObjects(); ++i)
  	{
  		LabelObjectType* l = collection->GetNthLabelObject(i);


  		std::cout << "    Considering object " << l->GetLabel() << " with size " << l->GetNumberOfPixels() << " and center " << l->GetCentroid() << std::endl;
                    
  	}
    

  return EXIT_SUCCESS;
  
}

