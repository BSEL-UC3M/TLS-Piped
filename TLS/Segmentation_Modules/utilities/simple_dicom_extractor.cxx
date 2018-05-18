
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"

#include "itkImage.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileReader.h"

#include <stdio.h>
#include <string>
#include <time.h>
#include <list>
#include <iostream>
#include <fstream>
#include <vector>

int main( int argc, char * argv[] )
{

  char * infilename  = argv[1];
  
  
  typedef signed short PixelType;
  const unsigned int Dimension=3;

  typedef unsigned long LabelType;

  typedef itk::Image< PixelType, Dimension >                       	ImageType;

	
	typedef ImageType::IndexType                                     	IndexType;
  typedef ImageType::SizeType                                      	SizeType;
  typedef ImageType::RegionType                                    	OutputImageRegionType;
 
  typedef itk::ImageSeriesReader< ImageType >                      	ReaderType;
	ReaderType::Pointer reader =                                     	ReaderType::New();
	
  typedef itk::ImageFileReader< ImageType >                      	      SimpleReaderType;
 	SimpleReaderType::Pointer simple_reader =                               	SimpleReaderType::New();

//--------- Reading the serie --------------
  
  typedef itk::GDCMImageIO                                         ImageIOType;
	ImageIOType::Pointer dicomIO =                                   ImageIOType::New();
	
  simple_reader->SetFileName( infilename );
  simple_reader->SetImageIO( dicomIO );
  
  try
    {
      simple_reader->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught!" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  
  //Datos del paciente
  typedef itk::MetaDataDictionary   DictionaryType;
  const  DictionaryType & dictionary = dicomIO->GetMetaDataDictionary();

  typedef itk::MetaDataObject< std::string > MetaDataStringType;
  
  DictionaryType::ConstIterator itr = dictionary.Begin();
  DictionaryType::ConstIterator end = dictionary.End();

  while( itr != end )
  {
  itk::MetaDataObjectBase::Pointer  entry = itr->second;
  MetaDataStringType::Pointer entryvalue = dynamic_cast<MetaDataStringType *>( entry.GetPointer() ) ;
  if( entryvalue )
    {
    std::string tagkey = itr->first;
    std::string tagname;
    dicomIO->GetLabelFromTag( tagkey, tagname);
    std::string tagvalue = entryvalue->GetMetaDataObjectValue();
    std::cout << tagkey << ":" << tagname << ":" << tagvalue << std::endl;
    }
    ++itr;
  }
  

}
