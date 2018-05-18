
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
 

//--------- Reading the serie --------------
  
  typedef itk::GDCMImageIO                                         ImageIOType;
	ImageIOType::Pointer dicomIO =                                   ImageIOType::New();
	reader->SetImageIO( dicomIO );

	typedef itk::GDCMSeriesFileNames NamesGeneratorType;
	
	NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
	nameGenerator->SetUseSeriesDetails( true );

	nameGenerator->SetDirectory( argv[1] );
	typedef std::vector<std::string> seriesIdContainer;
  const seriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
	
  seriesIdContainer::const_iterator seriesItr = seriesUID.begin();
  seriesIdContainer::const_iterator seriesEnd = seriesUID.end();
  
  std::string seriesIdentifier;
	seriesIdentifier = seriesUID.begin()->c_str();
	
	typedef std::vector<std::string> fileNamesContainer;
  fileNamesContainer fileNames;
  fileNames = nameGenerator->GetFileNames(seriesIdentifier);
  reader->SetFileNames( fileNames );
  reader->SetImageIO( dicomIO );
  reader->Update();

  
  //Datos del paciente
  typedef itk::MetaDataDictionary   DictionaryType;
  const  DictionaryType & dictionary = dicomIO->GetMetaDataDictionary();

  typedef itk::MetaDataObject< std::string > MetaDataStringType;
  
  DictionaryType::ConstIterator itr = dictionary.Begin();
  DictionaryType::ConstIterator end = dictionary.End();

  while( itr != end )
  {
  itk::MetaDataObjectBase::Pointer  entry = itr->second;
  MetaDataStringType::Pointer entryvalue =
    dynamic_cast<MetaDataStringType *>( entry.GetPointer() ) ;
  if( entryvalue )
    {
    std::string tagkey;
    dicomIO->GetLabelFromTag( itr->first, tagkey);
    std::string tagvalue = entryvalue->GetMetaDataObjectValue();
    std::cout << tagkey.c_str() << ":" << tagvalue << std::endl;
    }
  ++itr;
  }
  

}
