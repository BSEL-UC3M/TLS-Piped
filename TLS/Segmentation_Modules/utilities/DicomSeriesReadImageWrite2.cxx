/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: DicomSeriesReadImageWrite2.cxx,v $
  Language:  C++
  Date:      $Date: 2009-03-17 20:36:50 $
  Version:   $Revision: 1.17 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "dicom_utilities.h"
#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"


int main( int argc, char* argv[] )
{

  if( argc < 3 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " DicomDirectory  outputFileName  [seriesName]" 
              << std::endl;
    return EXIT_FAILURE;
    }

// Software Guide : BeginLatex
// 
// We define the pixel type and dimension of the image to be read. In this
// particular case, the dimensionality of the image is 3, and we assume a
// \code{signed short} pixel type that is commonly used for X-Rays CT scanners.
// 
// We also choose to use the \doxygen{OrientedImage} in order to make sure
// that the image orientation information contained in the direction cosines
// of the DICOM header are read in and passed correctly down the image processing
// pipeline.
//
// Software Guide : EndLatex

// Software Guide : BeginCodeSnippet
  typedef signed short    PixelType;
  const unsigned int      Dimension = 3;

  typedef itk::Image< PixelType, Dimension >         ImageType;

  typedef itk::ImageSeriesReader< ImageType >        ReaderType;
  ReaderType::Pointer reader = ReaderType::New();

  typedef itk::GDCMImageIO       ImageIOType;
  ImageIOType::Pointer dicomIO = ImageIOType::New();
  
  reader->SetImageIO( dicomIO );

// By default \code{SetUseSeriesDetails(true)} will use
// the following DICOM tags to sub-refine a set of files into multiple series:
// * 0020 0011 Series Number
// * 0018 0024 Sequence Name
// * 0018 0050 Slice Thickness
// * 0028 0010 Rows
// * 0028 0011 Columns
// If this is not enough for your specific case you can always add some more
// restrictions using the \code{AddSeriesRestriction()} method. In this example we will use
// the DICOM Tag: 0008 0021 DA 1 Series Date, to sub-refine each series. The format
// for passing the argument is a string containing first the group then the element
// of the DICOM tag, separed by a pipe (|) sign.

  typedef itk::GDCMSeriesFileNames NamesGeneratorType;
  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

  nameGenerator->SetUseSeriesDetails( true );
  nameGenerator->AddSeriesRestriction("0008|0021" );

  nameGenerator->SetDirectory( argv[1] );

  try
    {
    std::cout << std::endl << "The directory: " << std::endl;
    std::cout << std::endl << argv[1] << std::endl << std::endl;
    std::cout << "Contains the following DICOM Series: ";
    std::cout << std::endl << std::endl;


    typedef std::vector< std::string >    SeriesIdContainer;
    
    const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
    
    SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
    SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
    while( seriesItr != seriesEnd )
      {
      std::cout << seriesItr->c_str() << std::endl;
      seriesItr++;
      }

    std::string seriesIdentifier;

    if( argc > 3 ) // If no optional series identifier
      {
      seriesIdentifier = argv[3];
      }
    else
      {
      seriesIdentifier = seriesUID.begin()->c_str();
      }

    std::cout << std::endl << std::endl;
    std::cout << "Now reading series: " << std::endl << std::endl;
    std::cout << seriesIdentifier << std::endl;
    std::cout << std::endl << std::endl;


    typedef std::vector< std::string >   FileNamesContainer;
    FileNamesContainer fileNames;

    fileNames = nameGenerator->GetFileNames( seriesIdentifier );

    reader->SetFileNames( fileNames );

    try
      {
      reader->Update();
      }
    catch (itk::ExceptionObject &ex)
      {
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
      }

/*
    Copy input metadata dictionary

    typedef itk::MetaDataDictionary   DictionaryType;
 

    DictionaryType & dictionary = dicomIO->GetMetaDataDictionary();
    inputImage->SetMetaDataDictionary(dictionary);
 */
    ImageType::Pointer inputImage = reader->GetOutput();
    copy_dicom_data(reader->GetOutput(), inputImage);
 
    typedef itk::ImageFileWriter< ImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    
    writer->SetFileName( argv[2] );

    writer->SetInput( inputImage );

    std::cout  << "Writing the image as " << std::endl << std::endl;
    std::cout  << argv[2] << std::endl << std::endl;

    try
      {
      writer->Update();
      }
    catch (itk::ExceptionObject &ex)
      {
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
      }
    }
  catch (itk::ExceptionObject &ex)
    {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
