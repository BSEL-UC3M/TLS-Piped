
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "itkImage.h"
#include "itkImageFileReader.h"
//#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"

#include <stdio.h>
#include <string>
#include <time.h>
#include <list>
#include <fstream>

int main( int argc, char * argv[] )
{
  char * infilename  = argv[1];

  time_t inicio;
  inicio=time(NULL);
  typedef unsigned short PixelType; //Label type
  const unsigned int Dimension=3;
  typedef itk::Image< PixelType, Dimension >                       ImageType;
    
  typedef itk::ImageRegionConstIterator<ImageType>                 ConstIteratorType;
  //typedef itk::ImageRegionIterator<ImageType>                      IteratorType;
	typedef itk::ImageFileReader<ImageType>                          ReaderType;
	//typedef itk::ImageFileWriter<ImageType>                          WriterType;

  ReaderType::Pointer reader =                                     ReaderType::New();
  //WriterType::Pointer writer =                                     WriterType::New();
  
  typedef ImageType::IndexType                                     IndexType;
  typedef ImageType::SizeType                                      SizeType;
  typedef ImageType::RegionType                                    OutputImageRegionType;
	
	reader->SetFileName( infilename );
  reader->Update();
  std::cout << "Leida la imagen en: " << time(NULL)-inicio << "s." << std::endl;
  
  int dimension=reader->GetOutput()->GetRequestedRegion().GetImageDimension();
  short finded1Label = 0;
  int label = 0;
  int xMean = 0;
  int yMean = 0;
  int nPoints = 0;
  
  IndexType start;
  start = reader->GetOutput()->GetLargestPossibleRegion().GetIndex();
  SizeType size;
  size = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  size[2] = 1;
  ImageType::RegionType desiredRegion;
  desiredRegion.SetSize( size );
  desiredRegion.SetIndex( start );
  
  if (dimension==3)
  {
    ConstIteratorType in(reader->GetOutput(), desiredRegion);
    for(in.GoToBegin();!in.IsAtEnd();++in)
    {
      if(in.Get()!=0)
      {
        if(label==0)
          label=in.Get();
        if(label==in.Get())
        {
          finded1Label=1;
          xMean += in.GetIndex()[0];
          yMean += in.GetIndex()[1];
          nPoints++;
        }
        else
        {
          finded1Label=2;
        }
      }
    }
  }

  if(nPoints != 0&&finded1Label==1)
  {
    if(((xMean/nPoints)<200)||((xMean/nPoints)>300)||((yMean/nPoints)<200)||((yMean/nPoints)>300))
      finded1Label = 3;
  }

  if(finded1Label==0)
  	std::cout << "Traquea not founded" << std::endl;
  else if(finded1Label==1)
  	std::cout << "Finded traquea" << std::endl;
  else if(finded1Label==2)
  	std::cout << "Finded more than one posible traquea" << std::endl;
  else if(finded1Label==3)
  	std::cout << "Finded something like a traquea but out of the center of the image" << std::endl;
  else
  	std::cout << "Unexpected error" << std::endl;
  std::cout << "Total time: " << time(NULL)-inicio << "s." <<std::endl;   
  
  return finded1Label;
}

