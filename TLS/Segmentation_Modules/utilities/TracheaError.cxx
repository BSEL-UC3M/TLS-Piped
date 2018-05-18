
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "itkImage.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkConnectedComponentImageFilter.h"

#include <stdio.h>
#include <string>
#include <list>
#include <fstream>

int main( int argc, char * argv[] )
{
  char * infilename  = argv[1];

  typedef short PixelType; //Label type
  const int Dimension=3;
  typedef itk::Image< PixelType, Dimension >                       ImageType;
  typedef itk::Image< PixelType, 2 >                       				 SliceType;
  typedef itk::Image< bool, 2 >                       						 BoolSliceType;
    
  typedef itk::ImageRegionConstIterator<ImageType>                 ConstIteratorType;
  typedef itk::ImageRegionConstIterator<SliceType>                 ConstSliceIteratorType;
  typedef itk::ImageRegionIterator<BoolSliceType>                  IteratorBoolSliceType;
  typedef itk::ImageSeriesReader< ImageType >                      ReaderType;
	typedef itk::ImageFileWriter<SliceType>                          WriterType;

  ReaderType::Pointer reader =                                     ReaderType::New();
  WriterType::Pointer writer =                                     WriterType::New();
  
	typedef itk::ConnectedComponentImageFilter< BoolSliceType,SliceType > ConnectedFilter;
	ConnectedFilter::Pointer connected =                             ConnectedFilter::New();
	
  typedef ImageType::IndexType                                     IndexType;
  typedef ImageType::SizeType                                      SizeType;
  typedef BoolSliceType::IndexType                                 SliceIndexType;
  typedef BoolSliceType::SizeType                                  SliceSizeType;
  typedef ImageType::RegionType                                    OutputImageRegionType;
  
	typedef itk::GDCMImageIO                                         ImageIOType;
	ImageIOType::Pointer dicomIO =                                   ImageIOType::New();
		
	typedef itk::GDCMSeriesFileNames NamesGeneratorType;
	NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
	nameGenerator->SetUseSeriesDetails( true );
	nameGenerator->SetDirectory(infilename);
	typedef std::vector<std::string> seriesIdContainer;
  const seriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
	
  seriesIdContainer::const_iterator seriesItr = seriesUID.begin();
  seriesIdContainer::const_iterator seriesEnd = seriesUID.end();
  
  std::string seriesIdentifier = seriesUID.begin()->c_str();
	typedef std::vector<std::string> fileNamesContainer;
  fileNamesContainer fileNames;
  fileNames = nameGenerator->GetFileNames(seriesIdentifier);
	reader->SetImageIO(dicomIO);
	reader->SetFileNames(fileNames);
	reader->Update();
	
  //Bucle para buscar el mejor Ti
  float ub, un, Nb, Nn, Tii, Tiii;
  Tii = 0; Tiii = 0;
  
  //Creamos la mascara
  unsigned int vector[10000];
  for (int i = 0; i < 10000; i++)
    vector[i]  = 0;

  int max=0;  
  float  Ti=0;       
	  
  int dimension=reader->GetOutput()->GetImageDimension();
  short tracheaError = 1;
  /**********************************************************************************/
  /* 0 if it finds one labeled objet in the center of the image											*/
  /* 1 if it dosn't find a objet or if it is not centred.														*/
  /**********************************************************************************/
  
  int xMean[10000];
  int yMean[10000];
  int nPoints[10000];
  for (int i = 0; i < 10000; i++)
  {
    xMean[i] = 0;
    yMean[i] = 0;
    nPoints[i] = 0;
  }
  
  SizeType size;
  size = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  if (dimension==3)
    size[2] = 1;
  ImageType::RegionType desiredRegion;
  desiredRegion.SetSize( size );
  desiredRegion.SetIndex(reader->GetOutput()->GetLargestPossibleRegion().GetIndex());
  
  ConstIteratorType in(reader->GetOutput(), desiredRegion);
  //As a first step we get the histogram of the first slice
  for (in.GoToBegin(); !in.IsAtEnd(); ++in )
  {
    vector[in.Get()+1024]++;
    Ti += in.Get()+1024;
    if (in.Get()+1024>max)
      max=in.Get()+1024;
  }
  Ti=Ti/(size[0]*size[1]);
  //We stimate now the HU Threshold
  int loops = 0;
  while ((fabs(Tii-Ti)>0.1)&&(fabs(Tiii-Ti)>0.1)&&(loops<100))
  {
  	loops++;
    Tiii = Tii;
    Tii = Ti;
    ub=0;un=0;
    Nn=0;Nb=0;
    for (float i = 0; i < Ti; i++)
    {
      un+=i*vector[(int)i];
      Nn+=vector[(int)i];
    }
    for (float i = Ti; i < max; i++)
    {
      ub+=i*vector[(int)i];
      Nb+=vector[(int)i];
    }
    if (Nb!=0) ub=ub/Nb;
    else ub=0;
    if (Nn!=0) un=un/Nn;
    else un=0;
    Ti=(ub+un)/2;
  }
  if (loops == 100) std::cout << "\x1b[91mNot converged (Ti = " << Ti << ", Tii = " << Tii << ", Tiii = " << Tiii << ")\x1b[m" << std::endl;
  
  //We save the threshold of the first slice
  BoolSliceType::Pointer thresholded =  BoolSliceType::New();
  BoolSliceType::RegionType sliceRegion;
  SliceSizeType sliceSize;
  sliceSize[0] = size[0];
  sliceSize[1] = size[1];
  sliceRegion.SetSize(sliceSize);
  IndexType start;
  start = reader->GetOutput()->GetLargestPossibleRegion().GetIndex();
  SliceIndexType sliceStart;
  sliceStart[0] = start[0];
  sliceStart[1] = start[1];
  sliceRegion.SetIndex(sliceStart);
  thresholded->SetRegions(sliceRegion);
  thresholded->Allocate();
  
  IteratorBoolSliceType thres(thresholded, sliceRegion);
  for(in.GoToBegin(),thres.GoToBegin();!in.IsAtEnd(),!thres.IsAtEnd();++in,++thres)
  	thres.Set(in.Get()<Ti-1024);
  
  //We label the first slice
  connected->SetInput(thresholded);
  connected->Update();
  
  //We search for those labels centered in the image
  ConstSliceIteratorType lab(connected->GetOutput(), sliceRegion);
  max = 0;
  for(lab.GoToBegin();!lab.IsAtEnd();++lab)
  {
  	xMean[lab.Get()] += lab.GetIndex()[0];
    yMean[lab.Get()] += lab.GetIndex()[1];
    nPoints[lab.Get()]++;
    if (lab.Get()>max) max = lab.Get();
  }
  
  writer->SetInput(connected->GetOutput());
  writer->SetFileName("FistSliceLabeled.dcm");
  writer->Update();
  
	for (int i=1;i<max;i++)
	{
	  if(((xMean[i]/nPoints[i])<200)||((xMean[i]/nPoints[i])>300)||((yMean[i]/nPoints[i])<200)||((yMean[i]/nPoints[i])>300))
    {
      /*std::cout << "\x1b[91mUnknown objet with label " << i << " might not be the trachea." << std::endl;
      std::cout << "    Its position is " << xMean[i]/nPoints[i] << ", " << yMean[i]/nPoints[i] << std::endl;
      std::cout << "    Its size is " << nPoints[i] << "\x1b[m" << std::endl;*/
    }
    else
    {
      tracheaError = 0;
      std::cout << "\x1b[92mPossible trachea with label " << i << std::endl;
      std::cout << "    Its position is " << xMean[i]/nPoints[i] << ", " << yMean[i]/nPoints[i] << std::endl;
      std::cout << "    Its size is " << nPoints[i] << "\x1b[m" << std::endl;
    }
	}
  return tracheaError;
}

