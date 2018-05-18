
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkLabelMapMaskImageFilter.h"

#include <stdio.h>
#include <string>


int main( int argc, char * argv[] )
{
  if (argc < 2){
    std::cout << "Usage: " << argv[0] << " serie_dir [backup] [label]" << std::endl;
    return EXIT_SUCCESS;
  }
  
  if ( (strcmp(argv[1],"--help") == 0) || ( (strcmp(argv[1],"-h") == 0) ) ){
    std::cout << "Usage: " << argv[0] << " serie_dir [backup] [label]" << std::endl;
    return EXIT_SUCCESS;
  }
  int m_backup = 0; int m_label = 0;
  if (argc >= 3)
  {
    m_backup = atoi(argv[2]);
    
    std::cout << "Backup is: " << m_backup << std::endl;
  }
  
  if (argc >= 4)
  {
    m_label = atoi(argv[3]);
    
    std::cout << "Label " << m_backup << " is selected from command line" << std::endl;
  }

  
  char * infilename  = argv[1];
  
  std::cout << "Reading input serie  " << infilename << std::endl;
  

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
  
  typedef ImageType::Pointer ImagePointer;
  ImagePointer inputPtr = reader->GetOutput();

   // How big is the input image?
  RegionType inputRegion = inputPtr->GetLargestPossibleRegion();
  SizeType size = inputRegion.GetSize();
  IndexType startIndex = inputRegion.GetIndex();

  typedef itk::Point< double, Dimension > PointType;
  
  typedef unsigned long LabelType;
  typedef itk::ShapeLabelObject< LabelType, 3 >  LabelObjectType;
  typedef itk::LabelMap< LabelObjectType > LabelCollectionType;  
  typedef itk::LabelMapMaskImageFilter< LabelCollectionType,ImageType > MaskType;

  std::vector<LabelObjectType::Pointer> candids;
  std::vector<ImageType::IndexType> excludedPoints;
  
  std::vector<LabelObjectType::Pointer>::const_iterator cand;
  std::vector<ImageType::IndexType>::const_iterator p;
  
  //If the labelled objects contains those points it will be skipped...
  ImageType::IndexType bkg_point; //Background
  bkg_point[0] = 0;   bkg_point[1] = 0;   bkg_point[2] = 0;
  excludedPoints.push_back(bkg_point);
  float lowerBound = 2000;

  LabelCollectionType::LabelObjectContainerType::const_iterator it;


  const LabelCollectionType::LabelObjectContainerType & labelObjectContainer = collection->GetLabelObjectContainer();

  for( it = labelObjectContainer.begin(); it != labelObjectContainer.end(); it++ )
  {

    //const PixelType & label = it->first;  
    LabelObjectType * l = it->second;
    std::cout << "    Considering object " << l->GetLabel() << " with size " << l->GetSize() << std::endl;
        
  
//  for (int i=0;i<collection->GetNumberOfLabelObjects();i++) {
//    LabelObjectType::Pointer l = collection->GetLabelObject(i+1);
//    std::cout << "    Considering object " << l->GetLabel() << " with size " << l->GetSize() << std::endl;
    ImageType::RegionType region = l->GetRegion();
    bool isOk = -1;
    for(p=excludedPoints.begin(); p!=excludedPoints.end(); ++p)
      if (region.IsInside((*p))){
        //std::cout << "    Skipping label " << l->GetLabel() << " because is the image's background" << std::endl;
        isOk = 0;
        break;
      }
    if (isOk && (l->GetSize() > lowerBound))
      candids.push_back(l);
    else {
      std::cout << "    Skipping label " << l->GetLabel() << " because ";
      if (!isOk)
      	std::cout << "is in the background" << std::endl;
      else if (l->GetSize() <= lowerBound)
        std::cout << "has little size (<" << lowerBound << ")" << std::endl;
      else
        std::cout << "don't know!!" << std::endl;
    }
  }
        
  //print "Candidates for lungs are:",[str(c.GetLabel()) for c in candids]
        
  float cx = inputPtr->GetLargestPossibleRegion().GetSize()[0] /2.0;
  float cy = inputPtr->GetLargestPossibleRegion().GetSize()[1] /2.0;
  float cz = inputPtr->GetLargestPossibleRegion().GetSize()[2] /2.0;
  std::cout <<  "The center of the image is: "<< cx << "," << cy << "," << cz << std::endl;
  float minD = 10000;

  LabelObjectType::Pointer minL = NULL;
  
  std::cout <<  "Number of candidates: "<< candids.size() << std::endl;
  
  
  for (cand=candids.begin();cand!=candids.end();++cand) {
    //cand->Print(std::cout);
    
    /* Try to exclude the bed region in the lowest part of the image */
    int y_obj_start = (*cand)->GetRegion().GetIndex()[1];
    
    if (y_obj_start > cy) {
      std::cout << " The y start point for this object is very low...skipping because it can be the bed..." << std::endl;
      continue;
      }
    
    float x = (*cand)->GetCentroid()[0];
    float y = (*cand)->GetCentroid()[1];
    float z = (*cand)->GetCentroid()[2];
    float d = pow(pow(cx-x,2)+pow(cy-y,2)+pow(cz-z,2),0.5);
    std::cout << " with distance " << d << std::endl;
    
    std::cout << "Object " << (*cand)->GetLabel() << " " << (*cand)->GetSize() << " " << (*cand)->GetCentroid();
    std::cout << " with distance " << d << std::endl;
    std::cout << "Deltas: cx-x:" << (cx-x) << " cy-y:" << (cy-y) << " cz-z:" << (cz-z) << std::endl;
    
    if (d < minD)
    {
      minL = (*cand);
      minD = d;
    }
  }
    
  if (minL.IsNotNull())
  {
    std::cout << "    Chosen object with distance " << minD << std::endl;
    std::cout << "    Lungs should be " << minL->GetLabel() << " " << minL->GetSize() << " " << minL->GetCentroid() << std::endl;
    std::cout << "    Outputting original and binarized version of the lungs " << std::endl;
      
    MaskType::Pointer masker = MaskType::New();

    	
    masker->SetInput(collection);    
    masker->SetFeatureImage(inputPtr);
    masker->SetBackgroundValue(0);
    masker->SetLabel(minL->GetLabel());

    WriterType::Pointer writer = WriterType::New();
    writer->SetInput( masker->GetOutput() );
    writer->SetFileName("lungs-masked.dcm");
    writer->Update();
  } else
	std::cout << " No suitable object found! " << std::endl;

  return EXIT_SUCCESS;
  
}

