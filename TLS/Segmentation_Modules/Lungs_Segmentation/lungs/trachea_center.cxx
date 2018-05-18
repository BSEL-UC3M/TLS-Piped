#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include "itkExtractImageFilter.h"

#include "itkImageRegionConstIterator.h"
#include "itkLabelImageToShapeLabelMapFilter.h"
#include "itkLabelMap.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryImageToLabelMapFilter.h"



typedef signed short PixelType;
const unsigned int Dimension = 3;
typedef itk::Image< PixelType, Dimension >		ImageType;
typedef itk::Image< PixelType, 2 >				ImageType_2D;

bool find_in_slice(ImageType_2D::Pointer, int*,int*);


int main( int argc, char ** argv )
{
  // Verify the number of parameters in the command line
  
  unsigned int sliceStart = 0;
  
  if( argc < 2 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile [startFromSlice]" << std::endl;
    return EXIT_FAILURE;
    }

  if( argc == 3 )
      sliceStart = atoi( argv[2] );


// The image type is used as a template parameter to instantiate
// the reader and writer.

  typedef itk::ImageFileReader < ImageType >  ReaderType;

  ReaderType::Pointer reader = ReaderType::New();
  
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


// Extract slice and grow trachea around the given seed to have initial wavefront
  typedef itk::ExtractImageFilter< ImageType, ImageType_2D > ExtractFilterType;
  ExtractFilterType::Pointer extracter = ExtractFilterType::New();
  
  ImageType::SizeType imageSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  ImageType::RegionType extractRegion;
  extractRegion.SetIndex( 0, 0);
  extractRegion.SetIndex( 1, 0);
  extractRegion.SetSize( 0, imageSize[0]);
  extractRegion.SetSize( 1, imageSize[1]);
  extractRegion.SetSize( 2, 0);
  
  extracter->SetInput( reader->GetOutput() );
  bool trachea_found = false;

  while( sliceStart < imageSize[2] ){
    extractRegion.SetIndex( 2, sliceStart );
    extracter->SetExtractionRegion( extractRegion );
    extracter->Update();
    
    int x,y; x = y = 0;
    
    if (find_in_slice(extracter->GetOutput(), &x,&y )) {
      std::cout << x << "," << y << "," << sliceStart << std::endl;
      trachea_found = true;
      break;
    } else 
      sliceStart++;
  
  }
  
  if (trachea_found) {
      return EXIT_SUCCESS;
   } else
      return EXIT_FAILURE;
}

bool find_in_slice(ImageType_2D::Pointer slice, int* x_c, int* y_c ) {
    

	typedef unsigned long LabelType;
	typedef itk::ShapeLabelObject< LabelType, 2 >  LabelObjectType;
	typedef itk::LabelMap< LabelObjectType > LabelCollectionType;
	  typedef itk::BinaryThresholdImageFilter<ImageType_2D, ImageType_2D>   ThresholdType;
	  //typedef itk::ImageFileWriter< ImageType_2D >                        	WriterType;
	  typedef itk::BinaryImageToLabelMapFilter< ImageType_2D, LabelCollectionType > ConverterType;
	  typedef itk::ShapeLabelMapFilter< LabelCollectionType > ShapeFilterType;

	  ThresholdType::Pointer threshold =               ThresholdType::New();
	  threshold->SetLowerThreshold(-1024);
	  threshold->SetUpperThreshold(-1);
	  threshold->SetInsideValue(1);
	  threshold->SetInput(slice);
	  threshold->Update();

//	  WriterType::Pointer writer = WriterType::New();
//	    writer->SetInput(slice);
//	    writer->SetFileName("test_trachea_slice.dcm");
//	    writer->Update();
//
//	    writer->SetInput(threshold->GetOutput());
//	    writer->SetFileName("test_trachea_bin.dcm");
//	    writer->Update();


	  ConverterType::Pointer converter = ConverterType::New();
	    converter->SetInput( threshold->GetOutput());
	    converter->SetInputForegroundValue( 1 );//Any nonzero value should be read as True
	    converter->SetFullyConnected( 0 );
	    converter->SetNumberOfThreads( 1 );
	    converter->Update();

		  ShapeFilterType::Pointer shape = ShapeFilterType::New();
		  shape->SetInput( converter->GetOutput() );
		  shape->Update();



	LabelCollectionType::Pointer collection = shape->GetOutput();

//	  LabelCollectionType::LabelObjectContainerType::const_iterator it;
//	  const LabelCollectionType::LabelObjectContainerType & labelObjectContainer = collection->GetLabelObjectContainer();
	  float minD = 10000;
	  LabelObjectType::Pointer minL = NULL;

	      // Use automatic detection

	      std::vector<LabelObjectType::Pointer> candids;
	      std::vector<ImageType_2D::IndexType> excludedPoints;

	      std::vector<LabelObjectType::Pointer>::const_iterator cand;
	      std::vector<ImageType_2D::IndexType>::const_iterator p;

	      //If the labeled objects contains those points it will be skipped...
	      ImageType_2D::IndexType bkg_point; //Background
	      bkg_point[0] = 0;   bkg_point[1] = 0;
	      excludedPoints.push_back(bkg_point);
	      float lowerBound = 20;

	      for(unsigned int i = 0; i < collection->GetNumberOfLabelObjects(); ++i)
	        {
	        LabelObjectType* l = collection->GetNthLabelObject(i);
	        //std::cout << "    Considering object " << l->GetLabel() << " with size " << l->GetSize() << std::endl;


	    //  for (int i=0;i<collection->GetNumberOfLabelObjects();i++) {
	    //    LabelObjectType::Pointer l = collection->GetLabelObject(i+1);
	    //    std::cout << "    Considering object " << l->GetLabel() << " with size " << l->GetSize() << std::endl;
	        ImageType_2D::RegionType region = l->GetBoundingBox();
	        bool isOk = -1;
	        for(p=excludedPoints.begin(); p!=excludedPoints.end(); ++p)
	          if (region.IsInside((*p))){
	            //std::cout << "    Skipping label " << l->GetLabel() << " because is the image's background" << std::endl;
	            isOk = 0;
	            break;
	          }
	        if (isOk && (l->GetNumberOfPixels() > lowerBound))
	          candids.push_back(l);
	        else {
	          /*std::cout << "    Skipping label " << l->GetLabel() << " because ";
	          if (!isOk)
	          	std::cout << "is in the background" << std::endl;
	          else if (l->GetSize() <= lowerBound)
	            std::cout << "has little size (<" << lowerBound << ")" << std::endl;
	          else
	            std::cout << "don't know!!" << std::endl;
	            */
	        }
	      }

	      //print "Candidates for lungs are:",[str(c.GetLabel()) for c in candids]

	      float cx = slice->GetLargestPossibleRegion().GetSize()[0]*slice->GetSpacing()[0] /2.0;
	      float cy = slice->GetLargestPossibleRegion().GetSize()[1]*slice->GetSpacing()[1] /2.0;
	      //std::cout <<  "The center of the image is: "<< cx << "," << cy << std::endl;

	      //std::cout <<  "Number of candidates: "<< candids.size() << std::endl;

	      for (cand=candids.begin();cand!=candids.end();++cand) {
	        //cand->Print(std::cout);



	        float x = (*cand)->GetCentroid()[0];
	        float y = (*cand)->GetCentroid()[1];
	        float d = pow(pow(cx-x,2)+pow(cy-y,2),0.5);
	        //std::cout << " with distance " << d << std::endl;

	        //std::cout << "Object " << (*cand)->GetLabel() << " " << (*cand)->GetSize() << " " << (*cand)->GetCentroid();
	        //std::cout << " with distance " << d << std::endl;
	        //std::cout << "Deltas: cx-x:" << (cx-x) << " cy-y:" << (cy-y)  << std::endl;

	        if (d < minD)
	        {
	          minL = (*cand);
	          minD = d;
	        }
	      }
	      if (minL.IsNotNull())
	      {
	        //std::cout << "    Chosen object with distance " << minD << std::endl;
	        //std::cout << "    Trachea should be " << minL->GetLabel() << " " << minL->GetSize() << " " << minL->GetCentroid() << std::endl;
		      /*FIXME: why use pixels instead of spacing?
		      	  but then we will have to correct and check in all subsequent filters...
		      	  Better to leave it off for now (Mario 2012)
		      	  */

	        *(x_c) = minL->GetCentroid()[0]/slice->GetSpacing()[0]; *(y_c) = minL->GetCentroid()[1]/slice->GetSpacing()[1];
	        return true;
	      }

	      return false;




    
    


}
