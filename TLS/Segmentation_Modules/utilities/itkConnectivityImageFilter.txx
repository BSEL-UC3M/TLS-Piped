#ifndef __itkConnectivityImageFilter_txx
#define __itkConnectivityImageFilter_txx

#include "vnl/vnl_vector_fixed.h"
#include "itkProgressReporter.h"
#include "itkSize.h"
#include "itkImageRegion.h"
#include "itkConnectivityImageFilter.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"

namespace itk
{

template< class TInputImage, class TOutputImage >
ConnectivityImageFilter< TInputImage, TOutputImage >
::ConnectivityImageFilter()
{
  itkDebugMacro(<< "ConnectivityImageFilter::ConnectivityImageFilter() called");

  /*

  		 .X.       (-1,-1)(0,-1)(+1,-1)
  		 X.X       (-1, 0)(0, 0)(+1, 0)
  		 .X.       (-1,+1)(0,+1)(+1,+1)

  		*/

  if (TInputImage::ImageDimension == 2) {
	  	 OType offset1 = {{0,-1}};
	  	 OType offset2 = {{-1,0}};
	  	 OType offset3 = {{+1,0}};
	  	 OType offset4 = {{0,+1}};

		 this->offsets.push_back(offset1);
		 this->offsets.push_back(offset2);
		 this->offsets.push_back(offset3);
		 this->offsets.push_back(offset4);


	} else if (TInputImage::ImageDimension == 3) {


		 OType offset1 = {{0,-1,0}};
		 OType offset2 = {{-1,0,0}};
		 OType offset3 = {{+1,0,0}};
		 OType offset4 = {{0,+1,0}};
		 OType offset5 = {{0,0,-1}};
		 OType offset6 = {{0,0,+1}};

		 this->offsets.push_back(offset1);
		 this->offsets.push_back(offset2);
		 this->offsets.push_back(offset3);
		 this->offsets.push_back(offset4);
		 this->offsets.push_back(offset5);
		 this->offsets.push_back(offset6);

		}



}

template< class TInputImage, class TOutputImage >
void
ConnectivityImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion()
{
  
}


template< class TInputImage, class TOutputImage >
void
ConnectivityImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
  itkDebugMacro(<< "ConnectivityImageFilter::GenerateData() called");

  // Get the input and output pointers
  InputImageConstPointer  inputPtr  = this->GetInput(0);
  OutputImagePointer      outputPtr = this->GetOutput(0);
  
  // How big is the input image?
  typename TInputImage::SizeType size=inputPtr->GetRequestedRegion().GetSize();
  typename TInputImage::IndexType startIndex=inputPtr->GetRequestedRegion().GetIndex();
  int dimension=inputPtr->GetRequestedRegion().GetImageDimension();
  
  // Allocate the output
  OutputImageRegionType Region;
  Region.SetSize(size);
  Region.SetIndex(startIndex);
  outputPtr->SetRegions( inputPtr->GetLargestPossibleRegion() );
  outputPtr->Allocate();
  

  typedef itk::ImageRegionIterator<TInputImage>       ImageIterator;
  // A radius of 1 in all axial directions gives a 3x3x3x3x... neighborhood.

  typename NeighborhoodIterator::RadiusType radius;
      for (unsigned int i = 0; i < TInputImage::ImageDimension; ++i) radius[i] = 1;

      NeighborhoodIterator it(radius, inputPtr, outputPtr->GetRequestedRegion());

     typename std::vector<OType>::iterator offsets_it;
     for ( offsets_it=this->offsets.begin() ; offsets_it < this->offsets.end(); offsets_it++ )
    	 it.ActivateOffset(*offsets_it);

      // Initializes the iterators on the input & output image regions

      ImageIterator out(outputPtr, outputPtr->GetRequestedRegion());

      // Iterates over the input and output
      for (it.Begin(), out = out.Begin(); ! it.IsAtEnd(); ++it, ++out )
      {
          if (it.GetCenterPixel() != 0){

              //std::cout << "In: "  << it.GetIndex() <<  " = " << it.GetCenterPixel() << std::endl;

              int accum = 0;
              //for (unsigned int i = 0; i < it.Size(); ++i)
              for ( offsets_it=this->offsets.begin() ; offsets_it < this->offsets.end(); offsets_it++ )
                  	 accum += it.GetPixel(*offsets_it);

              //std::cout << "OUT: " << out.GetIndex() << " = " << accum << std::endl;
              //char str[101];
              //std::cin.getline(str, 101);
              // Remove center!
              if (accum > 0)
                  out.Set( accum );
          }


      }

      //this->GraftOutput(outputPtr);
	}

template< class TInputImage, class TOutputImage >
void
ConnectivityImageFilter< TInputImage, TOutputImage >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  
  os << indent << std::endl;
  
}

} // end namespace

#endif
