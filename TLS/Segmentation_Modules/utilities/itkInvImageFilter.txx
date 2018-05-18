#ifndef __itkInvImageFilter_txx
#define __itkInvImageFilter_txx

#include "vnl/vnl_vector_fixed.h"
#include "itkProgressReporter.h"
#include "itkSize.h"
#include "itkImageRegion.h"
#include "itkInvImageFilter.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"

namespace itk
{

template< class TInputImage, class TOutputImage >
InvImageFilter< TInputImage, TOutputImage >
::InvImageFilter()
{
  itkDebugMacro(<< "InvImageFilter::InvImageFilter() called");
}

template< class TInputImage, class TOutputImage >
void
InvImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion()
{
  
}


template< class TInputImage, class TOutputImage >
void
InvImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
  itkDebugMacro(<< "InvImageFilter::GenerateData() called");

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
  
  // Iterator Typedefs for this routine
  typedef ImageRegionConstIterator<TInputImage>                           InputIterator;
  typedef ConstNeighborhoodIterator< TInputImage >                        ConstNeighborhoodIteratorType;
  typedef ImageSliceConstIteratorWithIndex< TOutputImage >                SliceConstIteratorType;
  typedef ImageSliceIteratorWithIndex< TOutputImage >                     SliceIteratorType;
  
  SliceConstIteratorType in(  inputPtr,  inputPtr ->GetLargestPossibleRegion() );
  SliceIteratorType     out( outputPtr, outputPtr->GetLargestPossibleRegion() );
  in.SetFirstDirection(0); in.SetSecondDirection(1);
  out.SetFirstDirection(0);out.SetSecondDirection(1);
  
  typename TInputImage::IndexType                                         Indice;
  Indice[0]=0;
  Indice[1]=0;

  Indice[2]=size[2]-1;
  
  in.SetIndex(Indice);
  out.GoToBegin();
  while (!in.IsAtEnd())
	  {
	  while (!in.IsAtEndOfSlice())
	 	  {
	  	while (!in.IsAtEndOfLine())
	  	  {
   	  	/*std::cout << " -> (" << in.GetIndex().GetElement(0) << "," << in.GetIndex().GetElement(1) << "," << in.GetIndex().GetElement(2) << ")";
	      std::cout << " <- (" << out.GetIndex().GetElement(0) << "," << out.GetIndex().GetElement(1) << "," << out.GetIndex().GetElement(2) << ")";
      	std::cout << std::endl;
*/
	  	  //in.Get();
  		  out.Set(in.Get());
	   		++out;
	   		++in;
	   		}
	  	in.NextLine();
	  	out.NextLine();
	  	}
   	in.PreviousSlice();
   	Indice = in.GetIndex();
    Indice[1]=0;  //Go to pixel 0
    in.SetIndex(Indice); 
   	out.NextSlice();
	  }
	}

template< class TInputImage, class TOutputImage >
void
InvImageFilter< TInputImage, TOutputImage >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  
  os << indent << std::endl;
  
}

} // end namespace

#endif
