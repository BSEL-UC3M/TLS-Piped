#ifndef __itkMaskRegionImageFilter_txx
#define __itkMaskRegionImageFilter_txx


#include "itkMaskRegionImageFilter.h"


namespace itk
{
template< class TInputImage, class TOutputImage >
MaskRegionImageFilter< TInputImage, TOutputImage >
::MaskRegionImageFilter()
{

}


template< class TInputImage, class TOutputImage >
void
MaskRegionImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
  	
  // Get the input and output pointers
  InputImageConstPointer  inputPtr  = this->GetInput(0);

  // How big is the input image?
  RegionType inputRegion = inputPtr->GetLargestPossibleRegion();
  SizeType size = inputRegion.GetSize();
  IndexType startIndex = inputRegion.GetIndex();
  
  //Create output image
  
  OutputImagePointer out_image = OutputImageType::New();
  out_image->SetRegions(inputRegion);
  out_image->Allocate();
  out_image->SetSpacing(inputPtr->GetSpacing());
  
  ConstIteratorType in(inputPtr, m_RegionOfInterest);

  IteratorType outIt(out_image, m_RegionOfInterest);
 
  for (in.GoToBegin(), outIt.GoToBegin();!in.IsAtEnd(),!outIt.IsAtEnd();++in,++outIt)
  {
    outIt.Set( in.Get() );
  }
  
	this->GraftOutput(out_image);


}

template< class TInputImage, class TOutputImage >
void
MaskRegionImageFilter< TInputImage, TOutputImage >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
    
}

} // end namespace

#endif
