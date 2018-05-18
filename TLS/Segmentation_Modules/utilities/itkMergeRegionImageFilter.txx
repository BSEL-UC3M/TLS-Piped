#ifndef __itkMergeRegionImageFilter_txx
#define __itkMergeRegionImageFilter_txx

#include "itkMergeRegionImageFilter.h"
#include <itkNumericTraitsFixedArrayPixel.h>

namespace itk
{

template< class TInputImage, class TOutputImage >
MergeRegionImageFilter< TInputImage, TOutputImage >
::MergeRegionImageFilter()
{
  itkDebugMacro(<< "MergeRegionImageFilter::MergeRegionImageFilter() called");
}

/*
template< class TInputImage, class TOutputImage >
void
MergeRegionImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion()
{
  
}
*/

template< class TInputImage, class TOutputImage >
void
MergeRegionImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
  itkDebugMacro(<< "MergeRegionImageFilter::GenerateData() called");

  // Get the input and output pointers
  InputImageConstPointer  inputPtr  = this->GetInput(0);
  OutputImagePointer      outputPtr = this->GetOutput(0);
  
  //std::cout << "Original image " << inputPtr << std::endl;
  //std::cout << "Merge image " << m_MergeImage << std::endl;
  std::cout << "Merge region " << m_MergeRegion << std::endl;
  
  typedef typename itk::ImageDuplicator<TInputImage> DupType;
  typename DupType::Pointer dup = DupType::New();
  dup->SetInputImage(inputPtr);
  dup->Update();
  InputImagePointer output = dup->GetOutput();
  
  // How big is the input image?
  
  typename TInputImage::SizeType size=inputPtr->GetRequestedRegion().GetSize();
  typename TInputImage::IndexType startIndex=inputPtr->GetRequestedRegion().GetIndex();
    
 	typedef ImageRegionConstIterator< TInputImage > RegionConstIteratorType;
  RegionConstIteratorType merge_image_it( m_MergeImage, m_MergeImage->GetLargestPossibleRegion() );

  typedef ImageRegionIterator< TInputImage > RegionIteratorType;
  RegionIteratorType output_it( output, m_MergeRegion );
  
  for(merge_image_it.GoToBegin(),output_it.GoToBegin(); !merge_image_it.IsAtEnd(); ++merge_image_it, ++output_it)
  { 
    //We change the speed value only if its more than 0 (We want to set speed to 0 and never come up!)
    //FIXME: put a threshold: if speeds decreases below 0.7 put 0.0
    if (output_it.Get() > itk::NumericTraits< float >::min())
    {
      if (merge_image_it.Get() > m_ProbThreshold){
        output_it.Set(merge_image_it.Get());
      } else {
        output_it.Set( 0.0 );
      }
    }
//    else
  //    std::cout << "" << std::endl;
    
  }
  
  this->GraftOutput(output);  
  /*
  *  Jump to slice m_MergeRegion.index[2] of output image
  *  start copying m_MergeImage into output image   
  */
  
  
  
  
  

  
   
}

template< class TInputImage, class TOutputImage >
void
MergeRegionImageFilter< TInputImage, TOutputImage >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  
  os << indent << std::endl;
  
}

} // end namespace

#endif
