
#ifndef __itkLabelPipelineImageFilter_txx
#define __itkLabelPipelineImageFilter_txx

#include "vnl/vnl_vector_fixed.h"
#include "itkProgressReporter.h"
#include "itkSize.h"
#include "itkImageRegion.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionReverseIterator.h"


#include "itkLabelPipelineImageFilter.h"


namespace itk
{

template< class TInputImage, class TOutputImage >
LabelPipelineImageFilter< TInputImage, TOutputImage >
::LabelPipelineImageFilter()
{
  itkDebugMacro(<< "LabelPipelineImageFilter::LabelPipelineImageFilter() called");


  m_medianRadius.Fill(1);
  m_Backup = 0;

  m_thresholdActual = 0;
  m_tThreshold = 0;
  m_tMedian = 0;
  m_tLabeling = 0;
}

template< class TInputImage, class TOutputImage >
void
LabelPipelineImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion()
{
  
}


template< class TInputImage, class TOutputImage >
void
LabelPipelineImageFilter< TInputImage, TOutputImage >
::GenerateData()
{

  // Get the input and output pointers
  InputImageConstPointer  inputPtr  = this->GetInput(0);
  OutputImagePointer      outputPtr = this->GetOutput(0);

  typedef itk::MetaDataDictionary   DictionaryType;
  const DictionaryType & dictionary = inputPtr->GetMetaDataDictionary();

  itk::Vector<double, 3> spacing;
  spacing = inputPtr->GetSpacing(); 
	


	

	
  
   


    this->GraftOutput( masker->GetOutput() );

  }
  
}

template< class TInputImage, class TOutputImage >
void
LabelPipelineImageFilter< TInputImage, TOutputImage >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
    
}

} // end namespace

#endif
