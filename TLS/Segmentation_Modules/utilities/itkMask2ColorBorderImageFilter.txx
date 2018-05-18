#ifndef __itkMask2ColorBorderImageFilter_txx
#define __itkMask2ColorBorderImageFilter_txx

/*#include "vnl/vnl_vector_fixed.h"
#include "itkProgressReporter.h"
#include "itkSize.h"
#include "itkImageRegion.h"*/
#include "itkImageRegionIterator.h"
//#include "itkImageRegionConstIterator.h"
#include "itkMask2ColorBorderImageFilter.h"
//#include "itkImageSliceConstIteratorWithIndex.h"
//#include "itkImageSliceIteratorWithIndex.h"
#include "itkConstNeighborhoodIterator.h"

namespace itk
{

template< class TInputImage, class TOutputImage >
Mask2ColorBorderImageFilter< TInputImage, TOutputImage >
::Mask2ColorBorderImageFilter()
{
  itkDebugMacro(<< "Mask2ColorBorderImageFilter::Mask2ColorBorderImageFilter() called");

  m_BorderSize = 1;

  m_OutName = "Prueba.tif";

  m_RedMask = 0;
  m_BlueMask = 0;
  m_GreenMask = 0;
}

template< class TInputImage, class TOutputImage >
void
Mask2ColorBorderImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion()
{
  
}


template< class TInputImage, class TOutputImage >
void
Mask2ColorBorderImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
  itkDebugMacro(<< "Mask2ColorBorderImageFilter::GenerateData() called");

  // Get the input and output pointers
  InputImageConstPointer  inputPtr  = this->GetInput(0);
  OutputImagePointer      outputPtr = this->GetOutput(0);

  // How big is the input image?
  RegionType inRegion = inputPtr->GetRequestedRegion();
  SizeType size = inRegion.GetSize();
  
  IndexType startIndex = inRegion.GetIndex();

  // Allocate the output
  outputPtr->SetRegions(inputPtr->GetLargestPossibleRegion());
  outputPtr->Allocate();

  typedef typename itk::BinaryBallStructuringElement<bool, NDimensions> StructuringElementType;
  StructuringElementType structuringElement;
  structuringElement.SetRadius(m_BorderSize);
  structuringElement.CreateStructuringElement();

  typedef typename itk::BinaryErodeImageFilter<BinImageType, BinImageType, StructuringElementType> DilateType;
  typename DilateType::Pointer dilate = DilateType::New();
  dilate->SetKernel(structuringElement);
  dilate->SetInput(m_MaskImage);

  typedef typename itk::SubtractImageFilter<BinImageType, BinImageType, BinImageType> subtractType;
  typename subtractType::Pointer subtract = subtractType::New();
  subtract->SetInput1(dilate->GetOutput());
  subtract->SetInput2(m_MaskImage);

  typedef typename itk::ImageRegionConstIterator<TInputImage>  inIt;
  typedef typename itk::ImageRegionConstIterator<BinImageType> maskIt;
  typedef typename itk::ImageRegionIterator<ITKRGBImageType>   outIt;

  try {
    subtract->Update();
    inIt in     = inIt(inputPtr, inRegion);
    maskIt mask = maskIt(subtract->GetOutput(), subtract->GetOutput()->GetLargestPossibleRegion());
    outIt out   = outIt(outputPtr, inRegion);

    inPixelType min=USHRT_MAX, max=0;
    for (in.GoToBegin();!in.IsAtEnd();++in) {
      if (in.Get()>max) max = in.Get();
      else if (in.Get()<min) min = in.Get();
    }

    RGBPixelType maskColour;
    maskColour.Set(m_RedMask, m_GreenMask, m_BlueMask);
    float factor = (float)(max-min)/UCHAR_MAX;
    for (in.GoToBegin(), mask.GoToBegin(), out.GoToBegin();!in.IsAtEnd(), !mask.IsAtEnd(),
        !out.IsAtEnd();++in, ++mask, ++out)
      if (mask.Get()) out.Set(maskColour);
      else out.Set((unsigned int)((in.Get()-min)/factor));

      typedef itk::ImageFileWriter<ITKRGBImageType> WriterType;
      typename WriterType::Pointer writer = WriterType::New();
      writer->SetInput(outputPtr);
      writer->SetFileName(m_OutName.c_str());
      writer->SetUseCompression(0);

      writer->Update();
  }
  catch (itk::ExceptionObject exp){
    std::cout << "  Couldn't save image: " << m_OutName << std::endl;
    std::cout << "    Excepcion caught: " << exp << std::endl;
  }
}

template< class TInputImage, class TOutputImage >
void
Mask2ColorBorderImageFilter< TInputImage, TOutputImage >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  
  os << indent << std::endl;
  
}

} // end namespace

#endif
