#ifndef __itkBackgroundImageFilter_txx
#define __itkBackgroundImageFilter_txx

#include "itkBackgroundImageFilter.h"

namespace itk
{

  template< class TInputImage, class TOutputImage >
  BackgroundImageFilter< TInputImage, TOutputImage >
  ::BackgroundImageFilter()
  {
    itkDebugMacro(<< "BackgroundImageFilter::BackgroundImageFilter() called");
  }

  template< class TInputImage, class TOutputImage >
  void
  BackgroundImageFilter< TInputImage, TOutputImage >
  ::GenerateInputRequestedRegion()
  {

  }


  template <class TInputImage, class TOutputImage>
  void BackgroundImageFilter<TInputImage, TOutputImage>
  ::Background1Direction(ShortImagePointer inImage, InputImageConstPointer maskImage,
      OutputImagePointer outImage, unsigned short direction1, unsigned short direction2) {
    OutputImageRegionType inRegion=maskImage->GetLargestPossibleRegion();

    ShortSliceConstIteratorType in(inImage, inRegion);
    SliceConstIteratorType mask(maskImage, inRegion);
    SliceIteratorType out(outImage, inRegion);

    in.SetFirstDirection(direction1); in.SetSecondDirection(direction2);
    in.GoToBegin();

    mask.SetFirstDirection(direction1); mask.SetSecondDirection(direction2);
    mask.GoToBegin();

    out.SetFirstDirection(direction1); out.SetSecondDirection(direction2);
    out.GoToBegin();
    while (!mask.IsAtEnd()) {
      while (!mask.IsAtEndOfSlice()) {
        while (!mask.IsAtEndOfLine()&&mask.Get()) {
          out.Set(in.Get()!=-1400);
          ++in;++mask;++out;
        }
        while (!mask.IsAtEndOfLine()) {
          ++in;++mask;++out;
        }
        in.NextLine();
        out.NextLine();
        mask.NextLine();
      }
      in.NextSlice();
      mask.NextSlice();
      out.NextSlice();
    }

    in.GoToReverseBegin();
    out.GoToReverseBegin();
    mask.GoToReverseBegin();
    while (!out.IsAtReverseEnd()) {
      while (!out.IsAtReverseEndOfSlice()) {
        while (!out.IsAtReverseEndOfLine()&&mask.Get()) {
          out.Set(in.Get()!=-1400);
          --in;--mask;--out;
        }
        while (!mask.IsAtReverseEndOfLine()) {
          --in;--mask;--out;
        }
        in.PreviousLine();
        out.PreviousLine();
        mask.PreviousLine();
      }
      in.PreviousSlice();
      mask.PreviousSlice();
      out.PreviousSlice();
    }
  }


  template< class TInputImage, class TOutputImage >
  void
  BackgroundImageFilter< TInputImage, TOutputImage >
  ::GenerateData()
  {
    itkDebugMacro(<< "BackgroundImageFilter::GenerateData() called");

    // Get the input and output pointers
    InputImageConstPointer  inputPtr  = this->GetInput(0);
    OutputImagePointer      outputPtr = this->GetOutput(0);

    OutputImageRegionType inRegion = inputPtr->GetLargestPossibleRegion();

    // Allocate the output
    outputPtr->SetRegions(inRegion);
    outputPtr->Allocate();
    outputPtr->FillBuffer(0);

    for (unsigned char i=0; i<2; i++)
      Background1Direction(m_GrayScaleImage, inputPtr, outputPtr, i, 1-i);
  }

  template< class TInputImage, class TOutputImage >
  void
  BackgroundImageFilter< TInputImage, TOutputImage >
  ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf(os,indent);

    os << indent << std::endl;
  }

} // end namespace

#endif
