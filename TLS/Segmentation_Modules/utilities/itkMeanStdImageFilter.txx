#ifndef __itkMeanStdImageFilter_txx
#define __itkMeanStdImageFilter_txx

#include "itkMeanStdImageFilter.h"

namespace itk
{

  template< class TInputImage, class TOutputImage >
  MeanStdImageFilter< TInputImage, TOutputImage >
  ::MeanStdImageFilter()
  {
    itkDebugMacro(<< "MeanStdImageFilter::MeanStdImageFilter() called");
  }

  template< class TInputImage, class TOutputImage >
  void
  MeanStdImageFilter< TInputImage, TOutputImage >
  ::GenerateInputRequestedRegion()
  {

  }


  template<class TInputImage, class TOutputImage>
  void MeanStdImageFilter<TInputImage, TOutputImage>
    ::MaskImage(SLImagePointer inImage, BinImagePointer maskImage)
  {
    BinConstIteratorType mask = BinConstIteratorType(maskImage, maskImage->GetLargestPossibleRegion());
    SLIteratorType out = SLIteratorType(inImage, inImage->GetLargestPossibleRegion());

    for (mask.GoToBegin(), out.GoToBegin();!mask.IsAtEnd(), !out.IsAtEnd();++mask, ++out)
      if (!mask.Get())
        out.Set(0);
  }


  template<class TInputImage, class TOutputImage>
  void MeanStdImageFilter<TInputImage, TOutputImage>
    ::MaskImage(USImagePointer inImage, BinImagePointer maskImage)
  {
    BinConstIteratorType mask = BinConstIteratorType(maskImage, maskImage->GetLargestPossibleRegion());
    USIteratorType out = USIteratorType(inImage, maskImage->GetLargestPossibleRegion());

    for (mask.GoToBegin(), out.GoToBegin();!mask.IsAtEnd(), !out.IsAtEnd();++mask, ++out)
      if (!mask.Get())
        out.Set(0);
  }


  template<class TInputImage, class TOutputImage>
  void MeanStdImageFilter<TInputImage, TOutputImage>
    ::CalculateNewMask(BinImagePointer inImage, BinImagePointer outImage, RegionType outRegion)
  {
    RegionType movingRegion = inImage->GetLargestPossibleRegion();
    movingRegion.SetSize(m_RegionSize);

    BinConstIteratorType bin = BinConstIteratorType(inImage, outRegion);
    BinIteratorType out = BinIteratorType(outImage, outRegion);
    for (bin.GoToBegin(),out.GoToBegin();!bin.IsAtEnd(),!out.IsAtEnd();++bin,++out) {
      IndexType outIndex = bin.GetIndex();

      movingRegion.SetIndex(outIndex);
      BinConstIteratorType mask = BinConstIteratorType(inImage, movingRegion);

      mask.GoToBegin();
      while(!mask.IsAtEnd()&&mask.Get())
        ++mask;

      out.Set(mask.IsAtEnd());
    }
  }

  template<class TInputImage, class TOutputImage>
  void MeanStdImageFilter<TInputImage, TOutputImage>
  ::ProcessSecondLine(InputImageConstPointer inImage, SLSliceIteratorType outMean,
                  USSliceIteratorType outStd, unsigned short movingArea)
  {
    RegionType inRegion = inImage->GetLargestPossibleRegion();

    IndexType movingRegionIndex = outMean.GetIndex();
    SizeType movingRegionSize = m_RegionSize;
    movingRegionSize[1] = 1;

    RegionType movingRegion = inRegion;
    movingRegion.SetIndex(movingRegionIndex);
    movingRegion.SetSize(movingRegionSize);

    outMean.GoToBegin();
    outStd.GoToBegin();
    while (!outMean.IsAtEndOfSlice()&&!outStd.IsAtEndOfSlice()) {
      outPixelType meanArea = outMean.Get()*movingArea;
      float stdArea  = (pow(outStd.Get(),2) + pow(outMean.Get(), 2))*movingArea;
      ++outMean;
      ++outStd;

      while (!outMean.IsAtEndOfLine()&&!outStd.IsAtEndOfLine()) {
        movingRegion.SetIndex(movingRegionIndex);
        ConstIteratorType lineMinusMR = ConstIteratorType(inImage, movingRegion);
        for (lineMinusMR.GoToBegin();!lineMinusMR.IsAtEnd();++lineMinusMR) {
          meanArea -= lineMinusMR.Get();
          stdArea  -= pow(lineMinusMR.Get(),2);
        }

        movingRegionIndex[1] += m_RegionSize[1];
        movingRegion.SetIndex(movingRegionIndex);
        ConstIteratorType lineAddMR = ConstIteratorType(inImage, movingRegion);
        for (lineAddMR.GoToBegin();!lineAddMR.IsAtEnd();++lineAddMR) {
          meanArea += lineAddMR.Get();
          stdArea  += pow(lineAddMR.Get(),2);
        }
        outPixelType mean = meanArea/movingArea;
        outMean.Set(mean);
        float dev = sqrt(stdArea/movingArea - pow(mean, 2));
        outStd.Set(dev);

	    if (dev <= 10.0) {
	  	  std::cout << dev << std::endl;
	  	  std::cout << "Index: " << outStd.GetIndex() << std::endl;
  	    }

        movingRegionIndex[1] -= m_RegionSize[1]-1;
        ++outMean;
        ++outStd;
      }
      movingRegionIndex[1] = 0;
      movingRegionIndex[0] = (movingRegionIndex[0]+1)%503;

      outMean.NextLine();
      outStd.NextLine();
    }
    outMean.NextSlice();
    outStd.NextSlice();
  }


  template<class TInputImage, class TOutputImage>
  void MeanStdImageFilter<TInputImage, TOutputImage>
  ::ProcessFirstLine(InputImageConstPointer inImage, SLLinearIteratorType outMean,
                  USLinearIteratorType outStd, unsigned short movingArea)
  {
    RegionType inRegion = inImage->GetLargestPossibleRegion();

    outMean.GoToBegin();
    outStd.GoToBegin();

    RegionType movingRegion = inRegion;

    SizeType movingRegionSize = m_RegionSize;
    movingRegion.SetSize(movingRegionSize);

    IndexType movingRegionIndex = outMean.GetIndex();
    movingRegion.SetIndex(movingRegionIndex);

    ConstIteratorType inMR = ConstIteratorType(inImage, movingRegion);

    outPixelType meanArea = 0;
    float stdArea = 0;
/*
   To calculate std use the formula $\sqrt{\frac{1}{N}\sum x_{i}^{2}-\mu^{2}}$
*/
    for (inMR.GoToBegin(); !inMR.IsAtEnd(); ++inMR) {
      meanArea += inMR.Get();
      stdArea  += pow(inMR.Get(), 2);
    }

    outPixelType mean = meanArea/movingArea;
    outMean.Set(mean);
    float dev = sqrt(stdArea/movingArea - pow(mean, 2));

    outStd.Set(dev);
    ++outMean;
    ++outStd;

    movingRegionSize[0] = 1;
    movingRegion.SetSize(movingRegionSize);
    while (!outMean.IsAtEndOfLine()&&!outStd.IsAtEndOfLine()) {
      movingRegion.SetIndex(movingRegionIndex);

      ConstIteratorType lineMinusMR = ConstIteratorType(inImage, movingRegion);
      for (lineMinusMR.GoToBegin();!lineMinusMR.IsAtEnd();++lineMinusMR) {
        meanArea -= lineMinusMR.Get();
        stdArea  -= pow(lineMinusMR.Get(),2);
      }

      movingRegionIndex[0] += m_RegionSize[0];
      movingRegion.SetIndex(movingRegionIndex);
      ConstIteratorType lineAddMR = ConstIteratorType(inImage, movingRegion);
      for (lineAddMR.GoToBegin();!lineAddMR.IsAtEnd();++lineAddMR) {
        meanArea += lineAddMR.Get();
        stdArea  += pow(lineAddMR.Get(),2);
      }
      mean = meanArea/movingArea;
      outMean.Set(mean);
      dev = sqrt(stdArea/movingArea - pow(mean, 2));
      outStd.Set(dev);
	  if (dev <= 10.0) {
	  	std::cout << dev << std::endl;
	  	std::cout << "Index: " << outStd.GetIndex() << std::endl;
	  }

      ++outMean;
      ++outStd;
      movingRegionIndex[0] -= m_RegionSize[0]-1;
    }
  }


  template< class TInputImage, class TOutputImage >
  void
  MeanStdImageFilter< TInputImage, TOutputImage >
  ::GenerateData()
  {
    itkDebugMacro(<< "MeanStdImageFilter::GenerateData() called");

    // Get the input and output pointers
    InputImageConstPointer  inImage  = this->GetInput(0);
    m_MeanImage = this->GetOutput(0);

    RegionType inRegion  = inImage->GetLargestPossibleRegion();

    // Allocate the output
    m_MeanImage->SetBufferedRegion(inRegion);
    m_MeanImage->Allocate();

    m_StdImage = USImageType::New();
    m_StdImage->SetBufferedRegion(inRegion);
    m_StdImage->Allocate();

    SizeType outSliceSize = inRegion.GetSize();
    unsigned short movingArea = 1;
    for (unsigned int dim = 0; dim< NDimensions;dim++) {
      movingArea    *= m_RegionSize[dim];
      outSliceSize[dim]  -= m_RegionSize[dim];
    }
    outSliceSize[2]++;
    IndexType outIndex = inRegion.GetIndex();
    unsigned short slides = outSliceSize[2];
    outSliceSize[2] = 1;

    RegionType outRegion = inRegion;
    outRegion.SetSize(outSliceSize);
    for (unsigned short z=0; z<slides;z++) {
      outIndex[2] = z;

      //std::cout << "  Processing first line of slide " << z+1 << "/" << slides << std::endl;

      outRegion.SetIndex(outIndex);
      SLLinearIteratorType out1Mean(m_MeanImage, outRegion);
      USLinearIteratorType out1Std(m_StdImage, outRegion);
      ProcessFirstLine(inImage, out1Mean, out1Std, movingArea);

      SLSliceIteratorType out2Mean(m_MeanImage, outRegion);
      out2Mean.SetFirstDirection(1);
      out2Mean.SetSecondDirection(0);
      USSliceIteratorType out2Std(m_StdImage, outRegion);
      out2Std.SetFirstDirection(1);
      out2Std.SetSecondDirection(0);
      ProcessSecondLine(inImage, out2Mean, out2Std, movingArea);
    }
    RegionType maskRegion = inRegion;
    SizeType maskSize = inRegion.GetSize();
    maskSize[0] = outSliceSize[0];
    maskSize[1] = outSliceSize[1];
    maskRegion.SetSize(maskSize);

    BinImagePointer outMaskImage = BinImageType::New();
    outMaskImage->SetRegions(inRegion);
    outMaskImage->Allocate();

    CalculateNewMask(m_MaskImage, outMaskImage, maskRegion);

    typename BorderFilterType::Pointer backgroundBorderFilter = BorderFilterType::New();
    backgroundBorderFilter->SetInput(inImage);
    backgroundBorderFilter->SetMaskImage(outMaskImage);
    backgroundBorderFilter->SetGreenMask(255);
    backgroundBorderFilter->SetOutName("NewMaskImage.tif");

    backgroundBorderFilter->Update();

    MaskImage(m_MeanImage, outMaskImage);
    MaskImage(m_StdImage, outMaskImage);
  }

  template< class TInputImage, class TOutputImage >
  void
  MeanStdImageFilter< TInputImage, TOutputImage >
  ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf(os,indent);

    os << indent << std::endl;
  }

} // end namespace

#endif
