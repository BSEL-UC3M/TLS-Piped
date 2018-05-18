#ifndef __itkMask2ColorBorderImageFilter_h
#define __itkMask2ColorBorderImageFilter_h

#include "itkImageFunction.h"
#include "itkImageRegionIterator.h"
#include "itkImageToImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodIterator.h"
#include "itkSize.h"
#include "itkVector.h"
#include <limits.h>

#include <itkBinaryErodeImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkSubtractImageFilter.h>

namespace itk
{

  template<class TInputImage, class TOutputImage>
  class ITK_EXPORT Mask2ColorBorderImageFilter :
      public ImageToImageFilter<TInputImage, TOutputImage>
  {
  public:
    /** Standard class typedefs. */
    typedef Mask2ColorBorderImageFilter Self;
    typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
    typedef SmartPointer<Self>        Pointer;
    typedef SmartPointer<const Self>  ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(Mask2ColorBorderImageFilter, ImageToImageFilter);

    /** Number of dimensions */
    itkStaticConstMacro(NDimensions, unsigned int, TInputImage::ImageDimension);
    itkStaticConstMacro(NOutputDimensions, unsigned int, TOutputImage::ImageDimension);

    /** typedef for images */
    typedef unsigned char                           RGBType;
    typedef typename itk::RGBPixel<RGBType>         RGBPixelType;

    typedef TInputImage                             InputImageType;
    typedef TOutputImage                            OutputImageType;
    typedef typename itk::Image<bool, NDimensions>  BinImageType;
    typedef typename itk::Image<RGBPixelType,
        NDimensions>                                ITKRGBImageType;
    typedef typename OutputImageType::Pointer       OutputImagePointer;
    typedef typename InputImageType::Pointer        InputImagePointer;
    typedef typename InputImageType::ConstPointer   InputImageConstPointer;
    typedef typename BinImageType::Pointer          BinImagePointer;

    typedef typename TInputImage::RegionType        RegionType;
    typedef typename std::string                    string;

    /** Image size typedef */
    typedef Size<itkGetStaticConstMacro(NDimensions)> SizeType;

    /** Image index typedef */
    typedef typename TOutputImage::IndexType IndexType;

    /** Image pixel value typedef */
    typedef typename TInputImage::PixelType   inPixelType;
    typedef typename TOutputImage::PixelType  outPixelType;

    /** Typedef to describe the output image region type. */
    typedef typename TOutputImage::RegionType OutputImageRegionType;

    itkSetMacro(BorderSize, unsigned char);
    itkGetMacro(BorderSize, unsigned char);

    itkSetMacro(OutName, string);
    itkGetMacro(OutName, string);

    itkSetMacro(MaskImage, BinImagePointer);
    itkGetMacro(MaskImage, BinImagePointer);

    itkSetMacro(RedMask, unsigned char);
    itkGetMacro(RedMask, unsigned char);
    itkSetMacro(GreenMask, unsigned char);
    itkGetMacro(GreenMask, unsigned char);
    itkSetMacro(BlueMask, unsigned char);
    itkGetMacro(BlueMask, unsigned char);

    /** This filter needs to request a larger input than its requested output.
     * If this filter runs "Repetitions" iterations, then it needs an input
     * that is 2*Repetitions larger than the output. In other words, this
     * filter needs a border of "Repetitions" pixels. */
    void GenerateInputRequestedRegion();

  #ifdef ITK_USE_CONCEPT_CHECKING
    /** Begin concept checking */
    itkConceptMacro(SameDimensionCheck,
      (Concept::SameDimension<itkGetStaticConstMacro(NDimensions),
                              itkGetStaticConstMacro(NOutputDimensions)>));
    itkConceptMacro(InputConvertibleToDoubleCheck,
      (Concept::Convertible<inPixelType, double>));
    itkConceptMacro(DoubleConvertibleToOutputCheck,
      (Concept::Convertible<double, inPixelType>));
    /** End concept checking */
  #endif

  protected:
    Mask2ColorBorderImageFilter();
    virtual ~Mask2ColorBorderImageFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** Method for evaluating the implicit function over the image. */
    void GenerateData();

  private:
    Mask2ColorBorderImageFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    unsigned char m_BorderSize;

    string m_OutName;

    BinImagePointer m_MaskImage;

    unsigned char m_RedMask, m_GreenMask, m_BlueMask;
  };

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMask2ColorBorderImageFilter.txx"
#endif

#endif
