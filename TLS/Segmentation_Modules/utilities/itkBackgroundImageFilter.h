#ifndef __itkBackgroundImageFilter_h
#define __itkBackgroundImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"

namespace itk
{

template<class TInputImage, class TOutputImage>
class ITK_EXPORT BackgroundImageFilter :
    public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef BackgroundImageFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( BackgroundImageFilter, ImageToImageFilter );

  /** Number of dimensions */
  itkStaticConstMacro(NDimensions, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(NOutputDimensions, unsigned int, TOutputImage::ImageDimension);

  /** typedef for images */
  typedef TInputImage                             InputImageType;
  typedef TOutputImage                            OutputImageType;
  typedef typename itk::Image<signed short, NDimensions>
                                                  ShortImageType;

  typedef typename OutputImageType::Pointer       OutputImagePointer;
  typedef typename InputImageType::Pointer        InputImagePointer;
  typedef typename InputImageType::ConstPointer   InputImageConstPointer;
  typedef typename ShortImageType::Pointer        ShortImagePointer;

  /** typedef for iterators */
  typedef ImageSliceConstIteratorWithIndex
      <TInputImage>                               SliceConstIteratorType;
  typedef ImageSliceConstIteratorWithIndex
      <ShortImageType>                            ShortSliceConstIteratorType;
  typedef ImageSliceIteratorWithIndex
      <TOutputImage>                              SliceIteratorType;

  /** Image size typedef */
  typedef Size<itkGetStaticConstMacro(NDimensions)> SizeType;

  /** Image index typedef */
  typedef typename TOutputImage::IndexType IndexType;

  /** Image pixel value typedef */
  typedef typename TOutputImage::PixelType PixelType;

  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;

  itkSetMacro(GrayScaleImage, ShortImagePointer);

  /** This filter needs to request a larger input than its requested output.
   * If this filter runs "Repetitions" iterations, then it needs an input
   * that is 2*Repetitions larger than the output. In other words, this
   * filter needs a border of "Repetitions" pixels. */
  void GenerateInputRequestedRegion();

#ifdef ITK_USE_CONCEPT_CHECKING
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<itkGetStaticConstMacro(NDimensions),
                            itkGetStaticConstMacro(NOutputDimensions)>));
  itkConceptMacro(InputConvertibleToDoubleCheck,
    (Concept::Convertible<typename TInputImage::PixelType, double>));
  itkConceptMacro(DoubleConvertibleToOutputCheck,
    (Concept::Convertible<double, PixelType>));
#endif

protected:
  BackgroundImageFilter();
  virtual ~BackgroundImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  void Background1Direction(ShortImagePointer, InputImageConstPointer,
                            OutputImagePointer, unsigned short, unsigned short);

  /** Method for evaluating the implicit function over the image. */
  void GenerateData();

private:
  BackgroundImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  ShortImagePointer m_GrayScaleImage;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBackgroundImageFilter.txx"
#endif

#endif
