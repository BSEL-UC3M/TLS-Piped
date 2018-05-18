#ifndef __itkHuThresholdImageFilter_h
#define __itkHuThresholdImageFilter_h

#include "itkImageFunction.h"
#include "itkImageRegionIterator.h"
#include "itkImageToImageFilter.h"
#include "itkSize.h"


namespace itk
{

template<class TInputImage, class TOutputImage>
class ITK_EXPORT HuThresholdImageFilter :
    public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef HuThresholdImageFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( HuThresholdImageFilter, ImageToImageFilter );

  /** Number of dimensions */
  itkStaticConstMacro(NDimensions, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(NOutputDimensions, unsigned int, TOutputImage::ImageDimension);

  /** typedef for images */
  typedef TInputImage                             InputImageType;
  typedef TOutputImage                            OutputImageType;
  typedef typename OutputImageType::Pointer       OutputImagePointer;
  typedef typename InputImageType::Pointer        InputImagePointer;
  typedef typename InputImageType::ConstPointer   InputImageConstPointer;

  /** Image size typedef */
  typedef Size<itkGetStaticConstMacro(NDimensions)> SizeType;

  /** Image index typedef */
  typedef typename TOutputImage::IndexType IndexType;

  /** Image pixel value typedef */
  typedef typename TOutputImage::PixelType PixelType;

  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;
  
  itkSetMacro(Diff, double);
  itkGetMacro(Diff, double);
  itkSetMacro(Ti, double);
  itkGetMacro(Ti, double);
  itkSetMacro(Index, IndexType);
  itkGetMacro(Index, IndexType);
  itkSetMacro(Size, SizeType);
  itkGetMacro(Size, SizeType);
  itkSetMacro(Threshold, double);
  itkGetMacro(Threshold, double);
  itkSetMacro(PlusTh, double);
  itkGetMacro(PlusTh, double);

  void GenerateInputRequestedRegion();

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<itkGetStaticConstMacro(NDimensions),
                            itkGetStaticConstMacro(NOutputDimensions)>));
  itkConceptMacro(InputConvertibleToDoubleCheck,
    (Concept::Convertible<typename TInputImage::PixelType, double>));
  itkConceptMacro(DoubleConvertibleToOutputCheck,
    (Concept::Convertible<double, PixelType>));
  /** End concept checking */
#endif

protected:
  HuThresholdImageFilter();
  virtual ~HuThresholdImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method for evaluating the implicit function over the image. */
  void GenerateData();

private:
  HuThresholdImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  double m_Diff;
  IndexType m_Index;
  SizeType m_Size;
  double m_Ti;
  double m_Threshold;
  double m_PlusTh;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHuThresholdImageFilter.txx"
#endif

#endif
