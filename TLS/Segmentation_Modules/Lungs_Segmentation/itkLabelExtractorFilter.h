#ifndef __itkLabelExtractorFilter_h
#define __itkLabelExtractorFilter_h

#include <string>

#include "itkImageFunction.h"

#include "itkImageToImageFilter.h"
#include "itkSize.h"
#include "itkRegionFilter.h"


namespace itk
{

template<class TInputImage, class TOutputImage>
class ITK_EXPORT LabelExtractorFilter :
    public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef LabelExtractorFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( LabelExtractorFilter, ImageToImageFilter );

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
  
  itkSetMacro(BaseDir,std::string);
  itkSetMacro(Inc, float);
  itkGetMacro(Inc, float);
  itkSetMacro(Index, IndexType);
  itkGetMacro(Index, IndexType);
  itkSetMacro(Size, SizeType);
  itkGetMacro(Size, SizeType);
   
  
  
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
    (Concept::Convertible<typename TInputImage::PixelType, double>));
  itkConceptMacro(DoubleConvertibleToOutputCheck,
    (Concept::Convertible<double, PixelType>));
  /** End concept checking */
#endif

protected:
  LabelExtractorFilter();
  virtual ~LabelExtractorFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method for evaluating the implicit function over the image. */
  void GenerateData();

private:
  LabelExtractorFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  float m_Inc;
  IndexType m_Index;
  SizeType m_Size;
  std::string m_BaseDir;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLabelExtractorFilter.txx"
#endif

#endif
