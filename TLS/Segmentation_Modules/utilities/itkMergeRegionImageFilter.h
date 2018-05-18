#ifndef __itkMergeRegionImageFilter_h
#define __itkMergeRegionImageFilter_h


#include "itkImageToImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"

namespace itk
{

template<class TInputImage, class TOutputImage>
class ITK_EXPORT MergeRegionImageFilter :
    public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef MergeRegionImageFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( MergeRegionImageFilter, ImageToImageFilter );

  /** Number of dimensions */
  itkStaticConstMacro(NDimensions, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(NOutputDimensions, unsigned int, TOutputImage::ImageDimension);

  /** typedef for images */
  typedef TInputImage                             InputImageType;
  typedef TOutputImage                            OutputImageType;
  typedef typename OutputImageType::Pointer       OutputImagePointer;
  typedef typename InputImageType::Pointer        InputImagePointer;
  typedef typename InputImageType::ConstPointer   InputImageConstPointer;
  typedef typename InputImageType::RegionType         RegionType;

  /** Image size typedef */
  typedef Size<itkGetStaticConstMacro(NDimensions)> SizeType;

  /** Image index typedef */
  typedef typename TOutputImage::IndexType IndexType;

  /** Image pixel value typedef */
  typedef typename TOutputImage::PixelType PixelType;

  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;
  
  
  /*void SetMergeRegion(RegionType region) {
    m_MergeRegion = region;
  }
  
  RegionType GetMergeRegion() {
    return m_MergeRegion;
  }*/
    
  itkSetMacro(MergeRegion, RegionType);
  itkGetMacro(MergeRegion, RegionType);
  itkSetMacro(MergeImage, InputImagePointer);
  itkGetMacro(MergeImage, InputImagePointer);
  itkSetMacro(ProbThreshold, float);
  itkGetMacro(ProbThreshold, float);

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
  MergeRegionImageFilter();
  virtual ~MergeRegionImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method for evaluating the implicit function over the image. */
  void GenerateData();

private:
  MergeRegionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
  InputImagePointer m_MergeImage;
  RegionType m_MergeRegion;
  float m_ProbThreshold;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMergeRegionImageFilter.txx"
#endif

#endif
