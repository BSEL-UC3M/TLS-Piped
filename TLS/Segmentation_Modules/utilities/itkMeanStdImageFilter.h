#ifndef __itkMeanStdImageFilter_h
#define __itkMeanStdImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkImageLinearIteratorWithIndex.h"

#include "itkMask2ColorBorderImageFilter.h"

namespace itk
{

template<class TInputImage, class TOutputImage>
class ITK_EXPORT MeanStdImageFilter :
    public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef MeanStdImageFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( MeanStdImageFilter, ImageToImageFilter );

  /** Number of dimensions */
  itkStaticConstMacro(NDimensions, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(NOutputDimensions, unsigned int, TOutputImage::ImageDimension);

  /** typedef for pixel */
  typedef typename TInputImage::PixelType                   inPixelType;
  typedef typename TOutputImage::PixelType                  outPixelType;
  typedef unsigned char                                     RGBType;
  typedef itk::RGBPixel<RGBType>                            RGBPixelType;

  /** typedef for images */
  typedef TInputImage                                       InputImageType;
  typedef TOutputImage                                      OutputImageType;
  typedef typename itk::Image<unsigned short, NDimensions>  USImageType;
  typedef typename itk::Image<float, NDimensions>  			FImageType;
  typedef typename itk::Image<long, NDimensions>            SLImageType;
  typedef typename itk::Image<bool, NDimensions>            BinImageType;
  typedef itk::Image<RGBPixelType, NDimensions>             RGBImageType;

  /** typedef for pointers */
  typedef typename OutputImageType::Pointer                 OutputImagePointer;
  typedef typename InputImageType::Pointer                  InputImagePointer;
  typedef typename InputImageType::ConstPointer             InputImageConstPointer;
  typedef typename USImageType::Pointer                     USImagePointer;
  typedef typename SLImageType::Pointer                     SLImagePointer;
  typedef typename BinImageType::Pointer                    BinImagePointer;

  /** typedef for iterators */
  typedef ImageSliceConstIteratorWithIndex<TInputImage>     SliceConstIteratorType;
  typedef ImageSliceIteratorWithIndex<TOutputImage>         SLSliceIteratorType;
  typedef ImageSliceIteratorWithIndex<USImageType>          USSliceIteratorType;
  typedef ImageRegionConstIterator<TInputImage>             ConstIteratorType;
  typedef ImageLinearIteratorWithIndex <TOutputImage>       SLLinearIteratorType;
  typedef ImageLinearIteratorWithIndex <USImageType>        USLinearIteratorType;
  typedef ImageRegionConstIterator<BinImageType>            BinConstIteratorType;
  typedef ImageRegionIterator<BinImageType>                 BinIteratorType;
  typedef ImageRegionIterator<USImageType>                  USIteratorType;
  typedef ImageRegionIterator<SLImageType>                  SLIteratorType;

  /** typedef filter types */
  typedef Mask2ColorBorderImageFilter<TInputImage,
      RGBImageType>                                         BorderFilterType;

  /** Image size typedef */
  typedef typename TInputImage::SizeType          SizeType;

  /** Image index typedef */
  typedef typename TOutputImage::IndexType        IndexType;

  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType       RegionType;

  itkSetMacro(RegionSize, SizeType);
  itkGetMacro(MeanImage, OutputImagePointer);
  itkGetMacro(StdImage,  USImagePointer);
  itkSetMacro(MaskImage, BinImagePointer);

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
    (Concept::Convertible<inPixelType, double>));
  itkConceptMacro(DoubleConvertibleToOutputCheck,
    (Concept::Convertible<double, outPixelType>));
#endif

protected:
  MeanStdImageFilter();
  virtual ~MeanStdImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  void ProcessFirstLine(InputImageConstPointer, SLLinearIteratorType,
                     USLinearIteratorType, unsigned short);
  void ProcessSecondLine(InputImageConstPointer, SLSliceIteratorType,
                     USSliceIteratorType, unsigned short);
  void CalculateNewMask(BinImagePointer, BinImagePointer, RegionType);
  void MaskImage(USImagePointer, BinImagePointer);
  void MaskImage(SLImagePointer, BinImagePointer);

  /** Method for evaluating the implicit function over the image. */
  void GenerateData();

private:
  MeanStdImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  SizeType m_RegionSize;

  OutputImagePointer m_MeanImage;
  USImagePointer     m_StdImage;
  BinImagePointer    m_MaskImage;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMeanStdImageFilter.txx"
#endif

#endif
