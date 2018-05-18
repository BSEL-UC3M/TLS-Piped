#ifndef __itkConnectivityImageFilter_h
#define __itkConnectivityImageFilter_h

#include "itkImageFunction.h"
#include "itkImageRegionIterator.h"
#include "itkImageToImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodIterator.h"
#include "itkSize.h"
#include "itkVector.h"
#include "itkShapedNeighborhoodIterator.h"
#include <vector>

namespace itk
{

template<class TInputImage, class TOutputImage>
class ITK_EXPORT ConnectivityImageFilter :
    public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ConnectivityImageFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( ConnectivityImageFilter, ImageToImageFilter );

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
  typedef itk::ShapedNeighborhoodIterator<TInputImage> NeighborhoodIterator;
  typedef typename NeighborhoodIterator::OffsetType OType;

  std::vector<OType> offsets;


  
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
  ConnectivityImageFilter();
  virtual ~ConnectivityImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method for evaluating the implicit function over the image. */
  void GenerateData();

private:
  ConnectivityImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkConnectivityImageFilter.txx"
#endif
#endif
