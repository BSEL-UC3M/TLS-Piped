#ifndef __itkMaskRegionImageFilter_h
#define __itkMaskRegionImageFilter_h

#include "itkImage.h"
#include "itkImageToImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

#include <stdio.h>
#include <iostream>

namespace itk
{

template <class TInputImage, class TOutputImage>
class ITK_EXPORT MaskRegionImageFilter: public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef MaskRegionImageFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( MaskRegionImageFilter, ImageToImageFilter );

  /** Number of dimensions */
  itkStaticConstMacro(NDimensions, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(NOutputDimensions, unsigned int, TOutputImage::ImageDimension);

  /** typedef for images */
  
  typedef TInputImage                                                   InputImageType;
  typedef TOutputImage                                                  OutputImageType;
  typedef signed short 																								  PixelType;
  typedef itk::Image< PixelType, 3 >                       		 				  ImageType;
  typedef typename OutputImageType::Pointer                             OutputImagePointer;
  typedef typename InputImageType::Pointer                              InputImagePointer;
  typedef typename InputImageType::ConstPointer                         InputImageConstPointer;
  typedef typename InputImageType::RegionType														RegionType;
  typedef typename InputImageType::IndexType 													  IndexType;
  typedef typename InputImageType::SizeType 													  SizeType;
  
  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;
  
	typedef itk::ImageRegionIterator<OutputImageType>                    				IteratorType;
	typedef itk::ImageRegionConstIterator<InputImageType>                    	ConstIteratorType;

  
  //Not aleady implemented but in mind
  itkSetMacro(RegionOfInterest, RegionType);
  itkGetMacro(RegionOfInterest, RegionType);
  
protected:
  MaskRegionImageFilter();
  virtual ~MaskRegionImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method for evaluating the implicit function over the image. */
  void GenerateData();

private:
  MaskRegionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
  RegionType  m_RegionOfInterest;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMaskRegionImageFilter.txx"
#endif

#endif
