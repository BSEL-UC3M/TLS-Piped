#ifndef __itkLabelPipelineImageFilter_h
#define __itkLabelPipelineImageFilter_h

#include "itkImageFunction.h"
#include "itkImageRegionIterator.h"
#include "itkImageToImageFilter.h"
#include "itkSize.h"

#include "itkSubtractImageFilter.h"

#include "itkShapeLabelObject.h"
#include "itkLabelMap.h"


#include "itkLabelMapToLabelImageFilter.h"

#include "itkExtractImageFilter.h"
#include "itkInvertIntensityImageFilter.h"
#include "itkLabelExtractorFilter.h"
#include "itkShapeLabelObject.h"
#include "itkExtractImageFilter.h"

#include "itkRegionFilter.h"

//itk
#include "itkGrayscaleFillholeImageFilter.h"
#include "itkImageFileWriter.h"
#include <itkMedianImageFilter.h>
#include "itkImage.h"

#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryImageToLabelMapFilter.h"

#include <stdio.h>
#include <string>
#include <time.h>
#include <list>
#include <iostream>
#include <fstream>
#include <vector>

//#include <time.h>
#include <sys/types.h>
#include <sys/timeb.h>

namespace itk
{

template<class TInputImage, class TOutputImage>
class ITK_EXPORT LabelPipelineImageFilter :
    public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef LabelPipelineImageFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( LabelPipelineImageFilter, ImageToImageFilter );

  /** Number of dimensions */
  itkStaticConstMacro(NDimensions, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(NOutputDimensions, unsigned int, TOutputImage::ImageDimension);

  /** typedef for images */
  typedef unsigned long 																							  LabelType;
  typedef TInputImage                                                   InputImageType;
  typedef TOutputImage                                                  OutputImageType;
  typedef signed short 																								  PixelType;
  typedef itk::Image< PixelType, 3 >                     		 				  ImageType;
  typedef itk::Image< PixelType, 2>                                	  SliceType;
  typedef typename OutputImageType::Pointer                             OutputImagePointer;
  typedef typename InputImageType::Pointer                              InputImagePointer;
  typedef typename InputImageType::ConstPointer                         InputImageConstPointer;
  typedef typename InputImageType::IndexType 													  IndexType;
  typedef typename InputImageType::SizeType 													  SizeType;
  typedef typename InputImageType::RegionType 													RegionType;
  

  typedef itk::ShapeLabelObject< LabelType, 3 >            		 				  LabelObjectType;
  typedef itk::LabelMap< LabelObjectType >                         		  LabelCollectionType;  
	typedef itk::MedianImageFilter< ImageType, ImageType >           		  MedianImageFilterType;
	typedef itk::ImageFileWriter< ImageType >                         	  WriterType;
	typedef itk::ImageFileWriter< LabelCollectionType >                   WriterLabelType;

	
	typedef itk::RegionFilter<TOutputImage, TOutputImage>    									RegionFilter;
	
	typedef itk::ImageRegionIterator< ImageType >                    			IteratorType;
	typedef itk::ImageRegionConstIterator< ImageType >                    ConstIteratorType;

  typedef typename TOutputImage::RegionType 														OutputImageRegionType;
  

  itkSetMacro(medianRadius, SizeType);
  itkGetMacro(medianRadius, SizeType);
  
  itkSetMacro(Backup, bool);
  itkGetMacro(Backup, bool);
  
  itkSetMacro(thresholdActual, double);
  itkGetMacro(thresholdActual, double);
  
  itkSetMacro(tThreshold, double);
  itkGetMacro(tThreshold, double);
  itkSetMacro(tMedian, double);
  itkGetMacro(tMedian, double);
  itkSetMacro(tLabeling, double);
  itkGetMacro(tLabeling, double);
  itkSetMacro(tSelecting, double);
  itkGetMacro(tSelecting, double);
  
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
  LabelPipelineImageFilter();
  virtual ~LabelPipelineImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method for evaluating the implicit function over the image. */
  void GenerateData();

private:
  LabelPipelineImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool      m_Backup;
  SizeType  m_medianRadius;
 
  double    m_thresholdActual;
  double    m_tThreshold;
  double    m_tMedian;
  double    m_tLabeling;
  double    m_tSelecting;
  

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLabelPipelineImageFilter.txx"
#endif

#endif
