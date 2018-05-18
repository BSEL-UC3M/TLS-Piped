/*=========================================================================


=========================================================================*/
#ifndef __itkBinaryTreeSegmentationImageFilter_h
#define __itkBinaryTreeSegmentationImageFilter_h

#include "itkImage.h"
#include "itkImageToImageFilter.h"
#include "itkSegmentSpatialObject.h"

#include "itkImageRegionConstIterator.h"

#include "itkImageRegionIterator.h"
#include "itkNeighborhoodIterator.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkSpatialObjectTreeContainer.h"
#include "itkEllipseSpatialObject.h"



namespace itk
{
/** \class BinaryTreeSegmentationImageFilter
 *
 * \brief This filter performs segments a tree following the Philips framework for
 *  segmentation of tree-likestructures.
 * \author Xabier Artaechevarria, CIMA, University of Navarra,
 *
 *  Reserved values for the output:
 * 				- 2 is for wavefront voxels
 *				- 3 is used for the wavefront bifurcation check
 *				- 3, 4 and 5 are used to initialize segments
 */
template <typename TInputImage, typename TOutputImage = TInputImage>
class ITK_EXPORT BinaryTreeSegmentationImageFilter :
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef BinaryTreeSegmentationImageFilter 				Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > 	Superclass;
  typedef SmartPointer<Self> 								Pointer;
  typedef SmartPointer<const Self>  						ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro(BinaryTreeSegmentationImageFilter, ImageToImageFilter);

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  typedef typename TOutputImage::PixelType 				OutputPixelType;
  typedef typename TInputImage::PixelType 				InputPixelType;

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension );
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /** Image typedef support */
  typedef TInputImage  							InputImageType;
  typedef TOutputImage 							OutputImageType;
  typedef typename InputImageType::Pointer 		InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::IndexType   InputIndexType;
  typedef typename OutputImageType::IndexType	OutputIndexType;
  typedef Image<float, 3> PotentialImageType;
  typedef typename PotentialImageType::Pointer PotentialImagePointer;
  typedef typename PotentialImageType::IndexType PotentialIndexType;
  
  /**Typedef airway tree*/
  typedef SpatialObjectTreeContainer<3>	TreeType;
  typedef EllipseSpatialObject<3>       EllipseType;
  
  // define min-heaps
  typedef typename std::priority_queue< PotentialAndIndex, typename std::vector<PotentialAndIndex>, typename std::greater<PotentialAndIndex> > MinHeapOfPotentialAndIndex;
  typedef typename std::priority_queue<float, typename std::vector<float>,typename std::greater<float> > OrderedFloatType;
  
  /** Iterator typedef*/
  typedef ImageRegionConstIterator<TInputImage> ConstIteratorType;
  typedef ImageRegionIterator<TOutputImage> OutputIteratorType;
  typedef ConstNeighborhoodIterator<InputImageType> NeighborhoodIteratorType;
  typedef NeighborhoodIterator<OutputImageType> NeighborhoodOutputIteratorType;
  typedef ImageRegionIterator<PotentialImageType> PotentialImageIteratorType;
  /** Offset typdef*/
  typedef typename NeighborhoodOutputIteratorType::OffsetType OffsetType;
  
  
  /** Segment typedefs */
  typedef typename SegmentSpatialObject::Pointer SegmentPointerType;

  typedef typename SegmentSpatialObject::PointListType PointListType;

  /** Superclass typedefs. */
  typedef typename Superclass::OutputImageRegionType OutputImageRegionType;

  /**Methods to set and get propagation sigma, which allows for a  tolerance
  in propagation intensity*/
  itkSetMacro( PropagationSigma, float);
  itkGetMacro( PropagationSigma, float);
  itkSetMacro( BetaRadius, float);
  itkGetMacro( BetaRadius, float);
  itkSetMacro( TimeStep, float);
  itkGetMacro( TimeStep, float);

  /** Method to set initial seeds*/
  void SetSeed(typename TInputImage::IndexType);
  
  /** Method to set text file name*/
  void SetTextFileName(char*);

  /** Method to get text file name*/
  char* GetTextFileName()
    {
    return m_TextFileName;
    }
protected:
  BinaryTreeSegmentationImageFilter();
  virtual ~BinaryTreeSegmentationImageFilter() {}

  virtual void GenerateInputRequestedRegion ();
  void PrintSelf(std::ostream&, Indent) const;
  void GenerateData ();


  
  /** Method that performs Fast Marching propagation with fixed threshold*/
  SegmentPointerType PropagateWavefrontFixedThreshold(SegmentPointerType, InputImageConstPointer, OutputImagePointer, OutputImagePointer, PotentialImagePointer, float);
  
  /** Method that performs Fast Marching propagation with variable threshold*/
  SegmentPointerType PropagateWavefrontVariableThreshold(SegmentPointerType, InputImageConstPointer, OutputImagePointer, OutputImagePointer, PotentialImagePointer, float);

  OutputImagePointer MarkWavefrontInOutputImage(OutputImagePointer,  PointListType, PointListType);

  OutputImagePointer MarkWavefrontInOutputImage(OutputImagePointer,  PointListType);

  OutputImagePointer UnMarkWavefrontInOutputImage(OutputImagePointer,  PointListType);

  bool BifurcationCheck(SegmentPointerType, OutputImagePointer);

  bool StartSegments(SegmentPointerType, OutputImagePointer);

  void ComputeWavefrontStatistics(SegmentPointerType, InputImageConstPointer);

  void ComputeSegmentStatistics(SegmentPointerType, InputImageConstPointer);

  bool EvaluatePropagation(SegmentPointerType, OutputImagePointer, OutputImagePointer);

  bool EvaluateSegmentAndPossiblyAddToTree(SegmentPointerType, OutputImagePointer, OutputImagePointer);

  bool EqualWavefronts(SegmentSpatialObject::PointListType, SegmentSpatialObject::PointListType);

  int IsInAcceptedSegments(OutputIndexType);

  bool FindSegmentInList(std::vector<SegmentPointerType>, int);

  float ComputeCompactness(SegmentPointerType, OutputImagePointer);
  
  /**Method that returns 6 neighbors (if they are all inside the image)*/
  void GetSixNeighbors(OutputIndexType, typename std::vector<OutputIndexType>*);
  
  /**Method that returns 3 U values to compute the final potential value*/
  void GetUVector(OrderedFloatType*, PotentialImagePointer, PotentialIndexType*);
  
  /**Method that computes and returns viscosity from a certain index of the image*/
  float ComputeViscosity(InputImageConstPointer, InputIndexType*, InputPixelType*);
  
  /**Method that computes and returns potential u from 3 U values and viscosity*/
  float ComputePotential(OrderedFloatType*, const float*);
  
  /**Method that solves simple differential equation*/
  float SolveDifferentialEquation(typename std::vector<float>*, const float*);
 
  /**Method to know is a float is Nan*/
  bool isnan(float);
  
  /**Method to write segment data into text file*/
  void WriteSegmentData(SegmentPointerType,float);
  
private:
  BinaryTreeSegmentationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  std::vector<SegmentSpatialObject::Pointer> 	m_SegmentQueue;
  std::vector<SegmentSpatialObject::Pointer> 	m_AcceptedSegments;
  float 					m_Gamma;
  float 					m_PropagationSigma;
  float						m_SobelSigma;
  float						m_BetaRadius;
  std::vector<typename TInputImage::IndexType> 	m_Seeds;
  TreeType::Pointer    				m_AirwayTree;
  int						m_AirwayNumber;
  float			 			m_Threshold_lr;
  float						m_MaxGrowthRate;
  float						m_GradientThreshold;
  float						m_TimeStep;
  MinHeapOfPotentialAndIndex 			m_TrialMinHeap;

  // offsets for fast marching
  std::vector<OffsetType> m_SixOffsets;
  char*						m_TextFileName;

  /*ShapedNeighborhoodIteratorType::OffsetType offset1 = {{1,0,0}};;
  ShapedNeighborhoodIteratorType::OffsetType offset2 = {{-1,0,0}};;
  typename ShapedNeighborhoodIteratorType::OffsetType offset3 = {{0,1,0}};;
  typename ShapedNeighborhoodIteratorType::OffsetType offset4 = {{0,-1,0}};;
  typename ShapedNeighborhoodIteratorType::OffsetType offset5 = {{0,0,1}};;
  typename ShapedNeighborhoodIteratorType::OffsetType offset6 = {{0,0,-1}};;*/
  
  /*unsigned int m_N1;
  unsigned int m_N2;
  unsigned int m_MaximumScale;
  long double m_mask[16][5][5][8][8];*/
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBinaryTreeSegmentationImageFilter.txx"
#endif

#endif
