/*=========================================================================

  SegmentSpatialObject defines the basic SO for the segmentation of tubular
  structures using a generic framework

=========================================================================*/
#ifndef __itkSegmentSpatialObject_h
#define __itkSegmentSpatialObject_h
#include "itkBlobSpatialObject.h"
#include <queue>
#include "PotentialAndIndex.h"
namespace itk
{
class SegmentSpatialObject
  :public BlobSpatialObject<3>
{
public:
  typedef SegmentSpatialObject  			     Self;
  typedef BlobSpatialObject< 3 >			     Superclass;
  typedef SmartPointer < Self > 			     Pointer;
  typedef SmartPointer < const Self >            ConstPointer;
  typedef Superclass::PointType			         PointType;
  typedef Superclass::TransformType				 TransformType;
  typedef Superclass::PointListType				 PointListType;
  typedef Superclass::BlobPointType			     BlobPointType;
  typedef std::priority_queue< PotentialAndIndex, std::vector<PotentialAndIndex>, std::greater<PotentialAndIndex> > MinHeapOfPotentialAndIndex;
  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Method for creation through the object factory. */
  itkTypeMacro( SegmentSpatialObject, BlobSpatialObject );

  /** Methods to set/get variables */
  itkSetMacro( IsPending, bool);
  itkSetMacro( IsGrown, bool);
  itkGetMacro( IsPending, bool);
  itkGetMacro( IsGrown, bool);

  itkSetMacro( WavefrontMean, float);
  itkSetMacro( WavefrontStdDeviation, float);
  itkGetMacro( WavefrontMean, float);
  itkGetMacro( WavefrontStdDeviation, float);
  itkSetMacro( Mean, float);
  itkSetMacro( StdDeviation, float);
  itkGetMacro( Mean, float);
  itkGetMacro( StdDeviation, float);

  itkSetMacro( MinRadius, float);
  itkGetMacro( MinRadius, float);

  itkSetMacro( MeanGradient, float);
  itkGetMacro( MeanGradient, float);
  itkSetMacro( GradientStdDeviation, float);
  itkGetMacro( GradientStdDeviation, float);

  itkSetMacro( MeanRadius, float);
  itkGetMacro( MeanRadius, float);

  /** Method to set add individual point*/
  bool AddPoint(PointType point);

  /** Method to remove last added points*/
  void RemoveLastAddedPoints(int);

  /** Method to set the wavefront. */
  void  SetWavefront( PointListType & newPoints );

  /** Method to set the initial wavefront. */
  void  SetInitialWavefront( PointListType & newPoints );

  /** Method to add individual wavefront value. */
  void  AddWavefrontPoint( PointType & point );

  /** Method to get the wavefront. */
  PointListType & GetWavefront( void );

  /** Method to get the centerline. */
  PointListType & GetCenterline( void );

  //** Method to set add individual centerline point/
  bool AddCenterlinePoint(PointType &point);
  //** Method to compute the center of gravity of the current wavefront */
  void ComputeNewCenterlinePoint( void );

  /** Method to know whether a point is included in the point list */
  bool PointIndexIsInside( PointType );
  
  /** Method to add one trial point */
  void AddTrialPoint( PotentialAndIndex* );
  
  /** Method to add whole trial min-heap */  
  void SetTrialPoints(MinHeapOfPotentialAndIndex*  );

  /** Method to get trial min-heap */ 
  MinHeapOfPotentialAndIndex GetTrialPoints( void );
  
  void ComputeRadiusAndCheckIfMinimum( void );

  bool SegmentIsFullyGrown(float);

  float GetLength(void);

  void UpdateWavefrontGrowthVector(int);

  void AddGradientValues(std::vector<float>*);

  std::vector<float> & GetGradientValues();

  void ComputeMeanRadius(void);

  void AddNeighborSegment(int);

  void RemoveLastWavefrontPoint();

  std::vector<int> & GetNeighborSegments();

  float GetGrowthRate();

private:
  SegmentSpatialObject(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented


  PointListType             m_Centerline;
  PointListType				m_Wavefront;
  std::vector<float>        m_Gradients;
  std::vector<int>			m_NeighborSegments;
  PointListType				m_InitialWavefront;
  bool						m_IsPending;
  bool						m_IsGrown;
  float						m_WavefrontMean;
  float 					m_MeanRadius;
  float						m_WavefrontStdDeviation;
  float						m_Mean;
  float						m_StdDeviation;
  float						m_MeanGradient;
  float						m_GradientStdDeviation;
  float						m_MinRadius;
  std::vector<int>          m_WavefrontGrowth;
  MinHeapOfPotentialAndIndex  m_TrialMinHeap;
  SegmentSpatialObject()
    {
	  m_MinRadius = 10000;
  //  m_Orientation = Unknown;
  //  m_Thickness = 0.0;
    }
};
}
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSegmentSpatialObject.txx"
#endif

#endif
