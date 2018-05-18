/*=========================================================================


=========================================================================*/

#ifndef __itkSegmentSpatialObject_txx
#define __itkSegmentSpatialObject_txx

#include "itkSegmentSpatialObject.h"
#include "itkExceptionObject.h"

#include "itkImageRegionIterator.h"

namespace itk
{
  bool 
  SegmentSpatialObject
  ::AddPoint(PointType pointToAdd)
  {
    BlobPointType newPoint;
    newPoint.SetPosition(pointToAdd);
    this->GetPoints().push_back(newPoint);
    return true;
  }
  
  void SegmentSpatialObject
  ::RemoveLastAddedPoints(int numPoints)
  { 
  
    this->GetPoints().erase(this->GetPoints().end()-numPoints,this->GetPoints().end());
  }
  
  void SegmentSpatialObject
  ::AddWavefrontPoint(PointType &pointToAdd)
  {
    BlobPointType newPoint;
    newPoint.SetPosition(pointToAdd);
    this->GetWavefront().push_back(newPoint);
  }
  
  /** Add point to the centerline */
  bool 
    SegmentSpatialObject
    ::AddCenterlinePoint(PointType &pointToAdd)
    {
      BlobPointType newPoint;
      newPoint.SetPosition(pointToAdd);
      this->GetCenterline().push_back(newPoint);
      return true;
  }
  
  void SegmentSpatialObject
  ::RemoveLastWavefrontPoint()
  {
    m_Wavefront.erase(m_Wavefront.end()-1);
  }
  
  /** Get the list of points that define the centerline */
  SegmentSpatialObject::PointListType &  
  SegmentSpatialObject
  ::GetCenterline() 
  { 
    itkDebugMacro( "Getting Centerline list" );
    return m_Centerline;
  } 
  
  /** Set the points which are defining the wavefront */
    void  
  SegmentSpatialObject
  ::SetWavefront( PointListType & points )  
  {
    // in this function, passing a null pointer as argument will
    // just clear the list...
    m_Wavefront.clear();
          
    PointListType::iterator it,end;
    it = points.begin();
    end = points.end();
    while(it != end)
      {
      m_Wavefront.push_back(*it);
      it++;
      }
      
    this->ComputeBoundingBox();
    this->Modified();
  }
  
  /** Set the points which are defining the initial wavefront */
  void  
  SegmentSpatialObject
  ::SetInitialWavefront( PointListType & points )  
  {
    // in this function, passing a null pointer as argument will
    // just clear the list...
    m_InitialWavefront.clear();
          
    PointListType::iterator it,end;
    it = points.begin();
    end = points.end();
    while(it != end)
      {
      m_InitialWavefront.push_back(*it);
      it++;
      }
      
    this->ComputeBoundingBox();
    this->Modified();
  } 
    
  
  /** Get the list of points that conform the wavefront */
  SegmentSpatialObject::PointListType &  
  SegmentSpatialObject
  ::GetWavefront() 
  { 
    itkDebugMacro( "Getting BlobPoint list" );
    return m_Wavefront;
  }
  
  void SegmentSpatialObject
  ::ComputeRadiusAndCheckIfMinimum( void )
  {
    float W = this->GetWavefront().size();
    float radius = sqrt(W/3.1416);
    if(radius < m_MinRadius && radius>1)
      {
      m_MinRadius = radius;
      }
  }
  
  void SegmentSpatialObject
  ::ComputeNewCenterlinePoint( void )
  {
    PointListType wavefront = this->GetWavefront();
    PointListType::iterator it, end;
    it = wavefront.begin();
    end = wavefront.end();
    //std::cout << "Entered wavefront gravity center computation" << std::endl;
    int x = 0, y = 0, z = 0, count = 0;
    while(it != end)
      {
      x = x + (*it).GetPosition()[0];
      y = y + (*it).GetPosition()[1];
      z = z + (*it).GetPosition()[2];
      count++;
      ++it;
      }
    //std::cout << "After loop in centerline point computation" << std::endl;
    if(wavefront.size()>0)
      {
      PointType point;
      point[0]= x/count;
      point[1]= y/count;
      point[2]= z/count;
      //std::cout << point << std::endl,
      AddCenterlinePoint( point );
      }
  }
  
  /** Check whether a point is in the list */
  bool SegmentSpatialObject
  ::PointIndexIsInside( PointType pointToCheck)
  {
    PointListType allPoints = this->GetPoints();
    PointListType::iterator it, begin;
    it = allPoints.end();
    --it;
    begin = allPoints.begin();
    PointType p;
    while(it != begin)
      {
      p = (*it).GetPosition();
      if(pointToCheck[0]==p[0] && pointToCheck[1]==p[1] && pointToCheck[2]==p[2])
        {
        return true;
        }
      --it;
      }
      return false;
  }
  
  /** Check whether the length/radius ratio exceeds a threshold */
  bool SegmentSpatialObject
  ::SegmentIsFullyGrown( float T_lr)
  {
    float l = this->GetLength();
    this->ComputeMeanRadius();
    float r = this->GetMeanRadius(); 
    if (l/r < T_lr)
      {
      
      return false;
      }
    else
      {
      //std::cout << "L: " << l <<std::endl;
      //std::cout << "R: " << r <<std::endl;
      //std::cout << "W: " << this->GetWavefront().size() <<std::endl;
      //std::cout << "L/R: " << l/r <<std::endl;
      return true;
      }
  }
  
  void SegmentSpatialObject
  ::AddTrialPoint(PotentialAndIndex* newPoint)
  {
    m_TrialMinHeap.push(*newPoint);
  }
  
  void SegmentSpatialObject
  ::SetTrialPoints(MinHeapOfPotentialAndIndex* newTrialPoints)
  {
    m_TrialMinHeap = *newTrialPoints;
  }
  
  std::priority_queue< PotentialAndIndex, std::vector<PotentialAndIndex>,
  std::greater<PotentialAndIndex> > SegmentSpatialObject
  ::GetTrialPoints(void)
  {
    return m_TrialMinHeap;
  }
  
  void SegmentSpatialObject
  ::UpdateWavefrontGrowthVector(int wavefrontSize)
  {
    m_WavefrontGrowth.push_back(wavefrontSize);
  }
  
  float SegmentSpatialObject
  ::GetGrowthRate(void)
  {
    float rate = 0;
    std::vector<int>::iterator it1, it2, end;
    it1 = m_WavefrontGrowth.begin();
    it2 = m_WavefrontGrowth.begin();
    ++it2;
    end = m_WavefrontGrowth.end();
    while(it2!=end)
      {
      rate = rate + float(*it2)/float(*it1);
      //std::cout << "Rate: " << rate << std::endl;
      ++it1;
      ++it2;
      }
    if(m_WavefrontGrowth.size() > 1)
      {
      rate = rate/(m_WavefrontGrowth.size()-1.0);
      }
    else
      {
      rate = 1;
      }
    return rate;
  }
  
  void SegmentSpatialObject
  ::AddGradientValues(std::vector<float> *newGradients)
  {
    std::vector<float>::iterator it, end;
    it = (*newGradients).begin();
    end = (*newGradients).end();
    while(it!=end)
      {
      m_Gradients.push_back(*it);
      ++it;
      }
  }
  
  
  std::vector<float> & SegmentSpatialObject
  ::GetGradientValues(void)
  {
    return  m_Gradients;
  }
  
  void SegmentSpatialObject
  ::AddNeighborSegment(int ID)
  {
    m_NeighborSegments.push_back(ID);
  }
  
  std::vector<int> & SegmentSpatialObject
  ::GetNeighborSegments(void)
  {
    return m_NeighborSegments;
  }
  
  float SegmentSpatialObject
  ::GetLength(void)
  {
    float x_dist, y_dist, z_dist, length = 0;
    PointListType::iterator it1,it2, end;
    it1 = m_Centerline.begin();
    it2 = m_Centerline.begin();
    ++it2;
    end = m_Centerline.end();
    while(it2!=end)
      {
      x_dist = ((*it2).GetPosition()[0]-(*it1).GetPosition()[0]);
      y_dist = ((*it2).GetPosition()[1]-(*it1).GetPosition()[1]);
      z_dist = ((*it2).GetPosition()[2]-(*it1).GetPosition()[2]);
      length = length +  sqrt(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);
      ++it1;
      ++it2;
      }
    return length;
  }
  
  void SegmentSpatialObject
  ::ComputeMeanRadius(void)
  {
    float meanRadius = 0;
    int count = 0;
    std::vector<int>::iterator it, end;
    it = m_WavefrontGrowth.begin();
    end  = m_WavefrontGrowth.end();
    while(it!=end)
      {
      meanRadius = meanRadius + sqrt(float(*it)/3.1416);
      count = count + 1;
      ++it;
      }
    this->SetMeanRadius(meanRadius/float(count));
  }

}
#endif
