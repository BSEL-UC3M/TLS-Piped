/*=========================================================================


=========================================================================*/
#ifndef _itkBinaryTreeSegmentationImageFilter_txx
#define _itkBinaryTreeSegmentationImageFilter_txx

#include "itkBinaryTreeSegmentationImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodIterator.h"

#include "itkConstantBoundaryCondition.h"

#include "itkSegmentSpatialObject.h"
#include "itkEllipseSpatialObject.h"

#include "itkSpatialObjectToImageFilter.h"

#include "itkConstShapedNeighborhoodIterator.h"

#include "itkCastImageFilter.h"


namespace itk
{

template <typename TInputImage, typename TOutputImage>
void
BinaryTreeSegmentationImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

template <typename TInputImage, typename TOutputImage>
BinaryTreeSegmentationImageFilter<TInputImage, TOutputImage>
::BinaryTreeSegmentationImageFilter() 
{
  m_Gamma = 2;
  m_AirwayNumber = 1;
  m_Threshold_lr = 400;
  m_MaxGrowthRate = 1.2;
  m_AirwayTree = TreeType::New();
  m_GradientThreshold = 1800;
  m_SobelSigma = 2.5;
  m_TimeStep = 1;
  
  // add offsets to the offset vector
  OffsetType offset1 = {{1,0,0}};
  m_SixOffsets.push_back(offset1);
  OffsetType offset2 = {{-1,0,0}};
  m_SixOffsets.push_back(offset2);
  OffsetType offset3 = {{0,1,0}};
  m_SixOffsets.push_back(offset3);
  OffsetType offset4 = {{0,-1,0}};
  m_SixOffsets.push_back(offset4);
  OffsetType offset5 = {{0,0,1}};
  m_SixOffsets.push_back(offset5);
  OffsetType offset6 = {{0,0,-1}};
  m_SixOffsets.push_back(offset6);
  
    
}

template< typename TInputImage, typename TOutputImage >
void
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion ()
{
   // call the superclass's implementation of this method
  Superclass::GenerateInputRequestedRegion();

  if ( !this->GetInput() )
    {
    return;
    }

  // get pointers to the input 
  InputImagePointer  inputPtr  =   const_cast< InputImageType *>( this->GetInput() );

  // Request the entire input image
  InputImageRegionType inputRegion;
  inputRegion = inputPtr->GetLargestPossibleRegion();
  
  inputPtr->SetRequestedRegion(inputRegion);

  return;
}

template< typename TInputImage, typename TOutputImage >
void
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::SetSeed(typename TInputImage::IndexType seed)
{
  m_Seeds.push_back(seed);
}



template< typename TInputImage, typename TOutputImage >
typename SegmentSpatialObject::Pointer
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::PropagateWavefrontFixedThreshold(typename SegmentSpatialObject::Pointer inputSegment, InputImageConstPointer inputImage, OutputImagePointer outputImage, OutputImagePointer
segmentImage, PotentialImagePointer potentialImage, float timeStep)
{
  // this method propagates the wavefront one time-step using Fast Marching
  // returns the propagated new segment, which includes a new wavefront
  SegmentSpatialObject::PointListType oldWavefront;
  oldWavefront = inputSegment->GetWavefront();
  SegmentPointerType propagatedSegment = inputSegment;
  
    
  // propagation is done with Fast Marching algorithm (Deschamps and Cohen, Fast extraction of minimal paths in 3D images...)
  // Two checks are done
   // 	1. Intensity rule: Variable threshold
   //   2. Exclusivity: does not belong to another segment
  
  // mark the wavefront in the output image (as 2)
  //outputImage = this->MarkWavefrontInOutputImage(outputImage, oldWavefront);
  
  MinHeapOfPotentialAndIndex trialMinHeap = inputSegment->GetTrialPoints();
  if(trialMinHeap.size()==0)
    {
    return inputSegment;
    }
  //std::cout << "Trial points size: "<<trialMinHeap.size() << std::endl; 
  // get initial time step to have the reference
  float init_time = trialMinHeap.top().Potential;
  
  // move the smallest trial point to the alive set (the alive set is the output image)
  PotentialAndIndex trial;
  OutputIndexType index;
  OutputIteratorType outIt = OutputIteratorType( outputImage, outputImage->GetLargestPossibleRegion() );
  ConstIteratorType inIt = ConstIteratorType( inputImage, inputImage->GetLargestPossibleRegion() );
  
  bool march = true;
  MinHeapOfPotentialAndIndex TrialMinHeap_aux;
  
  typename SegmentSpatialObject::PointListType newPoints;
  typename SegmentSpatialObject::PointType p;
  typename SegmentSpatialObject::BlobPointType newPoint;
  float growth=0;
  InputPixelType threshold = -300;
  //std::cout << "New propagation: " << std::endl;
  
  while(march)
    {
    
   // std::cout << "Trial min-heap size:" << trialMinHeap.size() << std::endl;
    
          
    // select point with min U and add to alive
    if(!trialMinHeap.empty())
      {
      PotentialAndIndex trial = trialMinHeap.top();
      growth = trial.Potential - init_time;
      if(growth < 0)
        {
	//std::cout << "negative growth" << std::endl;
	break;
	}
      /*if(growth<0)
       {
       int loko;
       std::cin >> loko;
       std::cout << growth << std::endl;
       }*/
      //std::cout << "Growth: " << trial.Potential - init_time << std::endl;
      trialMinHeap.pop();
      index[0]= trial.x;
      index[1] = trial.y;
      index[2] = trial.z;  

      outIt.SetIndex(index);
      outIt.Set(1);
      p[0] = index[0];
      p[1] = index[1];
      p[2] = index[2];
      newPoint.SetPosition(p);
      newPoints.push_back(newPoint);
      propagatedSegment->AddPoint(p);
      // for debugging
      /*if(newPoints.size()==1)
        {
	std::cout << "First trial point: " << index << std::endl;
	}*/
      //std::cout << "Trial potential: " << trial.Potential << std::endl;
      if(trial.Potential - init_time > timeStep)
        {
	
        march = false;
        }
      }
    else
      {
      //std::cout << "Trial min-heap is empty!" << std::endl;   
      return inputSegment;
      }
  
    // get six neighbors and compute U
    typename std::vector<OutputIndexType> neighIndex;
    GetSixNeighbors(index, &neighIndex);
    
    typename std::vector<OutputIndexType>::iterator neighIt, neighEnd;
    neighIt = neighIndex.begin();
    neighEnd = neighIndex.end();
    PotentialImageIteratorType potIt = PotentialImageIteratorType( potentialImage, potentialImage->GetLargestPossibleRegion() );
    float potential = 0;
    int outValue = 0;
    PotentialIndexType potIndex;
    OrderedFloatType U_vector;
    float viscosity = INFINITY;
    float U = 0;
    short imageValue = 0;
    
    
    while(neighIt!=neighEnd)
      {
      //if it is far, compute it
      // check that it is far by looking at potential image
      for(int i=0; i < 3; i++)
        {
        potIndex[i] = (*neighIt)[i];
        index[i] = (*neighIt)[i];
	
        }
      potIt.SetIndex(potIndex);
      potential = potIt.Get();
      outIt.SetIndex(index);
      outValue = outIt.Get();
    
      if(outValue == 0)
        {
        // it is a far point: compute U and add to trial min-heap
	
	// get U vector (three values)
        GetUVector( &U_vector, potentialImage, &potIndex);
        
	
	
        // compute viscosity (a function of image intensity)
        viscosity = ComputeViscosity( inputImage, &index, &threshold);
      
        // compute U using the 3 ordered U values and viscosity
        U = ComputePotential( &U_vector, &viscosity );
	
	if(U!=INFINITY)
	  {
	  // add to trial min-heap (3 steps: min-heap, out-image and potential image)
	  // potential image
	  potIt.Set(U);
	  // output image: mark as trial
	  outIt.Set(2);
	  // add to trial min heap
	  trial.x = index[0];
	  trial.y = index[1];
	  trial.z = index[2];
	  trial.Potential = U;
          trialMinHeap.push(trial);
	  }
	else
	  {
	  // if it is infinite, add to alive (fix it? is it OK?)
	  potIt.Set(U);
	  // output image: mark as alive
	  outIt.Set(1);
	  }
        }
      else if (outValue == 1)
        {
        //std::cout << "Do not calculate, it is already alive" << std::endl;
        }
      else
        {
        // it is trial. Recompute and update min-heap.
	GetUVector( &U_vector, potentialImage, &potIndex);
      
        // compute viscosity (a function of image intensity)
        viscosity = ComputeViscosity( inputImage, &index, &threshold);
      
        // compute U using the 3 ordered U values and viscosity
        U = ComputePotential( &U_vector, &viscosity );
	
	if(U!=INFINITY)
	  {
	  // now, change potential image
	  potIt.Set(U);
	  // output image is OK, it is already 2
	  // min-heap has to be reordered with the new U value
	  while(!trialMinHeap.empty())
	    {
            // read the x,y,z
	    trial = trialMinHeap.top();
	    trialMinHeap.pop();
	    if(index[0]==trial.x && index[1]==trial.y && index[2]==trial.z)
	      {
	      trial.Potential =  U;
	      TrialMinHeap_aux.push(trial);
	      }
	    else
	      {
	      TrialMinHeap_aux.push(trial);
	      }
	    }
	  trialMinHeap = TrialMinHeap_aux;
	  // empty auxiliary priority queue
	  while(!TrialMinHeap_aux.empty())
	    {
	     TrialMinHeap_aux.pop();
	    }
	  }
	else
	  {
	   // if it is infinite, add to alive (fix it? is it OK?)
	  potIt.Set(U);
	  // output image: mark as alive
	  outIt.Set(1);
	  }
        }
      ++neighIt;
      //std::cout << "Potential is: " << U << std::endl;
      //std::cout << "in index: " << potIndex << std::endl;
      }
     
    
    }  
  
  propagatedSegment->SetTrialPoints(&trialMinHeap);
  propagatedSegment->SetWavefront(newPoints);
  propagatedSegment->ComputeNewCenterlinePoint();
  propagatedSegment->UpdateWavefrontGrowthVector(newPoints.size()); // this is required to later compute growth rate
  return propagatedSegment;
  
}

template< typename TInputImage, typename TOutputImage >
typename SegmentSpatialObject::Pointer
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::PropagateWavefrontVariableThreshold(typename SegmentSpatialObject::Pointer inputSegment, InputImageConstPointer inputImage, OutputImagePointer outputImage, OutputImagePointer
segmentImage, PotentialImagePointer potentialImage, float timeStep)
{
  // this method propagates the wavefront one time-step using Fast Marching
  // returns the propagated new segment, which includes a new wavefront
  SegmentSpatialObject::PointListType oldWavefront;
  oldWavefront = inputSegment->GetWavefront();
  SegmentPointerType propagatedSegment = inputSegment;
  
    
  // propagation is done with Fast Marching algorithm (Deschamps and Cohen, Fast extraction of minimal paths in 3D images...)
  // Two checks are done
   // 	1. Intensity rule: Variable threshold
   //   2. Exclusivity: does not belong to another segment
  
  // mark the wavefront in the output image (as 2)
  //outputImage = this->MarkWavefrontInOutputImage(outputImage, oldWavefront);
  
  MinHeapOfPotentialAndIndex trialMinHeap = inputSegment->GetTrialPoints();
  if(trialMinHeap.size()==0)
    {
    return inputSegment;
    }
  //std::cout << "Trial points size: "<<trialMinHeap.size() << std::endl; 
  // get initial time step to have the reference
  float init_time = trialMinHeap.top().Potential;
  
  // move the smallest trial point to the alive set (the alive set is the output image)
  PotentialAndIndex trial;
  OutputIndexType index;
  OutputIteratorType outIt = OutputIteratorType( outputImage, outputImage->GetLargestPossibleRegion() );
  ConstIteratorType inIt = ConstIteratorType( inputImage, inputImage->GetLargestPossibleRegion() );
  
  bool march = true;
  MinHeapOfPotentialAndIndex TrialMinHeap_aux;
  
  typename SegmentSpatialObject::PointListType newPoints;
  typename SegmentSpatialObject::PointType p;
  typename SegmentSpatialObject::BlobPointType newPoint;
  float growth=0;
  InputPixelType threshold = -625;
  
  
  // get values for intensity limits (parent is considered if exists)
  float meanSegment = 0, stdDev =0;
  int sonPointNumber = 0, parentPointNumber = 0;
  float sigma = m_PropagationSigma;
  this->ComputeSegmentStatistics(inputSegment, inputImage);
  if(inputSegment->HasParent())
    {
    SegmentSpatialObject * parent = dynamic_cast<SegmentSpatialObject*> (inputSegment->GetParent());
    sonPointNumber = inputSegment->GetPoints().size();
    parentPointNumber = parent->GetPoints().size();
    // if son is not large enough, we do weighted mean
    //if(parentPointNumber/3 > sonPointNumber) 
    if(false)
      {
      meanSegment = (sonPointNumber*inputSegment->GetMean() + parentPointNumber*parent->GetMean())/(sonPointNumber+parentPointNumber);
      }
    else
      {
      meanSegment = (inputSegment->GetMean() + parent->GetMean())/2;
      stdDev = parent->GetStdDeviation();
      // to try if maximum standard deviation improves
      if(parent->HasParent())
        {
        SegmentSpatialObject * grandParent = dynamic_cast<SegmentSpatialObject*> (parent->GetParent());
        float stdDevGrandPa = grandParent->GetStdDeviation();
        if(stdDevGrandPa>stdDev)
          {
          stdDev = stdDevGrandPa;
          }
        }
      }
    }
  else
    {
    meanSegment = inputSegment->GetMean();
    stdDev = inputSegment->GetStdDeviation();
    }
  
  threshold = meanSegment+stdDev*sigma;  
  
  //std::cout << "New propagation with Thrs: " << threshold << std::endl;
  
  while(march)
    {
    
   // std::cout << "Trial min-heap size:" << trialMinHeap.size() << std::endl;
    
          
    // select point with min U and add to alive
    if(!trialMinHeap.empty())
      {
      PotentialAndIndex trial = trialMinHeap.top();
            
           
      growth = trial.Potential - init_time;
      if(growth < 0)
        {
	//std::cout << "negative growth" << std::endl;
	//std::cout << growth << std::endl;
	//std::cout << trial.x << ", " << trial.y << ", " << trial.z << std::endl;
	//std::cout << trial.Potential << std::endl;
	int loko;
        std::cin >> loko;
	break;
	}
      
       
       
      //std::cout << "Growth: " << trial.Potential - init_time << std::endl;
      trialMinHeap.pop();
      index[0]= trial.x;
      index[1] = trial.y;
      index[2] = trial.z;  

      outIt.SetIndex(index);
      outIt.Set(1);
      p[0] = index[0];
      p[1] = index[1];
      p[2] = index[2];
      newPoint.SetPosition(p);
      newPoints.push_back(newPoint);
      propagatedSegment->AddPoint(p);
      
      // for debugging
      /*if(newPoints.size()==1)
        {
	std::cout << "First trial point: " << index << std::endl;
	}*/
      //std::cout << "Trial potential: " << trial.Potential << std::endl;
      if(trial.Potential - init_time > timeStep)
        {
	
        march = false;
        }
      }
    else
      {
      //std::cout << "Trial min-heap is empty!" << std::endl;   
      return inputSegment;
      }
  
    // get six neighbors and compute U
    typename std::vector<OutputIndexType> neighIndex;
    GetSixNeighbors(index, &neighIndex);
    
    typename std::vector<OutputIndexType>::iterator neighIt, neighEnd;
    neighIt = neighIndex.begin();
    neighEnd = neighIndex.end();
    PotentialImageIteratorType potIt = PotentialImageIteratorType( potentialImage, potentialImage->GetLargestPossibleRegion() );
    float potential = 0;
    int outValue = 0;
    PotentialIndexType potIndex;
    OrderedFloatType U_vector;
    float viscosity = INFINITY;
    float U = 0;
    short imageValue = 0;
    
    
    while(neighIt!=neighEnd)
      {
      //if it is far, compute it
      // check that it is far by looking at potential image
      for(int i=0; i < 3; i++)
        {
        potIndex[i] = (*neighIt)[i];
        index[i] = (*neighIt)[i];
	
        }
      potIt.SetIndex(potIndex);
      potential = potIt.Get();
      outIt.SetIndex(index);
      outValue = outIt.Get();
    
      if(outValue == 0)
        {
        // it is a far point: compute U and add to trial min-heap
	
	// get U vector (three values)
        GetUVector( &U_vector, potentialImage, &potIndex);
        
	
	
        // compute viscosity (a function of image intensity)
        viscosity = ComputeViscosity( inputImage, &index, &threshold);
      
        // compute U using the 3 ordered U values and viscosity
        U = ComputePotential( &U_vector, &viscosity );

	growth = U - init_time;
	
	if(U!=INFINITY && growth > 0)
	  {
	  // add to trial min-heap (3 steps: min-heap, out-image and potential image)
	  // potential image
	  potIt.Set(U);
	  // output image: mark as trial
	  outIt.Set(2);
	  // add to trial min heap
	  trial.x = index[0];
	  trial.y = index[1];
	  trial.z = index[2];
	  trial.Potential = U;
          trialMinHeap.push(trial);
	  }
	else if (U==INFINITY)
	  {
	  // if it is infinite, add to alive (fix it? is it OK?)
	  potIt.Set(U);
	  // output image: mark as alive
	  outIt.Set(1);
	  }
        }
      else if (outValue == 1)
        {
        //std::cout << "Do not calculate, it is already alive" << std::endl;
        }
      else
        {
        // it is trial. Recompute and update min-heap.
	GetUVector( &U_vector, potentialImage, &potIndex);
      
        // compute viscosity (a function of image intensity)
        viscosity = ComputeViscosity( inputImage, &index, &threshold);
      
        // compute U using the 3 ordered U values and viscosity
        U = ComputePotential( &U_vector, &viscosity );
	
	growth = U - init_time;
	
	if(U!=INFINITY && growth > 0)
	  {
	  // now, change potential image
	  potIt.Set(U);
	  // output image is OK, it is already 2
	  // min-heap has to be reordered with the new U value
	  while(!trialMinHeap.empty())
	    {
            // read the x,y,z
	    trial = trialMinHeap.top();
	    trialMinHeap.pop();
	    if(index[0]==trial.x && index[1]==trial.y && index[2]==trial.z)
	      {
	      trial.Potential =  U;
	      TrialMinHeap_aux.push(trial);
	      }
	    else
	      {
	      TrialMinHeap_aux.push(trial);
	      }
	    }
	  trialMinHeap = TrialMinHeap_aux;
	  // empty auxiliary priority queue
	  while(!TrialMinHeap_aux.empty())
	    {
	     TrialMinHeap_aux.pop();
	    }
	  }
	else if (U==INFINITY)
	  {
	   // if it is infinite, add to alive (fix it? is it OK?)
	  potIt.Set(U);
	  // output image: mark as alive
	  outIt.Set(1);
	  }
        }
      ++neighIt;
      //std::cout << "Potential is: " << U << std::endl;
      //std::cout << "in index: " << potIndex << std::endl;
      }
     
    
    }  
  
  propagatedSegment->SetTrialPoints(&trialMinHeap);
  propagatedSegment->SetWavefront(newPoints);
  propagatedSegment->ComputeNewCenterlinePoint();
  propagatedSegment->UpdateWavefrontGrowthVector(newPoints.size()); // this is required to later compute growth rate
  return propagatedSegment;
  
}

template< typename TInputImage, typename TOutputImage >
float 
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::ComputeViscosity(InputImageConstPointer inputImage, InputIndexType* index, InputPixelType* threshold)
{
  
  // get image value
  ConstIteratorType inIt = ConstIteratorType( inputImage, inputImage->GetLargestPossibleRegion() );
  inIt.SetIndex(*index);
  InputPixelType imageValue = inIt.Get();
    
  // return viscosity depending on value of pixel
  float viscosity = INFINITY;
  if(imageValue > 0)
    {
    viscosity = 1;
    }
      
  return viscosity;
}

template< typename TInputImage, typename TOutputImage >
float
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::ComputePotential(OrderedFloatType* U_vector,const float* viscosity)
{
  // pass to vector removing infinite values
  float U_value = 0;
  std::vector<float> coeff;
  while(!U_vector->empty())
    {
    U_value = U_vector->top();
    U_vector->pop();
    if(U_value != INFINITY)
      {
      coeff.push_back(U_value);
      //std::cout << "U: "<< U_value << std::endl;
      }
    }
  //std::cout << "U vector was that" << std::endl;
  float potential = INFINITY;
  while(!coeff.empty())
    {
    potential = SolveDifferentialEquation( &coeff, viscosity );
    //std::cout << "Viscosity is: " << *viscosity << std::endl;
    //std::cout << "Computed potential is: " << potential << std::endl;
    if(isnan(potential))
      {
      // delete last element
      //std::cout << "Infinite potential detected" << std::endl;
      coeff.pop_back();
      continue;
      }
    if(potential >= coeff.back())
      {
      if(isnan(potential))
        {
	//std::cout << "Unexpected nan potential" << std::endl;
	//int loko;
	//std::cin >> loko;
	}      
      return potential;
      }
    else
      {
      coeff.pop_back();
      }
    }
          
  return potential;
}

template< typename TInputImage, typename TOutputImage >
float
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::SolveDifferentialEquation(typename std::vector<float>* coeff, const float* viscosity)
{
  // solution to equation depends on different conditions
  if((*viscosity)==INFINITY)
    {
    return INFINITY;
    }
  
  // consider different options
  if(coeff->size()==1)
    {
    // in this case solution is very simple
    return ((*coeff)[0] + *viscosity);
    }
  
  // compute discriminant
  float a = coeff->size();
  float b = 0, c = 0;
  std::vector<float>::iterator coeffIt, coeffEnd;
  coeffIt = coeff->begin();
  coeffEnd = coeff->end();
  while(coeffIt!=coeffEnd)
    {
    b = b + (*coeffIt);
    c = c + (*coeffIt)*(*coeffIt);
    coeffIt++;
    }
  b = -2*b;
  c = c - (*viscosity)*(*viscosity);

  // debugging
  //std::cout << "a: " << a << std::endl;
  //std::cout << "b: " << b << std::endl;
  //std::cout << "c: " << c << std::endl;
  
  float D = b*b -4*a*c;
  
  if(D < 0)
    {
    return std::numeric_limits<float>::quiet_NaN();
    }
  
  float d = sqrt(D);
  // return largest solution
  float potential = -b-d;

  if( -b+d > potential)
    {
    potential = -b+d;
    }
  return (potential)/2/a;
}

template< typename TInputImage, typename TOutputImage >
bool
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::isnan(float x)
{
  // Nan-s are different to themselves
  return ((x)!=(x));
}

template< typename TInputImage, typename TOutputImage >
void
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::GetUVector(OrderedFloatType *U_priority, PotentialImagePointer potentialImage, PotentialIndexType *potIndex)
{
  
  std::vector<float> temp_vector;
    
  typename PotentialImageType::RegionType::SizeType imageSize = potentialImage->GetLargestPossibleRegion().GetSize();
  // initialize iterator
  PotentialImageIteratorType potIt = PotentialImageIteratorType( potentialImage, potentialImage->GetLargestPossibleRegion() );
  PotentialIndexType newIndex;
  bool inside = true;
  
  // loop to compute from A1 to C2
  for(int k=0; k<6; k++)
    {
    inside = true;
    for(int i = 0; i < 3; i++)
      {
      newIndex[i] = (*potIndex)[i] + m_SixOffsets[k][i];
      if( newIndex[i] < 0 || newIndex[i] >= imageSize[i])
        {
        inside = false;
        }
      }
    if( inside==true)
      {
      potIt.SetIndex(newIndex);
      temp_vector.push_back(potIt.Get());
      }
    else
      {
      temp_vector.push_back(INFINITY);
      }
    }
    
    float A = temp_vector[0];
    if(temp_vector[1]<A)
      {
      A = temp_vector[1];
      }
    float B = temp_vector[2];
    if(temp_vector[3]<B)
      {
      B = temp_vector[3];
      }
    float C = temp_vector[4];
    if(temp_vector[5]<C)
      {
      C = temp_vector[5];
      }
    
    // use priority queue to order
    U_priority->push(A);
    U_priority->push(B);
    U_priority->push(C);
    
    
}

template< typename TInputImage, typename TOutputImage >
void
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::ComputeWavefrontStatistics( SegmentPointerType segment, InputImageConstPointer inputImage)
{
  // get wavefront and define iterators
  PointListType wavefront = segment->GetWavefront();
  typename PointListType::iterator it, end;
  it = wavefront.begin();
  end = wavefront.end();
  
  // define iterator for input image
  ConstIteratorType inputIt = ConstIteratorType(inputImage, inputImage->GetLargestPossibleRegion());
  
  double mean = 0, mean_sqr =0;
  double val = 0;
  double stdDev = 0;
  int count = 0;
  SegmentSpatialObject::PointType p;
  typename InputImageType::IndexType index;
  //compute mean
  while(it!=end)
    {
    p = (*it).GetPosition();
    index[0] = p[0];
    index[1] = p[1];
    index[2] = p[2];
    inputIt.SetIndex(index);
    val = inputIt.Get();
    mean = mean + val;
    mean_sqr = mean_sqr+val*val;
    count = count + 1;
    ++it;
    }
  mean = mean/count;
  stdDev = sqrt(mean_sqr/count-mean*mean);
  segment->SetWavefrontMean(mean);
  segment->SetWavefrontStdDeviation(stdDev);
}

template< typename TInputImage, typename TOutputImage >
void
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::ComputeSegmentStatistics( SegmentPointerType segment, InputImageConstPointer inputImage)
{
  // get wavefront and define iterators
  PointListType points = segment->GetPoints();
  PointListType::iterator it, end;
  it = points.begin();
  end = points.end();
  
  // define iterator for input image
  ConstIteratorType inputIt = ConstIteratorType(inputImage, inputImage->GetLargestPossibleRegion());
  
  double mean = 0, mean_sqr =0;
  double val = 0;
  double stdDev = 0;
  int count = 0;
  SegmentSpatialObject::PointType p;
  typename InputImageType::IndexType index;
  //compute mean
  while(it!=end)
    {
    p = (*it).GetPosition();
    index[0] = p[0];
    index[1] = p[1];
    index[2] = p[2];
    inputIt.SetIndex(index);
    val = inputIt.Get();
    mean = mean + val;
    mean_sqr = mean_sqr+val*val;
    count = count + 1;
    ++it;
    }
  mean = mean/count;
  stdDev = sqrt(mean_sqr/count-mean*mean);
  segment->SetMean(mean);
  segment->SetStdDeviation(stdDev);
  
  // now compute the gradient statistics
  std::vector<float> gradients = segment->GetGradientValues();
  float gradMean = 0, grad_mean_sqr = 0;
  float gradStdDev = 0;
  count = 0;
  std::vector<float>::iterator gradIt, gradEnd;
  gradIt = gradients.begin();
  gradEnd = gradients.end();
  while(gradIt!=gradEnd)
    {
    val = *gradIt;
    gradMean = gradMean + val;
    grad_mean_sqr = grad_mean_sqr + val*val;
    count=count+1;
    ++gradIt;
    }
  if(count>0)
    {
    gradMean = gradMean/count;
    gradStdDev = sqrt(grad_mean_sqr/count - gradMean*gradMean);
    }
  segment->SetMeanGradient(gradMean);
  segment->SetGradientStdDeviation(gradStdDev);
}

template< typename TInputImage, typename TOutputImage >
typename TOutputImage::Pointer
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::MarkWavefrontInOutputImage(OutputImagePointer output, PointListType oldWavefront,PointListType newWavefront)
{
  
  typename SegmentSpatialObject::PointListType::iterator it,end;
  it = oldWavefront.begin();
  end = oldWavefront.end();
  SegmentSpatialObject::PointType p;
  typename OutputImageType::IndexType index;
  typedef ImageRegionIterator<OutputImageType> OutputIteratorType;
  OutputIteratorType outIt = OutputIteratorType( output, output->GetLargestPossibleRegion() );
  
  // set old wavefront to 1
  while(it != end)
    {
    p = (*it).GetPosition();
    index[0] = p[0];
    index[1] = p[1];
    index[2] = p[2];
    outIt.SetIndex(index);
    outIt.Set(1);
    ++it;
    }
  // set new wavefront to 3
  it = newWavefront.begin();
  end = newWavefront.end();
  while(it != end)
    {
    p = (*it).GetPosition();
    index[0] = p[0];
    index[1] = p[1];
    index[2] = p[2];
    outIt.SetIndex(index);
    outIt.Set(3);
    ++it;
    }
  return output;
}

template< typename TInputImage, typename TOutputImage >
typename TOutputImage::Pointer
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::MarkWavefrontInOutputImage(OutputImagePointer output, PointListType newWavefront)
{
  SegmentSpatialObject::PointListType::iterator it,end;
  it = newWavefront.begin();
  end = newWavefront.end();
  typename SegmentSpatialObject::PointType p;
  typename OutputImageType::IndexType index;
  //typedef ImageRegionIterator<OutputImageType> OutputIteratorType;
  OutputIteratorType outIt = OutputIteratorType( output, output->GetLargestPossibleRegion() );
  // set new wavefront to 3
  while(it != end)
    {
    p = (*it).GetPosition();
    index[0] = p[0];
    index[1] = p[1];
    index[2] = p[2];
    outIt.SetIndex(index);
    outIt.Set(3);
    ++it;
    }
  return output;
}

template< typename TInputImage, typename TOutputImage >
typename TOutputImage::Pointer
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::UnMarkWavefrontInOutputImage(OutputImagePointer output, PointListType Wavefront)
{
  typename SegmentSpatialObject::PointListType::iterator it,end;
  it = Wavefront.begin();
  end = Wavefront.end();
  SegmentSpatialObject::PointType p;
  typename OutputImageType::IndexType index;
  //typedef ImageRegionIterator<OutputImageType> OutputIteratorType;
  OutputIteratorType outIt = OutputIteratorType( output, output->GetLargestPossibleRegion() );
  // set new wavefront to segment 1
  while(it != end)
    {
    p = (*it).GetPosition();
    index[0] = p[0];
    index[1] = p[1];
    index[2] = p[2];
    outIt.SetIndex(index);
    outIt.Set(1);
    ++it;
    }
  return output;
}

template< typename TInputImage, typename TOutputImage >
bool
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::BifurcationCheck(SegmentPointerType segment, OutputImagePointer outputImage)
{
  // mark wavefront in output image
  outputImage = this->MarkWavefrontInOutputImage(outputImage, segment->GetWavefront());
  
  // get wavefront and define iterators
  typename SegmentSpatialObject::PointListType wavefront = segment->GetWavefront();
  typename SegmentSpatialObject::PointListType::iterator it,end;
  it = wavefront.begin();
  end = wavefront.end();
  SegmentSpatialObject::PointType p;
  typename OutputImageType::IndexType index_center, index_neighbor;
  
  // define iterators for output image
  typedef NeighborhoodIterator<OutputImageType> NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(3);
  NeighborhoodIteratorType outIt( radius, outputImage, outputImage->GetLargestPossibleRegion() );
  typedef ConstantBoundaryCondition< InputImageType > BoundaryConditionType;
  BoundaryConditionType constantBoundary;
  constantBoundary.SetConstant(0);
  outIt.OverrideBoundaryCondition(&constantBoundary);
  unsigned int count = 0;
  bool flag = true;
  
  
  // vector to store active neighbor values
  std::vector<typename OutputImageType::IndexType> active_points;
  // vector to store modified points
  std::vector<typename OutputImageType::IndexType> modified_points;
  
  // initialize by setting all wavefront neighbors of one wavefront point to 4
  p = (*it).GetPosition();
  index_center[0] = p[0];
  index_center[1] = p[1];
  index_center[2] = p[2];
  outIt.SetLocation(index_center);  
  for(int k=0; k < outIt.Size() ; k++)
    {
    index_neighbor = outIt.GetIndex(k);
    if(outIt.GetPixel(k)==3)
      {
      outIt.SetPixel(k,4);
      active_points.push_back(index_neighbor);
      modified_points.push_back(index_neighbor);
      }
    }
  // loop over active points, growing the region
  typename std::vector<typename OutputImageType::IndexType>::iterator indexIt;
  
  // variable for bifurcation evaluation
  float dif;
  
  while(!active_points.empty())
    {
    indexIt = active_points.begin();  
    index_center = *indexIt;
    outIt.SetLocation(index_center);
    for(int k=0; k < outIt.Size() ; k++)
      {
      index_neighbor = outIt.GetIndex(k);
      if(outIt.GetPixel(k)==3)
        {
        outIt.SetPixel(k,4);
        active_points.push_back(index_neighbor);
        modified_points.push_back(index_neighbor);
        }
      }
    active_points.erase(active_points.begin());
    //std::cout << active_points.size() << std::endl;
    }
  
  
  // second pass to check that all is 4 and return to original values
  it = wavefront.begin();
  while(it!=end)
    {
    p = (*it).GetPosition();
    index_center[0] = p[0];
    index_center[1] = p[1];
    index_center[2] = p[2];
    outIt.SetLocation(index_center);
    if(outIt.GetCenterPixel()==4)
      {
      count++;
      }
    outIt.SetCenterPixel(1); // we set to 1, we do not mark as wavefront (because many wavefronts may collide!)
    ++it;
    }
  // extra pass in case we set to three a voxel that was from another wavefront
  /*it = wavefront.begin();
  while(it!=end)
    {
    p = (*it).GetPosition();
    index_center[0] = p[0];
    index_center[1] = p[1];
    index_center[2] = p[2];
    outIt.SetLocation(index_center);  
    for(int k=0; k < outIt.Size() ; k++)
      {
      index_neighbor = outIt.GetIndex(k);
      if(outIt.GetPixel(k)==3)
        {
        outIt.SetPixel(k,1);
        active_points.push_back(index_neighbor);
        }
      }
    ++it;
    } */
    
  // pass to recover original image values
  typename std::vector<typename OutputImageType::IndexType>::iterator recoverIt, recoverEnd;
  recoverIt = modified_points.begin();
  recoverEnd = modified_points.end();
  while(recoverIt!=recoverEnd)
    {
    outIt.SetLocation(*recoverIt);
    outIt.SetCenterPixel(1);
    ++recoverIt;
    }
  // unmark the wavefront in the output image
  outputImage = this->UnMarkWavefrontInOutputImage(outputImage, wavefront);
  
  //check numbers of connected components and compare with wavefront pixels
  //std::cout << "Count is: " << count << ", wavefront is: " << wavefront.size() << std::endl;
  // we have a certain tolerance for single pixels adding to the segment later. If the bifurcation does
  // not represent more than 20% of the volume of the wavefront, it is not considered
  dif = float(count)/float(wavefront.size());
  if (dif > 0.5 )
    {
    dif = 1-dif;
    }
  
  
  if(count!=wavefront.size() && dif > 0.05)
  {
    //std::cout << "bifurcation!!!!" << std::endl;
    return true;
    }
  else
    {
    return false;
    }
  
}

template< typename TInputImage, typename TOutputImage >
bool
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::StartSegments(SegmentPointerType segment, OutputImagePointer outputImage)
{
  
  // mark wavefront in output image
  outputImage = this->MarkWavefrontInOutputImage(outputImage, segment->GetWavefront());
 
  
  // get wavefront and define iterators
  typename SegmentSpatialObject::PointListType wavefront = segment->GetWavefront();
  typename SegmentSpatialObject::PointListType::iterator it,end;
  it = wavefront.begin();
  end = wavefront.end();
  typename SegmentSpatialObject::PointType p;
  typename OutputImageType::IndexType index_center, index_neighbor;
    
  // define iterators for output image
  typedef NeighborhoodIterator<OutputImageType> NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(3);
  OutputIteratorType outputIt = OutputIteratorType(outputImage, outputImage->GetLargestPossibleRegion());
  NeighborhoodIteratorType outIt( radius, outputImage, outputImage->GetLargestPossibleRegion() );
  typedef ConstantBoundaryCondition< InputImageType > BoundaryConditionType;
  BoundaryConditionType constantBoundary;
  constantBoundary.SetConstant(0);
  outIt.OverrideBoundaryCondition(&constantBoundary);
  
  // value for region growing
  const int init_grow_value = 4;
  int grow_value = init_grow_value;
  
  //vector for region growing active points
  std::vector<typename OutputImageType::IndexType> active_points;
  std::vector<typename OutputImageType::IndexType> modified_points;
  typename std::vector<typename OutputImageType::IndexType>::iterator indexIt;
  
  // initialize list of active points for region growing
  
  
  while(it!=end)
    {
    p = (*it).GetPosition();
    index_center[0] = p[0];
    index_center[1] = p[1];
    index_center[2] = p[2];
    outIt.SetLocation(index_center);
    if(outIt.GetCenterPixel()==3 && grow_value<10)
      {
      // initialize neighborhood to grow_value
      
      for(int k=0; k < outIt.Size() ; k++)
        {
        index_neighbor = outIt.GetIndex(k);
        if(outIt.GetPixel(k)==3)
          {
          outIt.SetPixel(k,grow_value);
          active_points.push_back(index_neighbor);
          modified_points.push_back(index_neighbor);
          }
        }
      // grow region
      while(!active_points.empty())
        {
        indexIt = active_points.begin();  
        index_center = *indexIt;
        outIt.SetLocation(index_center);
        
        for(int k=0; k < outIt.Size() ; k++)
          {
          index_neighbor = outIt.GetIndex(k);
          if(outIt.GetPixel(k)==3)
            {
            outIt.SetPixel(k,grow_value);
            active_points.push_back(index_neighbor);
            modified_points.push_back(index_neighbor);
            }
          }
        
        active_points.erase(active_points.begin());
        } // end grow region loop
      ++grow_value;
      //std::cout << "Grow value: " << grow_value << std::endl;
      } // end if of new region
          
    ++it;
    }
  
  // second pass initializing as many segments as neccesary (generally 2, maybe 3 to 5)
  int num_segments = grow_value - init_grow_value;
  //std::cout << "Number of new segments: " << num_segments << std::endl;
  std::vector<SegmentSpatialObject::PointListType> newWavefronts;
  newWavefronts.resize(num_segments);
  
   //////
  // assign trial points to each new segment
  // create a vector of min-heaps
  MinHeapOfPotentialAndIndex parentTrials = segment->GetTrialPoints();
  std::vector<MinHeapOfPotentialAndIndex> vectorOfHeaps;
  vectorOfHeaps.resize(num_segments);
  
  //loop over parent's trials
  
  typename std::vector<OutputIndexType>::iterator neighIt, neighEnd;
  OutputPixelType pixelValue = 0;
  //std::cout << "Parent trials: " <<parentTrials.size() << std::endl;
  while(!parentTrials.empty())
    {
    PotentialAndIndex trial = parentTrials.top();
    parentTrials.pop();
    index_center[0]= trial.x;
    index_center[1] = trial.y;
    index_center[2] = trial.z;  
    
    // look at six neighbors
    /*typename std::vector<OutputIndexType> neighIndex;
    GetSixNeighbors(index_center, &neighIndex);
    neighIt = neighIndex.begin();
    neighEnd = neighIndex.end();
    while(neighIt!=neighEnd)
      {
      // look at value of image at that point and assing trial point accordingly
      for(int k=0; k<3;k++)
        {
	index_neighbor[k] = (*neighIt)[k];
	}
      outputIt.SetIndex(index_neighbor);
      pixelValue = outputIt.Get();
      if(pixelValue > 3 )
        {
	//std::cout << pixelValue-init_grow_value << std::endl;
        vectorOfHeaps[pixelValue-init_grow_value].push(trial);
	}
      neighIt++;
      }*/
     // look at 21 neighbors
     outIt.SetLocation(index_center);
     int k=0;
     while(k < outIt.Size())
       {
       pixelValue = outIt.GetPixel(k);
       if(pixelValue > 3)  //&& pixelValue<12) -> this check is required if infinite alive values are set to 12
        {
	vectorOfHeaps[pixelValue-init_grow_value].push(trial);
	break;
	}
       k++;
       }
     }
  //look at number of voxels in trials
  
  //for(int i=0; i<num_segments ; ++i)
    //{
    //std::cout << "Son "<< i+1 << ": " << vectorOfHeaps[i].size() << std::endl;
    //}
   
  // we want to see which voxels have been marked
  /*for(int i=0; i<num_segments ; ++i)
    {
    while(!vectorOfHeaps[i].empty())
      {
      PotentialAndIndex trial=vectorOfHeaps[i].top();
      vectorOfHeaps[i].pop();
      index_neighbor[0] = trial.x;
      index_neighbor[1] = trial.y;
      index_neighbor[2] = trial.z;
      outputIt.SetIndex(index_neighbor);
      outputIt.Set(6+i);
      }
    }*/
  ////
  
  
  
  it = wavefront.begin();
  while(it!=end)
    {
    //p.SetPosition((*it).GetPosition());
     p = (*it).GetPosition();
     index_center[0] = p[0];
     index_center[1] = p[1];
     index_center[2] = p[2];
     outIt.SetLocation(index_center);
     if(outIt.GetCenterPixel() > init_grow_value-1)
       {
       newWavefronts[outIt.GetCenterPixel()-init_grow_value].push_back(*it);
       }
       outIt.SetCenterPixel(1); // set back to segment value
       ++it;
     }
  
 
  
  for(int i=0; i < num_segments ; ++i)
    {
    if(newWavefronts[i].size()>wavefront.size()*0.05)
      {
      typename SegmentSpatialObject::Pointer newSegment = SegmentSpatialObject::New();
      newSegment->SetTrialPoints(&vectorOfHeaps[i]);
      newSegment->SetWavefront(newWavefronts[i]);
      newSegment->SetInitialWavefront(newWavefronts[i]);
      newSegment->SetPoints(newWavefronts[i]);
      newSegment->ComputeNewCenterlinePoint();
      newSegment->UpdateWavefrontGrowthVector(newWavefronts[i].size());
      //std::cout << "Add as son" << std::endl;
      segment->AddSpatialObject(newSegment); // here, we set as child of parent segment. It will be removed if rejected later.
      m_SegmentQueue.push_back(newSegment);
      }
    }
  // pass to recover original image values
  /*typename std::vector<typename OutputImageType::IndexType>::iterator recoverIt, recoverEnd;
  recoverIt = modified_points.begin();
  recoverEnd = modified_points.end();
  while(recoverIt!=recoverEnd)
    {
    outIt.SetLocation(*recoverIt);
    outIt.SetCenterPixel(1);
    ++recoverIt;
    }*/
    
  // unmark the wavefront in the output image
  outputImage = this->UnMarkWavefrontInOutputImage(outputImage, wavefront);
  return true;
}

template< typename TInputImage, typename TOutputImage >
bool
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::EvaluatePropagation(SegmentPointerType segment, OutputImagePointer outputImage, OutputImagePointer segmentImage)
{
  // in this version for binary trees, all propagations are assummed to be correct  
  return true;
}

template< typename TInputImage, typename TOutputImage >
bool
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::FindSegmentInList(std::vector<SegmentPointerType> segmentList, int segmentId)
{
  std::vector<SegmentSpatialObject::Pointer>::iterator itSegments, endSegments;
  itSegments = segmentList.begin();
  endSegments = segmentList.end();
  while(itSegments!=endSegments)
    {
    if((*itSegments)->GetId()==segmentId)
      {
      return true;
      }
    ++itSegments;
    }
  return false;
}

template< typename TInputImage, typename TOutputImage >
int
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::IsInAcceptedSegments(OutputIndexType index)
{
  std::vector<SegmentSpatialObject::Pointer>::iterator itSegments, endSegments;
  PointListType it, end;
  itSegments = m_AcceptedSegments.begin();
  endSegments = m_AcceptedSegments.begin();
  SegmentSpatialObject::PointType p;
  p[0] = index[0];
  p[1] = index[1];
  p[2] = index[2];
  while(itSegments!=endSegments)
    {
    if((*itSegments)->PointIndexIsInside(p))
      {
      return (*itSegments)->GetId();
      }
    ++itSegments;
    }
  return 0;  
}

template< typename TInputImage, typename TOutputImage >
float
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::ComputeCompactness(SegmentPointerType segment, OutputImagePointer outputImage)
{
  float compactness = 0;
  // mark segment in output image
  typename SegmentSpatialObject::PointListType points = segment->GetPoints();
  typename SegmentSpatialObject::PointListType::iterator it, end;
  it = points.begin();
  end = points.end();
  OutputIteratorType outIt = OutputIteratorType(outputImage, outputImage->GetLargestPossibleRegion());
  OutputIndexType index;
  
  while(it!=end)
    {
    index[0] = (*it).GetPosition()[0];
    index[1] = (*it).GetPosition()[1];
    index[2] = (*it).GetPosition()[2];
    outIt.SetIndex(index);
    outIt.Set(3);
    ++it;
    }
  
  // compute number of voxels that are in contact with outside
  // connectivity six
  typedef ConstShapedNeighborhoodIterator< OutputImageType> ShapedNeighborhoodIteratorType;
  typename ShapedNeighborhoodIteratorType::OffsetType offset1 = {{1,0,0}};;
  typename ShapedNeighborhoodIteratorType::OffsetType offset2 = {{-1,0,0}};;
  typename ShapedNeighborhoodIteratorType::OffsetType offset3 = {{0,1,0}};;
  typename ShapedNeighborhoodIteratorType::OffsetType offset4 = {{0,-1,0}};;
  typename ShapedNeighborhoodIteratorType::OffsetType offset5 = {{0,0,1}};;
  typename ShapedNeighborhoodIteratorType::OffsetType offset6 = {{0,0,-1}};;
  typename ShapedNeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);
  ShapedNeighborhoodIteratorType shapedIt( radius, outputImage, outputImage->GetLargestPossibleRegion());
  shapedIt.ActivateOffset(offset1);
  shapedIt.ActivateOffset(offset2);
  shapedIt.ActivateOffset(offset3);
  shapedIt.ActivateOffset(offset4);
  shapedIt.ActivateOffset(offset5);
  shapedIt.ActivateOffset(offset6);
   
  int A =0;
  bool border = false;
  it = points.begin();
  while(it!=end)
    {
    index[0] = (*it).GetPosition()[0];
    index[1] = (*it).GetPosition()[1];
    index[2] = (*it).GetPosition()[2];
    shapedIt.SetLocation(index);
    typename ShapedNeighborhoodIteratorType::ConstIterator ci;
    border = false;
    for(ci = shapedIt.Begin(); ci != shapedIt.End(); ci++)
      {
      if(ci.Get()!=3)
        {
        //border = true;
        A++;
        }
      }
    
    ++it;
    }
  
  float n = points.size();
  compactness = (n - A/6.0)/(n-pow(double(n),2.0/3.0));
  
  // remove segment mark from output image
  it = points.begin();
  while(it!=end)
    {
    index[0] = (*it).GetPosition()[0];
    index[1] = (*it).GetPosition()[1];
    index[2] = (*it).GetPosition()[2];
    outIt.SetIndex(index);
    outIt.Set(1);
    ++it;
    }
  return compactness;
}

template< typename TInputImage, typename TOutputImage >
bool
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::EvaluateSegmentAndPossiblyAddToTree(SegmentPointerType segment, OutputImagePointer outputImage, OutputImagePointer segmentImage)
{
  // inthis version for binary trees, we assume all
  // segments are correct
  
  //std::cout << "Accesing segment" << std::endl;
  m_AirwayNumber++;
  segment->SetId(m_AirwayNumber);
  if(m_AirwayNumber==2)
    {
    
    m_AirwayTree->SetRoot(segment.GetPointer());
    m_AcceptedSegments.push_back(segment);
    //std::cout << "Root set" << std::endl;
    }
    
    
  bool accept = true;
  
  segment->ComputeMeanRadius();
  m_AcceptedSegments.push_back(segment); // if accepted 
  
  // add to the segment image (if accepted)
  PointListType points = segment->GetPoints();
  typename PointListType::iterator it, end;
  typename SegmentSpatialObject::PointType p;
  OutputIteratorType segmentIt = OutputIteratorType(segmentImage, segmentImage->GetLargestPossibleRegion());
  it = points.begin();
  end = points.end();
  OutputIndexType index;
  while(it!=end)
    {
    p = (*it).GetPosition();
    index[0] = p[0];
    index[1] = p[1];
    index[2] = p[2];
    segmentIt.SetIndex(index);
    segmentIt.Set(m_AirwayNumber);
    ++it;
    } 
  
  this->WriteSegmentData(segment, 0);
  return true;
}

template< typename TInputImage, typename TOutputImage >
void
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::GetSixNeighbors(OutputIndexType index, typename std::vector<OutputIndexType>* neighborVector)
{
  // modifies the neighbor vector adding the 6 neighbors
  // if any of them is outside the image, it does not add it
  OutputIndexType newIndex;
  // get input image pointer to get size
  InputImageConstPointer input = this->GetInput();
  typename InputImageType::RegionType::SizeType imageSize = input->GetLargestPossibleRegion().GetSize();
  
  // clear vector to add new elements
  neighborVector->clear();
  
  int k = 0;
  bool inside = true;
  for(int i = 0; i < 6; i++)
    {
    inside = true;
    for(k = 0; k < 3; k++)
      {
      newIndex[k] = index[k]-m_SixOffsets[i][k];
      //check that new index is inside bounds
      if( newIndex[k] < 0 || newIndex[k] >= imageSize[k])
        {
	inside = false;
	}
      }
     if(inside)
       {
       neighborVector->push_back(newIndex);
       }
    }
}


template< typename TInputImage, typename TOutputImage >
void
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::SetTextFileName(char * Name)
{
  m_TextFileName = Name;
}

template< typename TInputImage, typename TOutputImage >
void
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::WriteSegmentData(SegmentPointerType segment, float compactness)
{
  char* filename = this->GetTextFileName();
  std::ofstream textFile;
  // we open text file with flags to append data
  textFile.open(filename, std::ios::out | std::ios::app);
  textFile << "Segment number: " << m_AirwayNumber;
  if(segment->GetLength()==0)
    {
    textFile << "\nLength: " << 1;
    }
  else
    {
    textFile << "\nLength: " << segment->GetLength();
    }
  textFile << "\nMean Radius: " << segment->GetMeanRadius();
  textFile << "\nVolume: " << segment->GetPoints().size();
  textFile << "\nNumber of children: " << segment->GetNumberOfChildren(1);
  if(segment->HasParent())
    {
    textFile << "\nParent number: " << segment->GetParent()->GetId();
    }
  else
    {
    textFile << "\nParent number: " << "no parent";
    }
  textFile << "\nGrowth rate: " << segment->GetGrowthRate();
  textFile << "\nCompactness: " << compactness;
  textFile << "\n\n";
  textFile.close();
  
}

template< typename TInputImage, typename TOutputImage >
void
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::GenerateData()
{

  // get input image
  InputImageConstPointer input = this->GetInput();
  
  //allocate memory for output image and define iterator
  OutputImagePointer output = this->GetOutput();
  output->SetRegions( input->GetLargestPossibleRegion() );
  output->Allocate();
  output->FillBuffer(0);
  typedef ImageRegionIterator<OutputImageType> OutputIteratorType;
  OutputIteratorType outIt = OutputIteratorType( output, output->GetLargestPossibleRegion() );
  
  
  // allocate memory for auxiliary image that saves segment information
  OutputImagePointer segmentImage = OutputImageType::New();
  segmentImage->SetRegions( input->GetLargestPossibleRegion() );
  segmentImage->Allocate();
  segmentImage->FillBuffer(0);
  OutputIteratorType segmentIt= OutputIteratorType( segmentImage, segmentImage->GetLargestPossibleRegion() );
  
  // allocate memory for auxiliary image that saves potential information
  PotentialImagePointer potentialImage = PotentialImageType::New();
  potentialImage->SetRegions( input->GetLargestPossibleRegion() );
  potentialImage->Allocate();
  potentialImage->FillBuffer(INFINITY);
  PotentialImageIteratorType potIt = PotentialImageIteratorType( potentialImage, potentialImage->GetLargestPossibleRegion() );
     
  // initialize first segment with first seed points as 
  // points and as wavefront
  typedef ImageRegionConstIterator<TInputImage> ConstIteratorType;
  ConstIteratorType inIt = ConstIteratorType(input, input->GetLargestPossibleRegion());
  inIt.GoToEnd();outIt.GoToEnd();
  --inIt;--outIt;
  
  typename SegmentSpatialObject::Pointer segment = SegmentSpatialObject::New();
  typename SegmentSpatialObject::PointListType wavefront, prevWavefront;
  typename SegmentSpatialObject::PointListType centerline;
  typedef SegmentSpatialObject::BlobPointType BlobPointType;
  
  // compute center of gravity of seed points
  typename std::vector<typename TInputImage::IndexType>::iterator seedIt, seedEnd;
  seedIt = m_Seeds.begin();
  seedEnd = m_Seeds.end();
  float xCenter = 0, yCenter = 0, zCenter = 0, cnt = 0;
  while(seedIt!=seedEnd)
    {
    xCenter = xCenter + (*seedIt)[0];
    yCenter = yCenter + (*seedIt)[1];
    zCenter = zCenter + (*seedIt)[2];
    cnt++;
    ++seedIt;
    }
  xCenter = round(xCenter/cnt);
  yCenter = round(yCenter/cnt);
  zCenter = round(zCenter/cnt);
  
  // copy centerpoint to wavefront
  BlobPointType p;
  p.SetPosition(xCenter, yCenter, zCenter);
  wavefront.push_back(p);

  /** initialize fast marching algorithm*/
  // add the initial point to the segment, 
  // mark point in segment image
  segment->SetWavefront(wavefront);
  segment->SetInitialWavefront(wavefront);
  segment->SetPoints(wavefront);
  segment->UpdateWavefrontGrowthVector(wavefront.size());
  segment->ComputeNewCenterlinePoint();
  OutputIndexType index;
  PotentialIndexType potIndex;
  for(int i = 0; i < 3; i++)
    {
    index[i] = p.GetPosition()[i];
    potIndex[i] = p.GetPosition()[i];
    }
  outIt.SetIndex(index);
  outIt.Set(1);
  potIt.SetIndex(potIndex);
  potIt.Set(0);
  //std::cout << "Initial point: " << index << std::endl;
  // add neighbors to the trial min-heap, with potential=1
  // also mark in output image as 2
  PotentialAndIndex trial;
  typename std::vector<OutputIndexType> neighborIndex;
  MinHeapOfPotentialAndIndex trialMinHeap;
  GetSixNeighbors(index, &neighborIndex);
  
  typename std::vector<OutputIndexType>::iterator neighIt, neighEnd;
  neighIt = neighborIndex.begin();
  neighEnd = neighborIndex.end();
  trial.Potential = 1;
  while(neighIt!=neighEnd)
    {
    trial.x = (*neighIt)[0];
    trial.y = (*neighIt)[1];
    trial.z = (*neighIt)[2];
    trialMinHeap.push(trial);
    for(int i = 0; i < 3; i++)
    {
    index[i] = (*neighIt)[i];
    potIndex[i] = (*neighIt)[i];
    }
    potIt.SetIndex(potIndex);
    potIt.Set(1);
    outIt.SetIndex(index);
    outIt.Set(2);
    ++neighIt;
    }
  //std::cout << "Setting trial points" << std::endl;
  segment->SetTrialPoints( &trialMinHeap );
  //std::cout << "Trial points set" << std::endl;
  // the rest are far, no need to do anything
  
  
  // print min-heap to check that it is correct
  /*while(!m_TrialMinHeap.empty())
    {
    trial = m_TrialMinHeap.top();
    std::cout << trial.x << ", " << trial.y << ", " << trial.z << ": " << trial.Potential << std::endl;
    m_TrialMinHeap.pop();
    }*/
  
  //  add segment to queue
  m_SegmentQueue.push_back(segment);
  
  // iterate over segment queue
  typename SegmentSpatialObject::Pointer segmentProp;
  
  const int number_of_initial_props = 20;
  
  bool propagate=true;
  bool stop = true;

  int frontSize= 0;
 
  float timeStep = m_TimeStep;

  int count = 0, cntBif=0;
  
  // first few propagations without propagation evaluation
  while(count<10 && !m_SegmentQueue.empty())
   {
    // get segment from queue
    segment = m_SegmentQueue.front();
       
    propagate = true;
    count=0;
    while(propagate==true)
      {
      count++;
      if(count > number_of_initial_props)
        {
	propagate = false;	
	}
      prevWavefront = segment->GetWavefront();
      
      // call Fast Marching propagation method
      segment = PropagateWavefrontFixedThreshold(segment, input, output, segmentImage, potentialImage, timeStep);
      wavefront = segment->GetWavefront();
      frontSize = wavefront.size();
      //std::cout << "Wavefront size: " << frontSize << std::endl;
      
      
      // check if wavefront stopped
      if(segment->GetWavefront().size()==0 || (EqualWavefronts(segment->GetWavefront(),prevWavefront)))
       {
       //std::cout << "m_SegmentQueue.size:" << m_SegmentQueue.size() << std::endl;
       //std::cout << "Wavefront stopped" << std::endl;
       EvaluateSegmentAndPossiblyAddToTree(segment,output,segmentImage);
       m_SegmentQueue.erase(m_SegmentQueue.begin());
       propagate = false;        
       }
      
      
      // evaluate if there is bifurcation
      else if(this->BifurcationCheck(segment, output))
        {
	//std::cout << "Bifurcation detected" << std::endl;
        cntBif++;	   
         
	// if there is bifurcation: evaluate parent segment. If it is accepted, start new ones.
        if(EvaluateSegmentAndPossiblyAddToTree(segment,output, segmentImage))
          {
          StartSegments(segment, output);
          //std::cout << "After starting new segments" << std::endl;
          }
             
        m_SegmentQueue.erase(m_SegmentQueue.begin()); // it is removed from queue
        propagate = false;        
        }              
      }     
    }
  
  
  // normal growth once the segment radius is stable
  //while(cntBif<10)
  while(!m_SegmentQueue.empty())
    {
    // get segment from queue
    
    segment = m_SegmentQueue.front();
                
    propagate = true;
    count=0;
    while(propagate==true)
      {
      count++;
      if(count>300) // to avoid incredibly long propagations
        {
	propagate = false;
	m_SegmentQueue.erase(m_SegmentQueue.begin());
	}
      prevWavefront = segment->GetWavefront();
      //std::cout << "Prev. wavefront size: " << prevWavefront.size() << std::endl;
      
      // call Fast Marching propagation method with variabke threhold
      segment = PropagateWavefrontVariableThreshold(segment, input, output, segmentImage, potentialImage, timeStep);
      wavefront = segment->GetWavefront();
      frontSize = wavefront.size();
      //std::cout << "Wavefront size: " << frontSize << std::endl;
      
      
      // check if wavefront stopped
      if(segment->GetWavefront().size()==0 || (EqualWavefronts(segment->GetWavefront(),prevWavefront)))
        {
        //std::cout << "m_SegmentQueue.size:" << m_SegmentQueue.size() << std::endl;
	//std::cout << "Wavefront stopped" << std::endl;
        EvaluateSegmentAndPossiblyAddToTree(segment,output,segmentImage);
        m_SegmentQueue.erase(m_SegmentQueue.begin());
        propagate = false;        
        }
      
      
      // evaluate if there is bifurcation
      else if(this->BifurcationCheck(segment, output))
        {
	//std::cout << "Bifurcation detected" << std::endl;
        cntBif++;
	// if there is bifurcation: evaluate parent segment. If it is accepted, start new ones.
        if(EvaluateSegmentAndPossiblyAddToTree(segment,output, segmentImage))
          {
          //std::cout << "Starting segments" << std::endl;
	  StartSegments(segment, output);
          //std::cout << "After starting new segments" << std::endl;
          }
          m_SegmentQueue.erase(m_SegmentQueue.begin()); // it is removed from queue
          propagate = false;
          // evaluate segment before accepting, then start new segments (only if parent accepted)
        }
        
   
      // evaluate propagation
      else if(!this->EvaluatePropagation(segment,output,segmentImage))
        {
        //std::cout << "Invalid propagation " << std::endl;
        EvaluateSegmentAndPossiblyAddToTree(segment, output,segmentImage);
        m_SegmentQueue.erase(m_SegmentQueue.begin());
        propagate = false;
        }
      /*else if(segment->SegmentIsFullyGrown(m_Threshold_lr))
        {
        //std::cout << "Segment fully grown" << std::endl;
        if(EvaluateSegmentAndPossiblyAddToTree(segment, output,segmentImage))
          {
          // if accepted, create a new segment that grows directly afterwards
          SegmentSpatialObject::Pointer newSegment = SegmentSpatialObject::New();
          PointListType lastWavefront = segment->GetWavefront();
          newSegment->SetWavefront(lastWavefront);
  	  newSegment->SetInitialWavefront(lastWavefront);
  	  newSegment->SetPoints(lastWavefront);
          newSegment->UpdateWavefrontGrowthVector(lastWavefront.size());
          newSegment->ComputeNewCenterlinePoint();
          segment->AddSpatialObject(newSegment);
          m_SegmentQueue.erase(m_SegmentQueue.begin());
          propagate = false;
          // this is exceptional: we put this segment in the front of the queue, 
          //because they are direct continuation of the previous segment
          m_SegmentQueue.insert(m_SegmentQueue.begin(),newSegment);
          }
        else
          {
          m_SegmentQueue.erase(m_SegmentQueue.begin());
          }
        }*/
      
      /*if(m_AirwayNumber>=7)
        {
	std::cout << "Airway number: "<< m_AirwayNumber  << std::endl;
	int loko;
	std::cin >> loko;
	}*/
	
      if(propagate==true)
        {
        segment->ComputeRadiusAndCheckIfMinimum();
        }      
      }   
    }
    
  // copy segments to output image
  
  /*SegmentSpatialObject::PointType point;
  for(outIt.GoToBegin(); !outIt.IsAtEnd(); ++outIt)
    {
    point[0] = outIt.GetIndex()[0];
    point[1] = outIt.GetIndex()[1];
    point[2] = outIt.GetIndex()[2];
    if(segment->IsInside(point))
      {
      outIt.set(m_AirwayNumber);
      }
    }*/
  this->GraftOutput(segmentImage);
  
  // cast image to see potential image
  /*typedef CastImageFilter<PotentialImageType, OutputImageType> CasterType;
  typename CasterType::Pointer caster = CasterType::New();
  caster->SetInput(potentialImage);
  caster->Update();
  this->GraftOutput(caster->GetOutput());*/
}

template< typename TInputImage, typename TOutputImage >
bool
BinaryTreeSegmentationImageFilter< TInputImage, TOutputImage >
::EqualWavefronts(SegmentSpatialObject::PointListType front1, SegmentSpatialObject::PointListType front2)
{
  if(front1.size()!=front2.size())
    {
    return false;
    }
  SegmentSpatialObject::PointListType::iterator front1It, front1End, front2It, front2End;
  front1It = front1.begin();
  front2It = front2.begin();
  front1End = front1.end();
  front2End = front2.end();
  while(front1It != front1End)
    {
    if((*front1It).GetPosition()[0]!=(*front2It).GetPosition()[0] || (*front1It).GetPosition()[1]!=(*front2It).GetPosition()[1] || (*front1It).GetPosition()[2]!=(*front2It).GetPosition()[2])
      {
      return false;
      }
    ++front1It;
    ++front2It;
    }
  return true;
}


}// end namespace itk

#endif
