#ifndef __itkRegionFilter_txx
#define __itkRegionFilter_txx

#include "vnl/vnl_vector_fixed.h"
#include "itkProgressReporter.h"
#include "itkSize.h"
#include "itkImageRegion.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionReverseIterator.h"
#include "itkRegionFilter.h"
#include "itkImageSliceConstIteratorWithIndex.h"

namespace itk
{

template< class TInputImage, class TOutputImage >
RegionFilter< TInputImage, TOutputImage >
::RegionFilter()
{
  itkDebugMacro(<< "RegionFilter::RegionFilter() called");
	IndexType indice;
  indice[0]=0;
  indice[1]=0;
  indice[2]=0;
  m_Index=indice;
  SizeType tamanio;
  tamanio[0]=0;
  tamanio[1]=0;
  tamanio[2]=0;
  m_Size=tamanio;
  m_Min=2;
}

template< class TInputImage, class TOutputImage >
void
RegionFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion()
{
  
}


template< class TInputImage, class TOutputImage >
void
RegionFilter< TInputImage, TOutputImage >
::GenerateData()
{
  itkDebugMacro(<< "RegionFilter::GenerateData() called");

  // Get the input and output pointers
  InputImageConstPointer  inputPtr  = this->GetInput(0);
  OutputImagePointer      outputPtr = this->GetOutput(0);

  // How big is the input image?
  typename TInputImage::SizeType size=inputPtr->GetRequestedRegion().GetSize();
  typename TInputImage::IndexType startIndex=inputPtr->GetRequestedRegion().GetIndex();
  int dimension=inputPtr->GetRequestedRegion().GetImageDimension();
   
 	typedef ImageSliceConstIteratorWithIndex< TOutputImage > SliceConstIteratorType;
  SliceConstIteratorType it( inputPtr, inputPtr->GetLargestPossibleRegion() );
  bool salir=0;
  int count=0;
  typedef ImageRegionConstIterator<TOutputImage>  InputIterator;
  typedef ImageRegionIterator<TOutputImage>       OutputIterator;
  
  for (int i=0;i<3;i++)
    {
    it.SetFirstDirection((i+1)%3);
    it.SetSecondDirection((i+2)%3);
    it.GoToBegin();
    while (!it.IsAtEnd()&&!salir)
		  {
		  while (!it.IsAtEndOfSlice())
		  	{
		  	while (!it.IsAtEndOfLine())
		  		{
			 	  if (it.Get()==1) 
				    {
  				  //std::cout << "Punto " << i+1 << "min: " << it.GetIndex() << std::endl;
	  			  count++;
	   			  }
	   			++it;
	   			}
			 it.NextLine();
			 }
  		if (count<m_Min)
  		  {
  		  count=0;
  		  it.NextSlice();
  		  }
  		else
  		  {
  		  count=0;
  		  salir=1;
  		  }
	   	}
    if (it.GetIndex()[i]>0) m_Index[i]=it.GetIndex()[i];
    else m_Index[i]=0;
    salir=0;
  
    it.GoToReverseBegin();
    while (!it.IsAtReverseEnd()&&!salir)
  	  {
  	  while (!it.IsAtReverseEndOfSlice())
			  {
			  while (!it.IsAtReverseEndOfLine())
				  {
				  if (it.Get()==1) 
				    {
				    count++;
				    //std::cout << "Punto " << i+1 << "max: " << it.GetIndex() << std::endl;
  				  }
  				  --it;
				  }
			  it.PreviousLine();
			  }
		  if(count<m_Min)
		    {
		    count=0;
		    it.PreviousSlice();
		    }
		  else
		    {
		    count=0;
		    salir=1;
		    }
		  }
    if (it.GetIndex()[i]>size[i]) m_Size[i]=size[i]-m_Index[i];
    else m_Size[i]=it.GetIndex()[i]-m_Index[i]+1;
    salir=0;
  }
  
  /*std::cout << "Indice: " << m_Index << ", Tamaño: " << m_Size;
  std::cout << ". Reducido el volumen al: " << 100*m_Size[0]/size[0]*m_Size[1]/size[1]*m_Size[2]/size[2] << "%" << std::endl;*/
  
  OutputImageRegionType Region;
  Region.SetSize(m_Size);
  Region.SetIndex(m_Index);
  InputIterator in = InputIterator(inputPtr, Region);

  OutputImageRegionType Region2;
  Region2.SetSize(m_Size);
  IndexType indice;
  indice[0]=0;
  indice[1]=0;
  indice[2]=0;
  Region2.SetIndex(indice);
  outputPtr->SetRegions( Region2 );
  outputPtr->Allocate();
  OutputIterator out = OutputIterator(outputPtr, Region2);
  
  for ( in.GoToBegin(), out.GoToBegin(); !in.IsAtEnd(), !out.IsAtEnd(); ++in, ++out )
    out.Set(in.Get());  
}

template< class TInputImage, class TOutputImage >
void
RegionFilter< TInputImage, TOutputImage >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  
  os << indent << std::endl;
  
}

} // end namespace

#endif
