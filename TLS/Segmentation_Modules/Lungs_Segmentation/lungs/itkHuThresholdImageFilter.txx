#ifndef __itkHuThresholdImageFilter_txx
#define __itkHuThresholdImageFilter_txx

#include "vnl/vnl_vector_fixed.h"
#include "itkProgressReporter.h"
#include "itkSize.h"
#include "itkImageRegion.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionReverseIterator.h"
#include "itkHuThresholdImageFilter.h"

namespace itk
{

template< class TInputImage, class TOutputImage >
HuThresholdImageFilter< TInputImage, TOutputImage >
::HuThresholdImageFilter()
{
  itkDebugMacro(<< "HuThresholdImageFilter::HuThresholdImageFilter() called");

  m_Diff = 0.1;
  m_PlusTh = 0.0;
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
}

template< class TInputImage, class TOutputImage >
void
HuThresholdImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion()
{
  
}


template< class TInputImage, class TOutputImage >
void
HuThresholdImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
  itkDebugMacro(<< "HuThresholdImageFilter::GenerateData() called");

  // Get the input and output pointers
  InputImageConstPointer  inputPtr  = this->GetInput(0);
  OutputImagePointer      outputPtr = this->GetOutput(0);

  // How big is the input image?
  typename TInputImage::SizeType size=inputPtr->GetRequestedRegion().GetSize();
  typename TInputImage::IndexType startIndex=inputPtr->GetRequestedRegion().GetIndex();
  //int dimension=inputPtr->GetRequestedRegion().GetImageDimension();
  
  // Iterator Typedefs for this routine
  typedef ImageRegionConstIterator<TInputImage>       InputIterator;
  typedef ImageRegionIterator<TOutputImage>           OutputIterator;

  OutputImageRegionType Region;
  Region.SetSize(size);
  Region.SetIndex(startIndex);
  typename TOutputImage::Pointer imagen           =  TOutputImage::New();
  imagen->SetRegions( Region );
  imagen->Allocate();
  
 
  InputIterator  in  = InputIterator (inputPtr, inputPtr->GetRequestedRegion());
  
  OutputIterator it =  OutputIterator(imagen, Region);
  double  Ti=m_Ti;
  
  //Bucle para buscar el mejor Ti
  double ub, un, Nb, Nn, Tii, i;
  Tii = -600;
  i=1;
  //Ti=33000;
  std::cout << "Initial Threshold: " << Ti << std::endl;
  //Creamos la mascara
  for ( in.GoToBegin(), it.GoToBegin(); !in.IsAtEnd(), !it.IsAtEnd(); ++in, ++it )
    it.Set(in.Get()>=Ti);
  
  while (fabs(Tii-Ti)>m_Diff){
    for ( in.GoToBegin(), it.GoToBegin(); !in.IsAtEnd(); ++in, ++it ){
      if (it.Get()){
        ub=ub+in.Get();
        Nb++;
      }
      else{
        un=un+in.Get();
        Nn++;
      }
    }
    if (Nb!=0) ub=ub/Nb;
    else ub=0;
    if (Nn!=0) un=un/Nn;
    else un=0;
    Ti=Tii;
    std::cout << "ub" << ub << std::endl;
    std::cout << "un" << un << std::endl;
    Tii=(ub+un)/2;
    ub=0;un=0;
    Nn=0;Nb=0;
    i++;
    std::cout << "Iteration: " << i << std::endl;
    std::cout << "Threshold: " << Tii << std::endl;
    std::cout << "" << ub << std::endl;
    for ( in.GoToBegin(), it.GoToBegin(); !in.IsAtEnd(); ++in, ++it )
      //it.Set( in.Get()>=Tii );
      //test
      it.Set( in.Get()<=Tii );
      
  	}
  //std::cout << "Alcanzado el optimo en " << i << " intentos."<< std::endl;
  std::cout << "Opt threshold: " << Tii << std::endl;
  
  m_Threshold = Tii;
  for ( in.GoToBegin(), it.GoToBegin(); !in.IsAtEnd(); ++in, ++it )
        it.Set( in.Get()<= Tii + m_PlusTh  );
  std::cout << "Final th: " << Tii + m_PlusTh << std::endl;
  this->GraftOutput(imagen);

}

template< class TInputImage, class TOutputImage >
void
HuThresholdImageFilter< TInputImage, TOutputImage >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  
  os << indent << "Maxima diferencia de: "<< this->m_Diff <<std::endl;
  
}

} // end namespace

#endif
