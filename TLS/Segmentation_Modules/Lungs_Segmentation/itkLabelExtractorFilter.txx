#ifndef __itkLabelExtractorFilter_txx
#define __itkLabelExtractorFilter_txx

#include "vnl/vnl_vector_fixed.h"
#include "itkProgressReporter.h"
#include "itkImageRegion.h"
#include "itkLabelExtractorFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkImageLinearIteratorWithIndex.h"


#include "itkShapeLabelObject.h"
#include "itkLabelMap.h"
#include "itkBinaryImageToLabelMapFilter.h"
#include "itkShapeLabelMapFilter.h"
#include "itkLabelMapToLabelImageFilter.h"


#include "itkImageFileWriter.h"






namespace itk
{


template< class TInputImage, class TOutputImage > LabelExtractorFilter< TInputImage, TOutputImage >::LabelExtractorFilter(){
  itkDebugMacro(<< "called");
  m_Inc=1;
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

template< class TInputImage, class TOutputImage > void LabelExtractorFilter< TInputImage, TOutputImage >::GenerateInputRequestedRegion(){}

const unsigned int Dimension=2;
typedef unsigned long LabelType;
typedef itk::ShapeLabelObject< LabelType, Dimension > LabelObjectType;
typedef itk::LabelMap< LabelObjectType > LabelCollectionType;


template< class TInputImage, class TOutputImage > void LabelExtractorFilter< TInputImage, TOutputImage >::GenerateData() {
 
  // Get the input and output pointers
  InputImageConstPointer  inputPtr  = this->GetInput(0);
  OutputImagePointer      outputPtr = this->GetOutput(0);
  
  // How big is the input image?
  typename TInputImage::SizeType size=inputPtr->GetRequestedRegion().GetSize();
  typename TInputImage::IndexType startIndex=inputPtr->GetRequestedRegion().GetIndex();
  int dimension=inputPtr->GetRequestedRegion().GetImageDimension();
  
  // Allocate the output
  OutputImageRegionType Region;
  Region.SetSize(size);
  Region.SetIndex(startIndex);
  outputPtr->SetRegions( Region );
  outputPtr->Allocate();

  typedef ImageSliceIteratorWithIndex< TOutputImage >  SliceIteratorType;
  SliceIteratorType     out( outputPtr, outputPtr->GetLargestPossibleRegion() );
  out.SetFirstDirection(0);out.SetSecondDirection(1);


  typedef ImageSliceConstIteratorWithIndex< TOutputImage >  SliceConstIteratorType;
  SliceConstIteratorType in(  inputPtr,  inputPtr ->GetLargestPossibleRegion() );
  in.SetFirstDirection(0); in.SetSecondDirection(1);
  
  
  //Creamos la imagen 2D
  typedef itk::Image< PixelType, Dimension > ImageType2D;
  typename ImageType2D::Pointer imagen =  ImageType2D::New();

  typename ImageType2D::IndexType Indice;
  Indice[0]=0;
  Indice[1]=0;
  
  typename ImageType2D::SizeType Size;
  Size[0]=size[0];
  Size[1]=size[1];
    
  typedef typename ImageType2D::RegionType                                OutputImageRegion2DType;
  OutputImageRegion2DType Region2D;
  Region2D.SetSize(Size);
  Region2D.SetIndex(Indice);
  
  imagen->SetRegions(Region2D);
  imagen->Allocate();
  
  typename ImageType2D::Pointer imagen2 =                                 ImageType2D::New();
  imagen2->SetRegions(Region2D);

  typedef ImageRegionIterator<ImageType2D>                                OutputIterator;
  OutputIterator imageIt = OutputIterator(imagen, Region2D);
 
  std::cout << "Starting label extraction filter " << std::endl;
  
  typename ImageType2D::IndexType tCentroid;
  OutputImageRegion2DType tRegion;

  //Start from the begin
  in.GoToBegin();
  out.GoToBegin();

  while (!in.IsAtEnd()&&!out.IsAtEnd()) {
	  
	  imageIt.GoToBegin();

	  //Copy the current slice to imagen
	  while (!in.IsAtEndOfSlice()&&!imageIt.IsAtEnd()) {
		  while (!in.IsAtEndOfLine()){
			  imageIt.Set(in.Get());
    			  ++imageIt;
			  ++in;
    		}
		  in.NextLine();
	  }

   
    
	  typedef itk::BinaryImageToLabelMapFilter< ImageType2D, LabelCollectionType > ConverterType;
	  typename ConverterType::Pointer converter = ConverterType::New();
	  converter->SetInput( imagen);
    converter->SetInputForegroundValue( 1 );
    //Any nonzero value should be read as True
    converter->SetFullyConnected(1);
    
    typedef itk::ShapeLabelMapFilter< LabelCollectionType > ShapeFilterType;
	  typename ShapeFilterType::Pointer shape = ShapeFilterType::New();
	  shape->SetInput( converter->GetOutput() );
    shape->Update();

    typedef itk::LabelMapToLabelImageFilter< LabelCollectionType, ImageType2D > MapperType;
	  typename MapperType::Pointer mapper = MapperType::New();
	  mapper->SetInput(shape->GetOutput());
	  mapper->Update();
	  
    //Now we can copy back the modified slice to the output volume
	  OutputIterator it2 = OutputIterator(mapper->GetOutput(), Region2D);
	  it2.GoToBegin();
	  while (!out.IsAtEndOfSlice()&&!it2.IsAtEnd()) {
		  while (!out.IsAtEndOfLine()){
  			out.Set(it2.Get());
	    	++it2;
			  ++out;
	    }
		  out.NextLine();
	  }
	  //}
	  
	  
	  out.NextSlice();
	  in.NextSlice();  	
  }

 }

template< class TInputImage, class TOutputImage > void LabelExtractorFilter< TInputImage, TOutputImage >::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  
  os << indent << "Maxima Incremento de: "<< this->m_Inc<< std::endl;
  
}


} // end namespace


#endif
