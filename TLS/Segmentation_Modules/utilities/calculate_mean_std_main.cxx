
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include <iostream>
#include <string>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionConstIterator.h"
#include "itkStatisticsImageFilter.h"
#include <itkMaskImageFilter.h>

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>
namespace po = boost::program_options;


int main(int argc, char * argv[])
{

	std::string inString;
	std::string maskString;
	bool useMask = false;
	int mask_id = 0;
	int cutoff = -1000;
	try {
    	// Declare the supported options.
    	po::options_description desc("Allowed options");
    	desc.add_options()
    	    ("help,h", "produce help message")
    	    ("input,i", po::value< std::string >(&inString), "input grayscale file")
    	    ("mask,m", po::value< std::string >(&maskString), "input mask file")
			("label,l", po::value<int>(&mask_id)->default_value(0),
					"Label value. If different from 0 only pixels with label value in the mask will be transparent")
			("cutoff,c", po::value<int>(&cutoff)->default_value(-1000),
					"In case the distribution is skewed, set this to discard pixels which value is lesser or equal to this.")
    	;

    	po::variables_map vm;
    	po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    	po::notify(vm);

    	if (vm.count("help") || (argc == 1)) {
			std::cout << desc << std::endl;
			return 1;
		}

    	if (vm.count("mask"))
    		useMask = true;


    }
    catch(std::logic_error &err)
    {
         std::cerr << "std::logic_error:" << err.what() << std::endl;
         for (int i=0; i< argc; i++)
             std::cerr << argv[i] << std::endl;
    }
    catch(std::exception& e) {
    	std::cout << e.what() << std::endl;
    }


  if (mask_id >= 0) {
	  if (mask_id > 0) {
		  std::cout << "Considering only pixels with value " << mask_id << std::endl;
	  }
  }  else {
	  std::cout << "Invalid mask value. Should be >0 instead of " << mask_id << std::endl;
	  mask_id = 0;
  }


  const int Dimension = 3;

  typedef signed short                                        SSPixelType;
  typedef unsigned short                                        USPixelType;
  typedef itk::Image<SSPixelType, Dimension>                         ShortImageType;
  typedef itk::Image<USPixelType, Dimension>                         BinImageType;

  typedef itk::ImageFileReader<ShortImageType>                     ShortReaderType;
  typedef itk::ImageFileReader<BinImageType>                       BinReaderType;

  typedef itk::MaskImageFilter< ShortImageType, BinImageType, ShortImageType> MaskFilterType;
  typedef itk::StatisticsImageFilter<ShortImageType> StatisticsImageFilterType;

  ShortReaderType::Pointer inReader = ShortReaderType::New();
  inReader->SetFileName(inString.c_str());

  BinReaderType::Pointer maskReader = BinReaderType::New();

  try {
    inReader->Update();

    if (useMask) {
    	maskReader->SetFileName(maskString.c_str());
    	maskReader->Update();
    }
  }
  catch (itk::ExceptionObject & excp) {
    std::cerr << "  Problems reading an image"   << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }

  ShortImageType::Pointer image;

  if (useMask) {
	  MaskFilterType::Pointer mask = MaskFilterType::New();
	  mask->SetInput1(inReader->GetOutput());
	  mask->SetInput2(maskReader->GetOutput());
	  mask->Update();
	  image = mask->GetOutput();
  } else {
	  image = inReader->GetOutput();
  }

  StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New ();
  statisticsImageFilter->SetInput(image);
  statisticsImageFilter->Update();

  std::cout << "Mean: " << statisticsImageFilter->GetMean() << std::endl;
  std::cout << "Std: " << statisticsImageFilter->GetSigma() << std::endl;
  std::cout << "Min: " << statisticsImageFilter->GetMinimum() << std::endl;
  std::cout << "Max: " << statisticsImageFilter->GetMaximum() << std::endl;


  return EXIT_SUCCESS;
}

