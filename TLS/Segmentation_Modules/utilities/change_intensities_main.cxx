#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkImageRegionIterator.h"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>
namespace po = boost::program_options;

typedef itk::Image<signed int, 3>  ImageType;

int main(int argc, char* argv[])
{

	std::string inputImage;
	std::string outputImage;
	int wmin = 0;
	int wMAX = 0;
	int omin = 0;
	int oMAX = 0;

	try {
    	// Declare the supported options.
    	po::options_description desc("Allowed options");
    	desc.add_options()
    	    ("help,h", "produce help message")
    	    ("input,i", po::value< std::string >(&inputImage), "input image")
    	    ("output,o", po::value< std::string >(&outputImage), "output image")
			("windowmin,w", po::value<int>(&wmin)->default_value(0),
					"Window minimum value")
			("windowMAX,W", po::value<int>(&wMAX)->default_value(0),
					"Window maximum value")
			("outmin,x", po::value<int>(&omin)->default_value(0),
					"Output minimum value")
			("outMAX,X", po::value<int>(&oMAX)->default_value(255),
					"Output maximum value")
    	;

    	po::variables_map vm;
    	po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    	po::notify(vm);

    	if (vm.count("help") || (argc == 1)) {
			std::cout << desc << std::endl;
			return 1;
		}

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

    typedef itk::ImageFileReader < ImageType >  ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( inputImage );

    try
      {
      reader->Update();
      }
    catch( itk::ExceptionObject & err )
      {
      std::cerr << "ExceptionObject caught!" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
      }

  std::cout << "Window min: " << wmin << std::endl;
  std::cout << "Window MAX: " << wMAX << std::endl;
  std::cout << "Output min: " << omin << std::endl;
  std::cout << "Output MAX: " << oMAX << std::endl;
  ImageType::Pointer image = reader->GetOutput();

  typedef itk::IntensityWindowingImageFilter <ImageType, ImageType> IntensityWindowingImageFilterType;

  IntensityWindowingImageFilterType::Pointer filter = IntensityWindowingImageFilterType::New();
  filter->SetInput(image);
  filter->SetWindowMinimum(wmin);
  filter->SetWindowMaximum(wMAX);
  filter->SetOutputMinimum(omin);
  filter->SetOutputMaximum(oMAX);
  filter->Update();

  typedef  itk::ImageFileWriter< ImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(outputImage);
  writer->SetInput(filter->GetOutput());
  writer->Update();
 
  return EXIT_SUCCESS;
}
