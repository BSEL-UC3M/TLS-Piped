
#include "dicom_utilities.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkDiscreteGaussianImageFilter.h"
#include "itkCastImageFilter.h"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>
namespace po = boost::program_options;

int main( int argc, char * argv[] )
{

	std::string infilepath;
	std::string outfilepath;
	float variance = 1.0;

	try {
		// Declare the supported options.
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("input,i", po::value< std::string >(&infilepath), "input file")
			("output,o", po::value< std::string >(&outfilepath), "output file")
			("variance,v", po::value< float >(&variance), "Variance")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		po::notify(vm);


		/*
		 * General parameters handling
		 */

		if (vm.count("help") || (argc == 1)) {
			std::cout << desc << std::endl;
			return 1;
		}


	}
	catch(std::logic_error &err)
	{
		 std::cerr << "std::logic_error:" << err.what() << std::endl;
		 for (unsigned int i=0; i< argc; i++)
			 std::cerr << argv[i] << std::endl;
	}
	catch(std::exception& e) {
		std::cout << e.what() << std::endl;
	}

  typedef   signed short   InputPixelType;
  typedef   float  FloatPixelType;
  typedef   signed short  OutputPixelType;
  const unsigned int      Dimension = 3;

  typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
  typedef itk::Image< FloatPixelType, Dimension >   FloatImageType;
  typedef itk::Image< OutputPixelType, Dimension >   OutputImageType;
  typedef InputImageType::SizeType SizeType;

  typedef itk::ImageFileReader< InputImageType  >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  reader->SetFileName( infilepath.c_str() );
  writer->SetFileName( outfilepath.c_str() );

  typedef itk::DiscreteGaussianImageFilter<
		  InputImageType, FloatImageType >  FilterType;

  FilterType::Pointer filter = FilterType::New();
  filter->SetVariance(variance);

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


  filter->SetInput( reader->GetOutput() );
  filter->Update();
  copy_dicom_data(reader->GetOutput(), filter->GetOutput());

  typedef itk::CastImageFilter< FloatImageType, OutputImageType > CasterType;

  CasterType::Pointer caster = CasterType::New();
  caster->SetInput(filter->GetOutput());

  writer->SetInput( caster->GetOutput() );
  writer->Update();

  return EXIT_SUCCESS;
}

