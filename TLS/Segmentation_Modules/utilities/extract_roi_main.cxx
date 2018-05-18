
#include "itkImageFileReader.h"
#include "itkExtractImageFilter.h"
#include "itkImageFileWriter.h"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>
namespace po = boost::program_options;

int main( int argc, char * argv[] )
{

    int xStart  = 0; int yStart  = 0; int zStart  = 0;
    int xExtent  = 0; int yExtent = 0; int zExtent  = 0;
    std::string infilename; 
    std::string outfilename;
    bool userSize = false;

    try {
    	// Declare the supported options.
    	po::options_description desc("Allowed options");
    	desc.add_options()
    	    ("help,h", "produce help message")
    	    ("xstart", po::value<int>(&xStart)->default_value(0), "X start index")
            ("ystart", po::value<int>(&yStart)->default_value(0), "Y start index")
    	    ("zstart", po::value<int>(&zStart)->default_value(0), "Z start index")            
            ("xsize,x", po::value<int>(&xExtent)->default_value(-1), "Size along x axis (Defaults to input image size)")
            ("ysize,y", po::value<int>(&yExtent)->default_value(-1), "Size along y axis (Defaults to input image size)")
            ("zsize,z", po::value<int>(&zExtent)->default_value(-1), "Size along z axis (Defaults to input image size)")
            ("infile,i", po::value<std::string>(&infilename), "Input image file")
            ("outfile,o", po::value<std::string>(&outfilename), "Output image file")
    	;

    	po::variables_map vm;
    	po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    	po::notify(vm);
    	
    	if (vm.count("help") || (argc == 1)) {
    	    std::cout << desc << std::endl;
    	    return 1;
    	}
    	
           
        if (vm.count("xsize") || vm.count("ysize") || vm.count("zsize")) 
            userSize=true;

    	
    	
    } catch(std::logic_error &err) {
         std::cerr << "std::logic_error:" << err.what() << std::endl;
         for (int i=0; i< argc; i++)
             std::cerr << argv[i] << std::endl;
    } catch(std::exception& e) {
    	std::cout << e.what() << std::endl;
    }
  
  typedef signed short PixelType;
  const unsigned int Dimension=3;

  typedef itk::Image< PixelType, Dimension >                       	ImageType;

  typedef itk::ImageFileReader< ImageType >                        	ReaderType;
  ReaderType::Pointer reader =                                     	ReaderType::New();
  typedef itk::ExtractImageFilter< ImageType, ImageType >		ExtractType;
  ExtractType::Pointer extractor =                                     	ExtractType::New();
  typedef itk::ImageFileWriter< ImageType >                        	WriterType;
  WriterType::Pointer writer =                                     	WriterType::New();
 

  reader->SetFileName( infilename.c_str() );
  writer->SetFileName( outfilename.c_str() );

  try {
	reader->Update();
  } catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught: can't read serie" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  extractor->SetInput( reader->GetOutput() );
  
  ImageType::RegionType in_region = reader->GetOutput()->GetLargestPossibleRegion();
 
 // Set extraction region start  
  ImageType::IndexType index;
  index.SetElement(0,xStart);
  index.SetElement(1,yStart);
  index.SetElement(2,zStart);

  // Set extraction region size
  ImageType::SizeType size=in_region.GetSize(); // Defaults to image size
  if (userSize) {
      if (xExtent >= 0)
          size.SetElement(0,xExtent);
      if (yExtent >= 0)
        size.SetElement(1,yExtent);
      if (zExtent >= 0)
        size.SetElement(2,zExtent);
  }
 
  ImageType::RegionType out_region;     
  out_region.SetSize(size);
  out_region.SetIndex(index);
  
  extractor->SetExtractionRegion( out_region );
  writer->SetInput( extractor->GetOutput() );
  
  try {
    writer->Update();
  } catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught: can't write serie" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }


  return EXIT_SUCCESS;

}
