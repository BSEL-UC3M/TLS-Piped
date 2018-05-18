#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>

#include "vtkOBJWriter.h"

int main (int argc, char *argv[])
{
  //Verify command line arguments
  if(argc != 3)
    {
    cout << "Required arguments: inputfile outputfile(obj)" << endl;
    return EXIT_FAILURE;
    }
  
  //Parse command line arguments
  vtkstd::string inputFilename = argv[1];
  vtkstd::string outputFilename = argv[2];
  
  vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
  reader->SetFileName(inputFilename.c_str());
  
  vtkSmartPointer<vtkOBJWriter> writer = 
      vtkSmartPointer<vtkOBJWriter>::New();
  writer->SetInputConnection(reader->GetOutputPort());
  writer->SetFileName(outputFilename.c_str());
  writer->Update();
  
  return EXIT_SUCCESS;
}
