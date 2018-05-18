

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR}/Segmentation_Modules/libs/)

#message(${ITK_USE_FILE} ITKUSEFILE)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE}  )

FIND_PACKAGE(VTK REQUIRED)
include(${VTK_USE_FILE})

#set(Boost_USE_STATIC_LIBS OFF) 
#set(Boost_USE_MULTITHREADED ON)  
#set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost COMPONENTS program_options REQUIRED)
#find_package(Boost 1.59 COMPONENTS program_options REQUIRED)

INCLUDE_DIRECTORIES( ${BOOST_LIBRARYDIR} )
INCLUDE_DIRECTORIES( ${BOOST_ROOT} )
include_directories(${Boost_INCLUDE_DIRS}) 

#include_directories(${Boost_DIR})


ADD_EXECUTABLE( change_int change_intensities_main.cxx)
TARGET_LINK_LIBRARIES(change_int ${ITK_LIBRARIES} ${Boost_LIBRARIES})

ADD_EXECUTABLE( copy_image copy_image.cxx) 
TARGET_LINK_LIBRARIES(copy_image ${ITK_LIBRARIES})

ADD_EXECUTABLE( image2vtkPolyData itkReadITKImageVTKPolyDataWrite.cxx )
TARGET_LINK_LIBRARIES( image2vtkPolyData ${ITK_LIBRARIES} ${VTK_LIBRARIES})

ADD_EXECUTABLE(convert2dicom convert2dicom.cxx )
TARGET_LINK_LIBRARIES(convert2dicom  ${ITK_LIBRARIES})

ADD_EXECUTABLE(ImageReadDicomSeriesWrite ImageReadDicomSeriesWrite.cxx )
TARGET_LINK_LIBRARIES(ImageReadDicomSeriesWrite  ${ITK_LIBRARIES})

ADD_EXECUTABLE(label_and_colorize label_and_colorize.cxx )
TARGET_LINK_LIBRARIES(label_and_colorize  ${ITK_LIBRARIES})

ADD_EXECUTABLE(MedianImageFilter MedianImageFilter.cxx )
TARGET_LINK_LIBRARIES(MedianImageFilter  ${ITK_LIBRARIES})

ADD_EXECUTABLE(extract_roi extract_roi_main.cxx)
TARGET_LINK_LIBRARIES(extract_roi ${ITK_LIBRARIES} ${Boost_LIBRARIES})

ADD_EXECUTABLE(calibrate Calibrate_main.cxx)
TARGET_LINK_LIBRARIES(calibrate ${ITK_LIBRARIES})

ADD_EXECUTABLE(calculate_mean_std calculate_mean_std_main.cxx)
TARGET_LINK_LIBRARIES(calculate_mean_std ${ITK_LIBRARIES} ${Boost_LIBRARIES})

ADD_EXECUTABLE(read_orientation read_orientation_main.cxx)
TARGET_LINK_LIBRARIES(read_orientation  ${ITK_LIBRARIES})

ADD_EXECUTABLE(apply_median apply_median.cxx )
TARGET_LINK_LIBRARIES(apply_median  ${ITK_LIBRARIES})

ADD_EXECUTABLE(apply_erosion apply_erosion.cxx )
TARGET_LINK_LIBRARIES(apply_erosion  ${ITK_LIBRARIES})

ADD_EXECUTABLE(apply_dilation apply_dilation.cxx )
TARGET_LINK_LIBRARIES(apply_dilation  ${ITK_LIBRARIES})

ADD_EXECUTABLE(apply_gaussian apply_gaussian.cxx )
TARGET_LINK_LIBRARIES(apply_gaussian  ${ITK_LIBRARIES} ${Boost_LIBRARIES})

ADD_EXECUTABLE(dicom_serie2image DicomSeriesReadImageWrite2.cxx )
TARGET_LINK_LIBRARIES(dicom_serie2image  ${ITK_LIBRARIES})

ADD_EXECUTABLE(TracheaTest TracheaTesting.cxx)
TARGET_LINK_LIBRARIES(TracheaTest  ${ITK_LIBRARIES})

ADD_EXECUTABLE(TracheaTestSerie TracheaTestingSerie.cxx)
TARGET_LINK_LIBRARIES(TracheaTestSerie  ${ITK_LIBRARIES})

ADD_EXECUTABLE(isTrachea TracheaError.cxx)
TARGET_LINK_LIBRARIES(isTrachea  ${ITK_LIBRARIES})

ADD_EXECUTABLE(dicom_extractor dicom_extractor.cxx)
TARGET_LINK_LIBRARIES(dicom_extractor  ${ITK_LIBRARIES})

ADD_EXECUTABLE(simple_dicom_extractor simple_dicom_extractor.cxx)
TARGET_LINK_LIBRARIES(simple_dicom_extractor  ${ITK_LIBRARIES})

ADD_EXECUTABLE(apply_mask apply_mask.cxx)
TARGET_LINK_LIBRARIES(apply_mask ${ITK_LIBRARIES})

ADD_EXECUTABLE(hist ImageHistogram2.cxx)
TARGET_LINK_LIBRARIES(hist ${ITK_LIBRARIES})

ADD_EXECUTABLE(inv_image itkInvImageFilter.txx inv_image.cxx)
TARGET_LINK_LIBRARIES(inv_image ${ITK_LIBRARIES} helpers)

ADD_EXECUTABLE(find_trachea find_trachea.cxx)
TARGET_LINK_LIBRARIES(find_trachea ${ITK_LIBRARIES})

ADD_EXECUTABLE(cast2char CastToChar.cxx)
TARGET_LINK_LIBRARIES(cast2char ${ITK_LIBRARIES})

ADD_EXECUTABLE(change_values ChangeValues.cxx)
TARGET_LINK_LIBRARIES(change_values ${ITK_LIBRARIES})

ADD_EXECUTABLE(shrink Shrink.cxx)
TARGET_LINK_LIBRARIES(shrink ${ITK_LIBRARIES})

ADD_EXECUTABLE(merge_images MergeImages.cxx)
TARGET_LINK_LIBRARIES(merge_images ${ITK_LIBRARIES})

ADD_EXECUTABLE(ResampleVolumesToBeIsotropic ResampleVolumesToBeIsotropic.cxx)
TARGET_LINK_LIBRARIES(ResampleVolumesToBeIsotropic ${ITK_LIBRARIES})
