#ifndef DICOM_UTILITIES_H_
#define DICOM_UTILITIES_H_

#include <iostream>
#include "itkGDCMImageIO.h"
#include "itkMetaDataObject.h"

template <class InputImageTypePointer, class OutputImageTypePointer>
void copy_dicom_data(InputImageTypePointer src, OutputImageTypePointer dst) {

	typedef itk::MetaDataDictionary   DictionaryType;
	DictionaryType & src_dict = src->GetMetaDataDictionary();
	DictionaryType & dst_dict = dst->GetMetaDataDictionary();

	// Be sure that origin and spacing info is transferred
	dst->SetOrigin(src->GetOrigin());
	dst->SetSpacing(src->GetSpacing());
	// Copy slice thickness information
	if (src_dict.HasKey("0018|0050")) {
		dst_dict["0018|0050"] = src_dict["0018|0050"];
	}

}

template <class ImageTypePointer>
float get_slice_thickness_tag(ImageTypePointer src) {

	typedef itk::MetaDataDictionary   DictionaryType;
	typedef itk::MetaDataObject< std::string > MetaDataStringType;
	DictionaryType & src_dict = src->GetMetaDataDictionary();
	DictionaryType::ConstIterator sl = src_dict.Find("0018|0050");

	std::string slice_thickness="1.0";

	if (sl != src_dict.End()) {
		MetaDataStringType::Pointer entryvalue =
			    dynamic_cast<MetaDataStringType *>( sl->second.GetPointer() ) ;

				  slice_thickness = entryvalue->GetMetaDataObjectValue();




	}


	return atof(slice_thickness.c_str());

}


#endif /* DICOM_UTILITIES_H_ */
