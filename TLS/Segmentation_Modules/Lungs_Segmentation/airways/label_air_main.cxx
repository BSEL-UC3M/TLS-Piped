#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cctype>
#include <math.h>
#include <map>

using namespace std;
using namespace boost;

typedef vector<string> LINE;

void printChild(multimap<int,int> m, int curNode, string name, int childCount, int left);
int main ( int argc, char ** argv )
{
	if( argc < 2 )
	{
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0] << " segment_tree.csv" << std::endl;
		return -1;
	}

	ifstream inputStream;
	typedef tokenizer< escaped_list_separator<char> > Tokenizer;
	multimap<int,int> mymm;
	//  multimap<int,int>::iterator it;

/*
	std::stringstream segments;
	segments << argv[1];
	string segmentsFile = segments.str();
*/
	inputStream.open(argv[1]);
//	inputStream.open(segmentsFile.c_str());

	string line;
	double l2, l3;
	if( !inputStream ) {
		cerr << "Error opening input stream" << endl;
	} else {

		while (getline(inputStream,line))
		{
			int loc=1;

			string child, parent, str;
			Tokenizer tok(line);
			for (Tokenizer::iterator it(tok.begin()),
					end(tok.end());
					it != end; ++it)
			{
				if (loc==1) // segmentID
				{
					child =  ((*it));
					if (atoi(child.c_str())==0) {
						break;
					}
					// new struct
					// //    	    	cout << "Reading segment #" << child << " into tree" << endl;
				}
				else if (loc==2)
				{
					parent = ((*it));

					mymm.insert(pair<int, int>(atoi(parent.c_str()), atoi(child.c_str())));
					//     	        	        	        	    	    	    	    	    	cout << "Inserting segment#" << child << " w parent segment #" << parent <<" into tree" << endl;
				}
				else if (loc==3)
				{
					str = ((*it));
					if (atoi(child.c_str())==2) l2 = atof(str.c_str());
					else if (atoi(child.c_str())==3) l3 = atof(str.c_str());
				}
				loc++;
			}
		}
	}

		// print content:
		cout << "elements in mymm:" << endl;
		//  cout << "y => " << mymm.find('y')->second << endl;
		//  cout << "z => " << mymm.find('z')->second << endl;

		if (l2 > l3)
			printChild(mymm, 1, "T", 0, 2);
		else
			printChild(mymm, 1, "T", 0, 3);
		//  printChild(mymm, 2, "R", 0);
		//  printChild(mymm, 3, "L", 0);
		return 0;

	}

	void printChild(multimap<int,int> m, int curNode, string name, int mechildCount, int left) {
		pair<multimap<int, int>::const_iterator, multimap<int, int>::const_iterator> it = m.equal_range(curNode);

		if (m.count(curNode)==0) {
			//     cout << "DEBUG" << " Node "<< curNode << " has no children" << endl;
			cout << curNode << " " << name << "." << mechildCount << endl;
		} else {
			multimap<int, int>::const_iterator it2;
			int childCount = 0;

			string newnameStr;
			if (curNode==1) name = "T";
			else if (curNode==2) {
				if (left==2)
					name = "L";
				else
					name = "R";
			}
			else if (curNode==3) {
				if (left==2)
					name = "R";
				else
					name = "L";
			}
			//  	if (curNode > 3){
			else {
				std::stringstream newname;
				newname << name << "." << mechildCount;
				newnameStr = newname.str();
			}
			if (curNode > 3)
				cout << curNode << " " << newnameStr  << endl;
			else
				cout << curNode << " " << name  << endl;

			for (it2=it.first; it2!=it.second; ++it2) {
				++childCount;

				int child = it2->second;
				//	cout << "DEBUG" << " Child of " << it2->first << ": " << it2->second << endl;
				if (curNode > 3){
					//        cout << curNode << " says: My name is " << newnameStr  << endl;
					printChild(m, child, newnameStr, childCount, left);
				} else {
					printChild(m, child, name, childCount, left);
				}
			}
		}
	}
