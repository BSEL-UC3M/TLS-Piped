#!/bin/bash
#
# Script Name: install.sh
#
# Author: pmacias
# Date : 10/09/2017
#
# Description: The following script reads install the TLS app.
#
# Run Information: The script could use one parameter to indicate the nunbers of cores to use 
#                  to employ within compilation processe. By default all the cores are employed.
#
APP_NAME="TLS"
BIN_DIR="TLS_BUILD"
echo "Installing "$APP_NAME" "$BIN_DIR

apt-get update && sudo apt-get -y upgrade

apt-get -qq -y install g++ gcc git-core git-svn libfontconfig-dev libgl1-mesa-dev libglu1-mesa-dev libncurses5-dev libosmesa6-dev libx11-dev libxrender-dev libxt-dev make python-dev python-numpy subversion libbz2-dev cmake libboost-dev python-pip

pip install -r requirements.txt

mkdir -m 777 ../$BIN_DIR
cd ../$BIN_DIR 
sudo cmake ../TLS -DUSE_GIT=ON -DUSE_Boost=ON -DUSE_OpenCV=ON -DUSE_ITK=ON -DUSE_VTK=ON -DUSE_zlib=ON
make -j$1
sudo cmake ../TLS -DUSE_GIT=ON -DUSE_Boost=ON -DUSE_OpenCV=ON -DUSE_ITK=ON -DUSE_VTK=ON -DUSE_zlib=ON -DTLS_SECOND_STEP=ON
make -j$1
