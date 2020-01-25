#!/usr/bin/env bash

# install CUDA 10
echo "Checking for CUDA and installing."
echo "---------------------------------"
# Check for CUDA and try to install.
if [[ "$(dpkg-query -W cuda)" != *"10."* ]]; then
  sudo apt-get install dirmngr -y
  wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub  | sudo apt-key add -
  curl -O http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1604-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
  sudo dpkg -i ./cuda-repo-ubuntu1604-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
  sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
  sudo apt-get update
  sudo apt-get install cuda -y
fi

if [ -z "$(cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2)" ]; then
  # Install cuDNN 5.1
  echo "Install cuDNN 5.1"
  echo "-----------------"
  CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
  wget -c ${CUDNN_URL}
  sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
  rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
fi



echo "Set CUDA env vars"
echo "-----------------"

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc

echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc

source ~/.bashrc


if [[ "$(python3.7 --version)" != *"3.7"* ]]; then
  echo "Install python 3.7"
  echo "------------------"
  sudo apt-get install libffi-dev libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev

  wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz
  tar -xf Python-3.7.4.tgz

  cd Python-3.7.4
  ./configure --enable-optimizations
  make -j `nproc`
  sudo make install

  cd ..

  # sudo rm /usr/bin/python3
  # sudo ln -s /usr/local/bin/python3.7 /usr/bin/python3

  python3.7 --version
fi


echo "Creating virtual environment"
python3.7 -m venv vibe-env
echo "Activating virtual environment"

source $PWD/vibe-env/bin/activate

$PWD/vibe-env/bin/pip install numpy torch torchvision

$PWD/vibe-env/bin/pip install -r requirements.txt

sudo apt install freeglut3-dev
sudo apt-get install libav-tools
