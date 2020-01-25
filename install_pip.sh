#!/usr/bin/env bash

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
