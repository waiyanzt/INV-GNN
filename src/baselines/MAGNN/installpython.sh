!/bin/bash

apt update
apt install -y build-essential zlib1g-dev

curPath=${pwd}
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar xzf Python-3.7.9.tgz
cd Python-3.7.9
#./configure --enable-optimizations
./configure 
make
make install
cd ${pwd}
