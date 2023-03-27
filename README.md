# Required libraries:
- METIS
- NetworkX
- Bitsets
## METIS:
### Install METIS package:
1) Make sure the environment has `gcc` and `cmake`.\
   If not:\
   a) Install `gcc`:
   ```
   sudo apt update
   sudo apt install build-essential
   ```
   b) Install `cmake`:\
   Install dependency packages:
   ```
   sudo apt update
   sudo apt-get install build-essential libssl-dev\
   ```
   Download source code (may create a new directory or download directly into `/tmp`):
   ```
   cd /tmp
   wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
   ```
   Extract file:
   ```
   tar -zxvf cmake-3.20.0.tar.gz
   ```
   Finish installation:
   ```
   cd cmake-3.20.0
   ./bootstrap
   make
   sudo make install
   ```
2) Download and install METIS package
```
cd /tmp
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
gunzip metis-5.x.y.tar.gz
tar -xvf metis-5.x.y.tar
make config shared=1
```
### Install METIS library in Python environment:
1) Activate your environment.
2) Install METIS library by `pip`:
```
pip install metis
```
## NeworkX
```
pip install networkx
```
## Bitsets
```
pip install bitsets
```

# Run experiments:
Run independently
```
<!-- CIFAR-10 -->
bash ./script/part2/cifar10_ideal0.sh
bash ./script/part2/cifar10_ideal1.sh
bash ./script/part2/cifar10_ideal2.sh
bash ./script/part2/cifar10_ideal3.sh
bash ./script/part2/cifar10_ideal4.sh
bash ./script/part2/cifar10_ideal5.sh
bash ./script/part2/cifar10_ideal6.sh
bash ./script/part2/cifar10_ideal7.sh
bash ./script/part2/cifar10_ideal8.sh
bash ./script/part2/cifar10_ideal9.sh

<!-- CIFAR-100 -->
bash ./script/part2/cifar100_ideal0.sh
bash ./script/part2/cifar100_ideal1.sh
bash ./script/part2/cifar100_ideal2.sh
bash ./script/part2/cifar100_ideal3.sh
bash ./script/part2/cifar100_ideal4.sh
bash ./script/part2/cifar100_ideal5.sh
bash ./script/part2/cifar100_ideal6.sh
bash ./script/part2/cifar100_ideal7.sh
bash ./script/part2/cifar100_ideal8.sh
bash ./script/part2/cifar100_ideal9.sh
```