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
bash ./script/run_remote/cifar10_classification_cnum16_dist2_skew0.6_seed0.sh
bash ./script/run_remote/cifar10_classification_cnum16_dist6_skew0.6_seed0.sh
bash ./script/run_remote/cifar10_classification_cnum50_dist0_skew0_seed0.sh
bash ./script/run_remote/cifar10_classification_cnum50_dist2_skew0.6_seed0.sh
bash ./script/run_remote/mnist_classification_cnum16_dist2_skew0.6_seed0.sh
bash ./script/run_remote/mnist_classification_cnum16_dist6_skew0.6_seed0.sh
bash ./script/run_remote/mnist_classification_cnum50_dist0_skew0_seed0.sh
bash ./script/run_remote/mnist_classification_cnum50_dist2_skew0.6_seed0.sh
```