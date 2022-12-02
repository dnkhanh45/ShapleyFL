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
1) Generate fedtask:
```
bash ./script/gen_task/gen_data.sh
```
2) Train:\
Run independently:
```
bash ./script/run_exp/cifar10/train16_iid.sh
bash ./script/run_exp/cifar10/train16_noniid.sh
bash ./script/run_exp/cifar10/train50_iid.sh
bash ./script/run_exp/mnist/train16_iid.sh
bash ./script/run_exp/mnist/train16_noniid.sh
bash ./script/run_exp/mnist/train50_iid.sh
```
3) Calculate Shapley values:\
Run independently:
```
bash ./script/run_sv/cifar10/sv16_iid_const.sh
bash ./script/run_sv/cifar10/sv16_iid_exact.sh
bash ./script/run_sv/cifar10/sv16_iid_opt.sh
bash ./script/run_sv/cifar10/sv16_noniid_const.sh
bash ./script/run_sv/cifar10/sv16_noniid_exact.sh
bash ./script/run_sv/cifar10/sv16_noniid_opt.sh
bash ./script/run_sv/cifar10/sv50_iid_const.sh
bash ./script/run_sv/cifar10/sv50_iid_exact.sh
bash ./script/run_sv/cifar10/sv50_iid_opt.sh
bash ./script/run_sv/mnist/sv16_iid_const.sh
bash ./script/run_sv/mnist/sv16_iid_exact.sh
bash ./script/run_sv/mnist/sv16_iid_opt.sh
bash ./script/run_sv/mnist/sv16_noniid_const.sh
bash ./script/run_sv/mnist/sv16_noniid_exact.sh
bash ./script/run_sv/mnist/sv16_noniid_opt.sh
bash ./script/run_sv/mnist/sv50_iid_const.sh
bash ./script/run_sv/mnist/sv50_iid_exact.sh
bash ./script/run_sv/mnist/sv50_iid_opt.sh
```