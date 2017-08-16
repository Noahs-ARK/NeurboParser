# NeurboParser
coming soon...

## Instructions to run
### Get code
```
git clone https://github.com/Noahs-ARK/NeurboParser.git
cd NeurboParser
```

### Install dependencies
```
./install_deps.sh
```

### Install boost
```
cd deps
wget https://newcontinuum.dl.sourceforge.net/project/boost/boost/1.64.0/boost_1_64_0.tar.gz
tar xvzf boost_1_64_0.tar.gz
cd boost_1_64_0
./bootstrap.sh --prefix=`pwd`/../local
./b2 install
cd ..
cd ..
```

### Install dynet
```
git clone https://github.com/clab/dynet.git
cd dynet
git checkout d64ef48768d13d6115207ff54e488e6ad5baa976
hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb
ln -s eigen/Eigen .
ln -s eigen/unsupported .
cd ..
```

### Compile Parser
```
cd NeurboParser
mkdir build
cd build
cmake -DBOOST_INCLUDEDIR=`pwd`/../../deps/local/include ..
make
```

You'll also need to get embeddings (eg. from GloVe), and create a few folders like `log`, `model`, `prediction` before running `train.sh` or `test.sh`.

P.S. We have to use a specific version of dynet, since there are breaking changes after the commit referenced above.
