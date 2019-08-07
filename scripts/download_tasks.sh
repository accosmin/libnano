#!/bin/bash

dir_exp=$HOME/experiments/results
dir_data=$HOME/experiments/datasets

# MNIST dataset
function download_mnist {
    local dir=${dir_data}/mnist/
    mkdir -p ${dir}

    wget -N http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ${dir}
    wget -N http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ${dir}
    wget -N http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ${dir}
    wget -N http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ${dir}
}

# Fashion-MNIST dataset
function download_fashion_mnist {
    local dir=${dir_data}/fashion-mnist/
    mkdir -p ${dir}

    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -P ${dir}
    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -P ${dir}
    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -P ${dir}
    wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -P ${dir}
}

# CIFAR10 dataset
function download_cifar10 {
    local dir=${dir_data}/cifar10/
    mkdir -p ${dir}

    wget -N http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P ${dir}
}

# CIFAR100 dataset
function download_cifar100 {
    local dir=${dir_data}/cifar100/
    mkdir -p ${dir}

    wget -N http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -P ${dir}
}

# IRIS dataset
function download_iris {
    local dir=${dir_data}/iris/
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -P ${dir}
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names -P ${dir}
}

# WINE dataset
function download_wine {
    local dir=${dir_data}/wine/
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data -P ${dir}
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names -P ${dir}
}

# ADULT dataset
function download_adult {
    local dir=${dir_data}/adult
    mkdir -p ${dir}

    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -P ${dir}
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -P ${dir}
    wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names -P ${dir}
}

# FOREST fires dataset
function download_forest_fires {
    local dir=${dir_data}/forest-fires
    mkdir -p ${dir}

    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv -P ${dir}
    wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.names -P ${dir}
}

# CALIFORNIA housing dataset
function download_cal_housing {
    local dir=${dir_data}/cal-housing
    mkdir -p ${dir}

    wget -N http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz -P ${dir}
    tar xvf ${dir}/cal_housing.tgz -C ${dir}
    mv ${dir}/CaliforniaHousing/* ${dir}/
}

# Process command line
function usage {
    cat <<EOF
usage: $0 [OPTIONS]

options:
    -h,--help
        print usage
    --mnist
        download MNIST dataset
    --fashion-minst
        download Fashion-MNIST dataset
    --iris
        download IRIS dataset
    --wine
        download WINE dataset
    --adult
        download ADULT dataset
    --cifar10
        download CIFAR-10 dataset
    --cifar100
        download CIFAR-100 dataset
    --cal-housing
        download California Housing dataset
    --forest-fires
        download Forest Fires dataset
EOF
    exit 1
}

if [ "$1" == "" ]; then
    usage
fi

while [ "$1" != "" ]; do
    case $1 in
        -h | --help)        usage
                            ;;
        --wine)             download_wine
                            ;;
        --iris)             download_iris
                            ;;
        --adult)            download_adult
                            ;;
        --mnist)            download_mnist
                            ;;
        --fashion-mnist)    download_fashion_mnist
                            ;;
        --cifar10)          download_cifar10
                            ;;
        --cifar100)         download_cifar100
                            ;;
        --cal-housing)      download_cal_housing
                            ;;
        --forest-fires)     download_forest_fires
                            ;;
        *)                  echo "unrecognized option $1"
                            echo
                            usage
                            ;;
    esac
    shift
done

exit 0
