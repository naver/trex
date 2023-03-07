# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

if [ "$#" -ne 1 ]; then
    echo "We expect one argument: Directory to download the dataset"
    exit
fi


echo ""
echo "**************************************************"
echo "Downloading Cars196"

dir=${1}
mkdir -p ${dir}
cd ${dir}

wget http://ai.stanford.edu/~jkrause/car196/car_ims.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_annos.mat
tar -xzf car_ims.tgz

echo "Done!"