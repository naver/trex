# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

if [ "$#" -ne 1 ]; then
    echo "We expect one argument: Directory to download the dataset"
    exit
fi

echo ""
echo "**************************************************"
echo "Downloading Flowers"

dir=${1}
mkdir -p ${dir}
cd ${dir}

wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz
wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/imagelabels.mat
wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/setid.mat
tar -xzf 102flowers.tgz

echo "Done!"