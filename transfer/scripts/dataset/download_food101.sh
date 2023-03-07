# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

if [ "$#" -ne 1 ]; then
    echo "We expect one argument: Directory to download the dataset"
    exit
fi

echo ""
echo "**************************************************"
echo "Downloading Food101"

dir=${1}
mkdir -p ${dir}
cd ${dir}

wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf food-101.tar.gz
mv food-101/* .
rm -r food-101/

echo "Done!"