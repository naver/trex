# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

if [ "$#" -ne 1 ]; then
    echo "We expect one argument: Directory to download the dataset"
    exit
fi

echo ""
echo "**************************************************"
echo "Downloading DTD"

dir=${1}
mkdir -p ${dir}
cd ${dir}

wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/README.txt
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz
mv dtd/* .
rm -r dtd/

echo "Done!"