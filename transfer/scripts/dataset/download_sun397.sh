# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

if [ "$#" -ne 1 ]; then
    echo "We expect one argument: Directory to download the dataset"
    exit
fi

echo ""
echo "**************************************************"
echo "Downloading SUN397"

dir=${1}
mkdir -p ${dir}
cd ${dir}

wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
wget https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip

tar -xzf SUN397.tar.gz
unzip Partitions.zip

echo "Done!"