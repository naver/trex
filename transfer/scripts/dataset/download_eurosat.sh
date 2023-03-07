# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

if [ "$#" -ne 1 ]; then
    echo "We expect one argument: Directory to download the dataset"
    exit
fi

echo ""
echo "**************************************************"
echo "Downloading EuroSAT"

dir=${1}
mkdir -p ${dir}
cd ${dir}

wget https://madm.dfki.de/files/sentinel/EuroSAT.zip
wget https://madm.dfki.de/files/sentinel/EuroSATallBands.zip
unzip EuroSAT.zip
mv 2750/* .
rm -r 2750/

echo "Done!"