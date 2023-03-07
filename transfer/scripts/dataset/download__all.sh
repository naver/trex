# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

if [ "$#" -ne 1 ]; then
    echo "We expect one argument: Root directory where to download all datasets"
    exit
fi

root_dir=${1}
mkdir -p ${root_dir}

bash transfer/scripts/dataset/download_aircraft.sh ${root_dir}/aircraft
bash transfer/scripts/dataset/download_cars196.sh ${root_dir}/cars196
bash transfer/scripts/dataset/download_dtd.sh ${root_dir}/dtd
bash transfer/scripts/dataset/download_eurosat.sh ${root_dir}/eurosat
bash transfer/scripts/dataset/download_flowers.sh ${root_dir}/flowers
bash transfer/scripts/dataset/download_food101.sh ${root_dir}/food101
bash transfer/scripts/dataset/download_pets.sh ${root_dir}/pets
bash transfer/scripts/dataset/download_sun397.sh ${root_dir}/sun397