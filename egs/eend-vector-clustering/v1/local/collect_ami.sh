#!/usr/bin/env bash


AMI=$1
data=$2

for dset in dev eval train; do
    rm -rf $data/ami/$dset
    mkdir -p $data/ami/$dset
    cat $AMI/rttm/$dset/*.rttm >> $data/ami/$dset/rttm
    cat $AMI/wav.scp/$dset >> $data/ami/$dset/wav.scp
    cat $AMI/reco2file_and_channel/$dset >> $data/ami/$dset/reco2file_and_channel
    local/convert_rttm_to_utt2spk_and_segments.py $data/ami/$dset/rttm $data/ami/$dset/reco2file_and_channel $data/ami/$dset/utt2spk $data/ami/$dset/segments
done