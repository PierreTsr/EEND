#!/usr/bin/env bash


CHIME5=$1
data=$2

for dset in dev eval train; do
    rm -rf $data/chime5/$dset
    mkdir -p $data/chime5/$dset
    cat $CHIME5/$dset/rttm >> $data/chime5/$dset/rttm
    cat $CHIME5/$dset/wav.scp >> $data/chime5/$dset/wav.scp
    cat $CHIME5/$dset/reco2file_and_channel >> $data/chime5/$dset/reco2file_and_channel
    local/convert_rttm_to_utt2spk_and_segments.py $data/chime5/$dset/rttm $data/chime5/$dset/reco2file_and_channel $data/chime5/$dset/utt2spk $data/chime5/$dset/segments
done