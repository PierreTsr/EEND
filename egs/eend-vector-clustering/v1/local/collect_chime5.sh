#!/usr/bin/env bash


CHIME5=$1
data=$2

stage=0

if [ $stage -le 0]; then
    echo "constructing CHIME5's rttm"
    for dset in dev eval train; do
        mkdir -p "$CHIME5"/rttm/"$dset"
        local/annotations_to_rttm.py -i "$CHIME5"/transciptions/"$dset" -o "$CHIME5"/rttm/"$dset"
    done
fi

if [ $stage -le 1]; then
    for dset in dev eval train; do
        rm -rf $data/chime5/$dset
        mkdir -p $data/chime5/$dset
        cat $CHIME5/$dset/rttm >> $data/chime5/$dset/rttm
        cat $CHIME5/$dset/wav.scp >> $data/chime5/$dset/wav.scp
        cat $CHIME5/$dset/reco2file_and_channel >> $data/chime5/$dset/reco2file_and_channel
        local/convert_rttm_to_utt2spk_and_segments.py $data/chime5/$dset/rttm $data/chime5/$dset/reco2file_and_channel $data/chime5/$dset/utt2spk $data/chime5/$dset/segments
        utils/fix_data_dir.sh $data/chime5/$dset
    done
fi
