#!/usr/bin/env bash

stage=2

if [ $stage -le 0 ]; then
    local/get_amicorpus.sh
    mv CCBY4.0.txt amicorpus/
    mv amiBuild-* amicorpus/
fi

if [ $stage -le 1 ]; then
    tar -xf ami_annoations.tar -C amicorpus/
fi

if [ $stage -le 2 ]; then
    rm -rf data/ami
    mkdir -p data/ami
    touch data/ami/wav.scp
    touch data/ami/reco2file_and_channel
    for filename in amicorpus/*; do
        if [ -d "$filename"/audio ]; then
            file=$(basename "$filename")
            path=$(realpath "$filename"/audio/*)
            echo "$file" "$path" >> data/ami/wav.scp
            echo "$file" "$file" A >> data/ami/reco2file_and_channel
        fi
    done
    local/convert_rttm_to_utt2spk_and_segments.py amicorpus/annotations.rttm data/ami/reco2file_and_channel data/ami/utt2spk data/ami/segments
    utils/utt2spk_to_spk2utt.pl data/ami/utt2spk > data/ami/spk2utt
    utils/validate_data_dir.sh --no-feats --no-text --non-print data/ami
fi