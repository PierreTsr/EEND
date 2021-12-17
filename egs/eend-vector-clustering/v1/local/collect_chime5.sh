#!/usr/bin/env bash
# pht2119
# Script made entirely by myself to pre-process the CHiME5 dataset

CHIME5=$1
data=$2

stage=1

if [ $stage -le 0 ]; then
    echo "extracting channels and subsampling"
    if [ ! -f chime5/.done ]; then
        for dset in dev eval train; do
            mkdir -p chime5/"$dset"
            for filename in "$CHIME5"/audio/"$dset"/*.wav; do
                echo $filename
                n_channels=$(ffprobe -i "$filename" -show_entries stream=channels -select_streams a:0 -of compact=p=0:nk=1 -v 0)
                file=$(basename -s .wav $filename )
                if [ "$n_channels" -eq "1" ]; then
                    sox -G -v 0.99 "$filename" -t wav -r 8000 -b 16 chime5/"$dset"/"$file".wav
                fi
                if [ "$n_channels" -eq "2" ]; then
                    sox -G -v 0.99 "$filename" -t wav -r 8000 -b 16 chime5/"$dset"/"$file".CH1.wav remix 1
                    sox -G -v 0.99 "$filename" -t wav -r 8000 -b 16 chime5/"$dset"/"$file".CH2.wav remix 2
                fi
            done
        done
    fi
fi

if [ $stage -le 1 ]; then
    echo "constructing CHIME5's rttm"
    for dset in dev eval train; do
        mkdir -p chime5/rttm/"$dset"
        python local/annotations_to_rttm.py -i "$CHIME5"/transcriptions/"$dset" -o chime5/rttm/"$dset" --wav chime5/"$dset"
    done
fi

if [ $stage -le 2 ]; then
    for dset in dev eval train; do
        rm -rf $data/chime5/$dset
        mkdir -p $data/chime5/$dset
        cat chime5/rttm/$dset/rttm >> $data/chime5/$dset/rttm
        cat chime5/rttm/$dset/wav.scp >> $data/chime5/$dset/wav.scp
        cat chime5/rttm/$dset/reco2file_and_channel >> $data/chime5/$dset/reco2file_and_channel
        python local/convert_rttm_to_utt2spk_and_segments.py $data/chime5/$dset/rttm $data/chime5/$dset/reco2file_and_channel $data/chime5/$dset/utt2spk $data/chime5/$dset/segments
        utils/fix_data_dir.sh $data/chime5/$dset
    done
fi
