#!/bin/bash

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# Modified by Pierre Tessier according to the experimental process described in:
# Advances in integration of end-to-end neural and clustering-based diarization
# for real conversational speech, K. Kinoshita et al.
#
# This script prepares kaldi-style data sets shared with different experiments
#   - data/xxxx
#     callhome, sre, swb2, and swb_cellular datasets
#   - data/simu_${simu_outputs}
#     simulation mixtures generated with various options
# This script does NOT include the composition of train/valid/test sets.
# The composition will be done at stage 1 of ./run.sh

stage=0

# This script distributes simulated data under these directories
simu_actual_dirs=(
	$PWD/data/local/diarization-data
)

# simulation options
simu_opts_overlap=yes

. path.sh
. cmd.sh
. parse_options.sh || exit

librispeech_dev=LibriSpeech/dev-clean/
librispeech_test=LibriSpeech/test-clean/
librispeech_train_100=LibriSpeech/train-clean-100/
librispeech_train_360=LibriSpeech/train-clean-360/
musan=musan/
sim_rir_8k=simulated_rirs_8k/
chime5=../../../../corpora/CHIME5

if [ $stage -le 0 ]; then
	if [ ! -d $librispeech_dev ]; then
		local/get_librispeech.sh
		librispeech_dev=LibriSpeech/dev-clean/
		librispeech_test=LibriSpeech/test-clean/
		librispeech_train_100=LibriSpeech/train-clean-100/
		librispeech_train_360=LibriSpeech/train-clean-360/
	fi

	if [ ! -d $musan ]; then
		wget https://us.openslr.org/resources/17/musan.tar.gz
		tar -xf musan.tar.gz
		musan=musan
	fi

	if [ ! -d $sim_rir_8k ]; then
		wget https://us.openslr.org/resources/26/sim_rir_8k.zip
		unzip -q sim_rir_8k.zip
		sim_rir_8k=simulated_rirs_8k
	fi

	if [ ! -d $CHIME5 ]; then
		echo "$0 unable to find CHIME5 directory" && exit
	fi
fi


if [ $stage -le 1 ]; then
	echo "prepare kaldi-style datasets"
	if [ ! -f data/dev_clean/.done ]; then
		local/data_prep.sh $librispeech_dev data/dev_clean || exit
		touch data/dev_clean/.done
	fi
	if [ ! -f data/test_clean/.done ]; then
		local/data_prep.sh $librispeech_dev data/test_clean || exit
		touch data/test_clean/.done
	fi
	if [ ! -f data/train_clean/.done ]; then
		local/data_prep.sh $librispeech_dev data/train_clean_100 || exit
		local/data_prep.sh $librispeech_dev data/train_clean_360 || exit
		utils/data/combine_data.sh data/train_clean/ data/train_clean_100 data/train_clean_360 || exit
		rm -rf data/train_clean_100
		rm -rf data/train_clean_360
		touch data/train_clean/.done
	fi
	if [ ! -f data/musan/.done ]; then
		steps/data/make_musan.sh --sampling-rate 8000 --use-vocals false $musan data/musan
		touch data/musan/.done
	fi
	if [ ! -f data/sim_rir_8k/.done ]; then
		mkdir -p data/sim_rir_8k
		find $sim_rir_8k -iname "*.wav" |
			awk '{n=split($1,A,/[\/\.]/); print A[n-3]"_"A[n-1], $1}' |
			sort >data/sim_rir_8k/wav.scp
		awk '{print $1, $1}' data/sim_rir_8k/wav.scp >data/sim_rir_8k/utt2spk
		utils/fix_data_dir.sh data/sim_rir_8k
		touch data/sim_rir_8k/.done
	fi
fi

simudir=data/simu
if [ $stage -le 2 ]; then
	echo "simulation of mixture"
	mkdir -p $simudir/.work
	random_mixture_cmd=random_mixture_nooverlap.py
	make_mixture_cmd=make_mixture_nooverlap.py
	if [ "$simu_opts_overlap" == "yes" ]; then
		random_mixture_cmd=random_mixture.py
		make_mixture_cmd=make_mixture.py
	fi

	while read -r line; do
		while IFS='-' read -ra config; do
			[[ "${config[0]}" =~ ([A-Za-z]+)\ ([0-9]+) ]] && dset="${BASH_REMATCH[1]}" && n_config="${BASH_REMATCH[2]}"
			total_utterances=0
			folders=()
			for i in $(seq 1 "$n_config"); do
				[[ "${config[$i]}" =~ ([0-9]+)\ ([0-9]+)\ ([0-9]+)\ ([0-9]+)\ ([0-9]+) ]] &&
					n_mixtures="${BASH_REMATCH[1]}" && min_utt_per_spk="${BASH_REMATCH[2]}" &&
					max_utt_per_spk="${BASH_REMATCH[3]}" && beta="${BASH_REMATCH[4]}" && n_speakers="${BASH_REMATCH[5]}"
				total_utterances=$((total_utterances + n_mixtures))
				src_dset="${dset}_clean"
				simuid=${src_dset}_ns${n_speakers}_beta${beta}_${n_mixtures}
				folders+=( $simudir/data/"$simuid" )

				echo "Simulating from $dset with $n_mixtures utterances, beta = $beta, $n_speakers speakers, and between $min_utt_per_spk and $max_utt_per_spk utt/spk"
				# check if you have the simulation
				if ! validate_data_dir.sh --no-text --no-feats $simudir/data/"$simuid"; then
					# random mixture generation
					$simu_cmd $simudir/.work/random_mixture_"$simuid".log \
						$random_mixture_cmd --n_speakers "$n_speakers" --n_mixtures "$n_mixtures" \
						--speech_rvb_probability 1 \
						--sil_scale "$beta" \
						--min_utts "$min_utt_per_spk" --max_utts "$max_utt_per_spk"\
						data/"$src_dset" data/musan/musan data/sim_rir_8k \
						\> $simudir/.work/mixture_"$simuid".scp
					nj=25
					mkdir -p $simudir/wav/"$simuid"
					# distribute simulated data to $simu_actual_dir
					split_scps=
					for n in $(seq $nj); do
						split_scps="$split_scps $simudir/.work/mixture_$simuid.$n.scp"
						mkdir -p $simudir/.work/data_"$simuid"."$n"
						actual=${simu_actual_dirs[($n - 1) % ${#simu_actual_dirs[@]}]}/$simudir/wav/$simuid/$n
						mkdir -p "$actual"
						ln -nfs "$actual" $simudir/wav/"$simuid"/"$n"
					done
					# shellcheck disable=SC2086
					utils/split_scp.pl $simudir/.work/mixture_"$simuid".scp $split_scps || exit 1

					$simu_cmd --max-jobs-run 32 JOB=1:$nj $simudir/.work/make_mixture_"$simuid".JOB.log \
						$make_mixture_cmd --rate=8000 \
						$simudir/.work/mixture_"$simuid".JOB.scp \
						$simudir/.work/data_"$simuid".JOB $simudir/wav/"$simuid"/JOB
					utils/combine_data.sh $simudir/data/"$simuid" $simudir/.work/data_"$simuid".*
					steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
						$simudir/data/"$simuid"/utt2spk $simudir/data/"$simuid"/segments \
						$simudir/data/"$simuid"/rttm
					utils/data/get_reco2dur.sh $simudir/data/"$simuid"
				fi
			done
			utils/data/combine_data.sh data/simu/"$dset"_"$total_utterances" "${folders[@]}"
		done <<<"$line"
	done <conf/simulation.conf
fi
