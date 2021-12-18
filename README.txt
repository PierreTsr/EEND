Pierre Hugo Tessier (Pierre Tessier), pht2119
12/17/2021

Project title: End to End Neural Diarization with Vector Clustering

Abstract:

The problem of diarizing an audio signal with numerous speakers interacting, sometimes overlapping, was long dominated by clustering-based approaches. Many improvements were found through the years and the current state-of-the-art using this pipeline, with x-vectors, remains today a golden standard. But recently appeared the first End-to-End Neural Diarization (EEND) models, which abstract all the internal sub-tasks and only learn to solve the high-level problem. In a few years, this technique became the current state-of-the-art in speech diarization, and its precision keeps increasing with the last research efforts. Moreover, it covers a case previously impossible with clustering: speech overlapping, which is surprisingly common in real-life scenarios. However, EEND suffers from a severe drawback: the computational cost explodes when the input sequence becomes too long, which limits potential uses to less than 10 minutes. That's why K. Kinoshita, M. Delcoix & N. Tawara proposed this year, an EEND variation making it possible to run a model on successive segments of the same audio and use embeddings as a disambiguation tool to merge all the diarization results. I will try in this project to reproduce their results and evaluate the performances of this approach compared to state-of-the-art diarization tools, as well as other EEND variations.

Project structure:

This project is based on https://github.com/hitachi-speech/EEND and keep the same structure.
All the paths are considered from the project root:
- eend/ contains the Python source files
- tools/ contains the recipes to build the environement
- utils/ is a single script to output the best diarization score (not very useful)
- egs/eend-vector-clustering/v1 is a Kaldi style egs to run the project
- EEND_tessier.diff contains the diff file with the original repository

I identified all my contributions with 
# pht2119
I also provided a quick explanation of my contributions when needed. When none is provided, it's  because I wrote the whole file or the whole function below the comment.

Requirements:

- Python 3.7 and building the project environement (see instructions below)
- activating the Python environement
- optional: downloading (LibriSpeech + MUSAN + sim_rir_8k) and/or AMI and/or CHiME5, to use other data than the provided sample

Executables to test the code:

- egs/eend-vector-clustering/v1/run.sh to run inference and scoring (training will be deactivated by default)
- egs/eend-vector-clustering/v1/local/run_prepare_shared.sh to run diarization data simulation

Instructions:

I will do my possible to setup the environement myself, but I might not be able to do it if I need to install additional packages.
From the project root, run:
$ cd tools
$ make KALDI=<path_to_kaldi_root>
Warning: the Makefile WILL try to build kaldi if the path to Kaldi's root is not provided.
Warning: If the Makefile doesn't end with 
"echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> env.sh"
(it doesn't always raise an error when failing), delete the created files in tools/ and rerun after fixing the failing part.

Activate the Python environement from the project root with:
$ source tools/miniconda3/bin/activate eend

Once the setup is done, to run the inference, from egs/eend-vector-clustering/v1/ run:
$ ./run.sh
I provide a default data sample. If you want to change it, please modify the "train_set" variable.
The experiment results stored in egs/eend-vector-clustering/v1/exp/diarize/scoring/ and from there it follows an intuitive naming convention depending on the exp setup. The script uses multiple smoothing parameters, and create one rttm and one result file for each of them.
The model is stored in egs/eend-vector-clustering/v1/exp/diarize/model/train_20000.dev_1000.clustering_train/avg58-60.nnet.npz
Some intermediate inference results are stored in egs/eend-vector-clustering/v1/exp/diarize/infer

To run the data simulation script, a few datasets are needed. The script is able to download automatically most of them, but not CHiME5. Moreover, this consumes a lot of storage (300-400 Gb). Thus, one must provide a correct path to CHiME5 in the "chime5" variable. Once this is done, please run:
$ ./local/run_prepare_shared.sh
This will create simulations according to the provided configuration. Once again, I will provide a demo configuration in order not build hundreds of Gb of simulated data.