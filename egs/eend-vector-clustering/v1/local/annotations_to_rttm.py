# pht2119
# Script made entirely by myself to convert the CHiME5 annotations to the rttm format
import pandas as pd
import json
import csv
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Convert CHIME-5 annotations to rttm annotations')
parser.add_argument("-i", help="Input folder")
parser.add_argument("-o", help="Output folder")
parser.add_argument("--wav", help="path to the wav files")

channels_id = {
    "CH1": "A",
    "CH2": "B"
}

channels_num = {
    "A": "1",
    "B": "2"
}

def load_dataset(folder):
    data = []
    print(folder)
    for filename in Path(folder).rglob('*.json'):
        with open(filename) as file:
            data += json.load(file)
    df = pd.json_normalize(data, sep="_")

    for key in df.keys():
        if "time" in key:
            df[key] = pd.to_timedelta(df[key])

    end_times = list(filter(lambda s: "end_time" in s, df.keys()))
    start_times = list(filter(lambda s: "start_time" in s, df.keys()))
    other_keys = list(set(df.keys()) - set(end_times) - set(start_times))

    names = [key.split("_")[-1] for key in end_times]

    tmp1 = df.melt(id_vars=other_keys, value_vars=end_times, value_name="end_time", var_name="file")
    tmp1.file = tmp1.file.apply(lambda f: f.split("_")[-1])

    tmp2 = df.melt(id_vars=other_keys, value_vars=start_times, value_name="start_time", var_name="file")
    tmp2.file = tmp2.file.apply(lambda f: f.split("_")[-1])

    tmp1["start_time"] = tmp2["start_time"]
    df = tmp1
    df = df[df.file != "original"]
    df = df.dropna()
    df = df.reset_index(drop=True)
    df["start_time"] = df["start_time"].apply(pd.Timedelta.total_seconds)
    df["end_time"] = df["end_time"].apply(pd.Timedelta.total_seconds)
    df["filename"] = df.session_id + '_' + df.file
    df = df[["speaker", "end_time", "start_time", "filename"]]
    return df

def create_dataset(df):
    annotations = pd.DataFrame({
        "FILE": df.filename,
        "CHANNEL": pd.NA,
        "SPEAKER": df.speaker, 
        "ONSET":df.start_time, 
        "DURATION": (df.end_time - df.start_time).apply(lambda x: round(x,2)),
        "TYPE": "SPEAKER",
        "BLANK1": pd.NA,
        "BLANK2": pd.NA,
        "BLANK3": pd.NA
    })
    annotations = annotations[["TYPE", "FILE", "CHANNEL", "ONSET", "DURATION", "BLANK1", "BLANK2", "SPEAKER", "BLANK3"]]
    annotations = annotations[annotations["DURATION"] > 0].reset_index(drop=True)
    annotations = annotations.sort_values("ONSET")
    all_channels = []
    for filename in set(annotations.FILE.values):
        tmp = annotations[annotations.FILE == filename].reset_index(drop=True)
        if "U" in filename:
            tmp.CHANNEL = "CH1"
            all_channels.append(tmp.copy())
            tmp.CHANNEL = "CH2"
            all_channels.append(tmp.copy())
            tmp.CHANNEL = "CH3"
            all_channels.append(tmp.copy())
            tmp.CHANNEL = "CH4"
            all_channels.append(tmp.copy())

            #tmp.to_csv(folder / (filename + ".CH4" + ".rttm"), sep=" ", na_rep="<NA>", header=False, index=False)
        else:
            tmp.CHANNEL = "CH1"
            all_channels.append(tmp.copy())
            tmp.CHANNEL = "CH2"
            all_channels.append(tmp.copy())
    df = pd.concat(all_channels).reset_index(drop=True)
    mono = df[df.FILE.apply(lambda f: "U" in f)].reset_index(drop=True)
    mono.FILE = mono.FILE + "." + mono.CHANNEL
    mono.CHANNEL = "A"
    stereo = df[df.FILE.apply(lambda f: "P" in f)].reset_index(drop=True)
    stereo.CHANNEL = stereo.CHANNEL.apply(lambda x: channels_id[x])
    return pd.concat((mono, stereo)).reset_index(drop=True)

def create_rttm(df, folder):
    df = df.copy()
    filename = Path(folder)/"rttm"
    df.to_csv(filename, sep="\t", na_rep="<NA>", header=False, index=False, quoting=csv.QUOTE_NONE)

def create_reco2file_and_channel(df, folder):
    df = df.copy()
    filename = Path(folder) / "reco2file_and_channel"
    tmp = pd.DataFrame({
        "RECO": df.FILE.apply(lambda s: s.replace('.', '-').replace('_', '-')) + "-" + df.CHANNEL,
        "FILE": df.FILE,
        "CHANNEL": df.CHANNEL
    })
    tmp = tmp.drop_duplicates()
    tmp.to_csv(filename, sep="\t", na_rep="<NA>", header=False, index=False, quoting=csv.QUOTE_NONE)

def create_wavscp(df, folder, wav_folder):
    df = df.copy()
    dset = folder.split("/")[-1]
    filename = Path(folder) / "wav.scp"
    wav_folder = str(Path(wav_folder).resolve())
    mono = df[df.FILE.apply(lambda f: "U" in f)]
    stereo = df[df.FILE.apply(lambda f: "P" in f)]
    tmp1 = pd.DataFrame({
        "RECO": mono.FILE.apply(lambda s: s.replace('.', '-').replace('_', '-')) + "-" + mono.CHANNEL,
        "FILE":  wav_folder + "/" + mono.FILE + ".wav",
    })
    tmp2 = pd.DataFrame({
        "RECO": stereo.FILE.apply(lambda s: s.replace('.', '-').replace('_', '-')) + "-" + stereo.CHANNEL,
        "FILE": wav_folder + "/" + stereo.FILE + ".CH" + stereo.CHANNEL.apply(lambda s: channels_num[s]) + ".wav"
    })
    tmp = pd.concat((tmp1, tmp2))
    tmp = tmp.drop_duplicates()
    tmp.to_csv(filename, sep="\t", na_rep="<NA>", header=False, index=False, quoting=csv.QUOTE_NONE)


if __name__=="__main__":
    args = parser.parse_args()
    df = load_dataset(args.i)
    df = create_dataset(df)
    create_rttm(df, args.o)
    create_reco2file_and_channel(df, args.o)
    create_wavscp(df, args.o, args.wav)
