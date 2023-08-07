import os
import torchaudio
import collections

DATA_DIR = "D:/datasets/lofi_dataset/beats"

def main():
    total_samples = 0
    srs = collections.defaultdict(int)

    for i, filename in enumerate(os.listdir(DATA_DIR)):
        wave, sr = torchaudio.load(os.path.join(DATA_DIR, filename))
        srs[sr] += 1
        total_samples += wave.size()[1]
        print(i)

    total_time = total_samples / 48000
    dataset_length = len(os.listdir(DATA_DIR))

    print("Dataset length:", dataset_length)
    print("Total samples:", total_samples)
    print("Sampling rates:", srs)
    print("Total time (s):", total_time)
    print("Average song length (s):", (total_time / dataset_length))

if __name__ == "__main__":
    main()