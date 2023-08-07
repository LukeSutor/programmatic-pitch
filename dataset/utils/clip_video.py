import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


DATA_DIR = "D:/datasets/crop_2"
SAVE_DIR = "D:/datasets/cropped_2"


def extract_subclip(filename, time, i):
    '''
    Time formatting: mm:ss-mm:ss
    '''
    start = (60 * int(time.split("-")[0].split(":")[0])) + int(time.split("-")[0].split(":")[1])
    end = (60 * int(time.split("-")[1].split(":")[0])) + int(time.split("-")[1].split(":")[1])

    i_path = os.path.join(DATA_DIR, filename + ".wav")
    o_path = os.path.join(SAVE_DIR, filename + f"_{i}" + ".wav")

    ffmpeg_extract_subclip(i_path, start, end, o_path)


if __name__ == "__main__":
    filename = "Yasumu - Unravel🌷 [lofi hip hop⧸relaxing beats]"

    # "",
    times = [
        "0:00-2:08",
        "2:09-4:14",
        "4:15-6:24",
        "6:25-8:41",
        "8:42-11:32",
        "11:33-13:52",
        "13:55-16:08",
        "16:10-18:28",
        ]

    for i, time in enumerate(times):
        extract_subclip(filename, time, i)