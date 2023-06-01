import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


DATA_DIR = "D:/datasets/crop"
SAVE_DIR = "D:/datasets/cropped"


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
    filename = "Distant Space 🌙 Deep Chill Beats"

    # "",
    times = [
        "1:41-3:50",
        "3:51-6:28",
        "11:42-14:08",
        "14:09-17:23",
        "17:25-19:55",
        "19:56-22:33",
        "22:34-23:47",
        "23:49-26:06",
        "26:07-28:38",
        "33:01-35:10",
        "35:11-37:48",
        "43:02-45:28",
        "45:29-48:42",
        "48:44-51:14",
        "51:15-53:52",
        "53:53-55:07",
        "55:08-57:27",
        "57:28-59:58",
    ]

    for i, time in enumerate(times):
        extract_subclip(filename, time, i)

