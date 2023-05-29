import yt_dlp


def download(URLs):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': 'D:/datasets/lofi/%(title)s.%(ext)s',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(URLs)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    links = [
        "https://www.youtube.com/@LofiRecords/videos",
        "https://www.youtube.com/@thebootlegboy/videos"
    ]

    download(links)
    # for i in range(1, 990, 5):
    #     download(i, 'https://www.youtube.com/playlist?list=PLofht4PTcKYnaH8w5olJCI-wUVxuoMHqM')

    # for i in range(1, 870, 5):
    #     download(i, 'https://www.youtube.com/playlist?list=PLOzDu-MXXLliO9fBNZOQTBDddoA3FzZUo')

    print("fin.")