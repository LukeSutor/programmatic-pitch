import youtube_dl


def download(index, URL):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': '../youtube_clips/%(title)s.%(ext)s',
        'playliststart': index,
        'playlistend': index + 4
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([URL])
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Download the playlist in batches of five so that the script
    # doesn't fail and stop downloading everything
    for i in range(1, 990, 5):
        download(i, 'https://www.youtube.com/playlist?list=PLofht4PTcKYnaH8w5olJCI-wUVxuoMHqM')

    for i in range(1, 870, 5):
        download(i, 'https://www.youtube.com/playlist?list=PLOzDu-MXXLliO9fBNZOQTBDddoA3FzZUo')


    print("fin.")