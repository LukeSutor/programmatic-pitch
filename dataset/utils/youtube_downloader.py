from __future__ import unicode_literals
import youtube_dl

URL = 'https://www.youtube.com/playlist?list=PLofht4PTcKYnaH8w5olJCI-wUVxuoMHqM'

def download():
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': '../youtube_clips/%(title)s.%(ext)s',
        'playliststart': 1,
        'playlistend': 500
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([URL])
    except Exception as e:
        print(e)


if __name__ == "__main__":
    download()


    print("fin.")