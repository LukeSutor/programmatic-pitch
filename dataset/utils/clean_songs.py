import shutil
import whisper
import os


def check_song(filename: str, bad_names: list[str]) -> bool:
    '''
    Return true if song name is bad, false otherwise
    '''
    for name in bad_names:
        if filename.find(name) != -1:
            return True
        
    return False


def get_subtitles(model, path: str, filename: str) -> list[str]:
    '''
    Use openai Whisper to transcribe songs
    '''
    result = model.transcribe(os.path.join(path, filename))
    return result["text"]


if __name__ == "__main__":
    model = whisper.load_model("tiny.en")
    bad_names = [
        "Live Performance",
        "Behind the scene",
        "Lofi live show",
        "Beat breakdown",
        "Live Session",
        "Tips/tricks",
        "Q&A",
        "Chill Lofi Beats"
    ]
    
    data_dir = "D:/datasets/lofi"
    check_dir = "D:/datasets/check"

    files = os.listdir(data_dir)
    n = len(files)

    for i, file in enumerate(files):
        sub = get_subtitles(model, data_dir, file)
        if check_song(file, bad_names) or len(sub.split(" ")) >= 50:
            # Move to check dir
            shutil.move(os.path.join(data_dir, file), os.path.join(check_dir, file))
        print(i, "/", n)

    print("fin.")