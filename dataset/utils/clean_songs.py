import shutil
# import whisper
import os
import csv


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


def rename(path: str):
    '''
    Renames files and creates a CSV with old names and new names
    '''
    names = []
    for i, file in enumerate(os.listdir(path)):
        old_name = file
        new_name = "{:04d}".format(i + 1) + ".wav"
        names.append([old_name, new_name])
        # Rename file
        os.rename(os.path.join(path, old_name), os.path.join(path, new_name))

    with open('filenames.csv', 'w', newline='\n', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["old_name", "new_name"])          
        writer.writerows(names)


if __name__ == "__main__":
    # model = whisper.load_model("tiny.en")
    # bad_names = [
    #     "Live Performance",
    #     "Behind the scene",
    #     "Lofi live show",
    #     "Beat breakdown",
    #     "Live Session",
    #     "Tips/tricks",
    #     "Q&A",
    #     "Chill Lofi Beats"
    # ]
    
    data_dir = "D:/datasets/lofi_3"
    # check_dir = "D:/datasets/check"

    rename(data_dir)

    # files = os.listdir(data_dir)
    # n = len(files)

    # for i, file in enumerate(files):
    #     sub = get_subtitles(model, data_dir, file)
    #     if check_song(file, bad_names) or len(sub.split(" ")) >= 50:
    #         # Move to check dir
    #         shutil.move(os.path.join(data_dir, file), os.path.join(check_dir, file))
    #     print(i, "/", n)

    # print("fin.")