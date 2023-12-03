import os
import random
import shutil

directory = "E:/Praca_inz/MusicGenreRecog/content/spectograms3sec/train"
genres = 'blues jazz classical country disco pop hiphop metal reggae rock'
genres = genres.split()

for g in genres:
    filenames = os.listdir(os.path.join(directory,f"{g}"))
    random.shuffle(filenames)
    test_files = filenames[0:100]

    for f in test_files:
        shutil.move(directory + "/" + f"{g}" + "/" + f, "E:/Praca_inz/MusicGenreRecog/content/spectograms3sec/test" + "/" + f"{g}")
