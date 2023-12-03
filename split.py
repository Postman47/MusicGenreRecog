import os
import wave

from pydub import AudioSegment

genres = 'blues jazz classical country disco pop hiphop metal reggae rock'
genres = genres.split()

i=0
for g in genres:
    j=0
    print(f"{g}")
    for filename in os.listdir(os.path.join('E:/Praca_inz/MusicGenreRecog/content/genres',f"{g}")):

        song = os.path.join(f'E:/Praca_inz/MusicGenreRecog/content/genres/{g}',f'{filename}')
        j=j+1
        for w in range(0,10):
            i=i+1
            t1 = 3*(w)*1000
            t2 = 3*(w+1)*1000
            newAudio = AudioSegment.from_wav(song)
            new = newAudio[t1:t2]
            new.export(f'E:/Praca_inz/MusicGenreRecog/content/audio3sec/{g}/{g+str(j)+str(w)}.wav', format = "wav")