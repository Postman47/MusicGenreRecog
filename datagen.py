from keras_preprocessing.image import ImageDataGenerator

train_dir = "E:/Praca_inz/MusicGenreRecog/content/spectograms3sec/train/"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,  target_size=(288,432), color_mode="rgba", class_mode='categorical', batch_size=128)

validation_dir = "E:/Praca_inz/MusicGenreRecog/content/spectograms3sec/test/"
vali_datagen = ImageDataGenerator(rescale=1./255)
vali_generator = train_datagen.flow_from_directory(validation_dir,  target_size=(288,432), color_mode="rgba", class_mode='categorical', batch_size=128)