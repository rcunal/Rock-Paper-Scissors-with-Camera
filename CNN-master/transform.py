from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
import os

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Generates an array with the images from the folder:
path = 'Dataset'
class_list = os.listdir(path)

image_names = []
labels = []
for class_name in class_list:
    image_list = os.listdir(os.path.join(path, class_name))
    img_dir = 'Dataset\\' + class_name  # the image directory
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        data.append(f1)
    for j in data:
        img = load_img(j)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (300, 300, 3)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 300, 300, 3)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        # the loop makes 5 transformations of each image, this can be changed setting the if i > 5: to any other number you wish

        i = 0
        # the new folder directory is added in the save_to_dir:
        for batch in datagen.flow(x, batch_size=1, save_to_dir='Dataset\\' + class_name, save_prefix='el', save_format='jpeg'):
            i += 1
            if i > 5:
                break  # otherwise the generator would loop indefinitely


