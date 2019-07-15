#画像の水増し
import os
import glob
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# 画像を拡張する関数
def draw_images(generator, x, dir_name, index, output_dir):
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=save_name, save_format='jpeg')

    # 1つの入力画像から何枚拡張するかを指定（今回は50枚）
    for i in range(50):
        bach = g.next()


def extend_img(directory_name):
    # 処理前のフォルダ
    input_dir = './imgs/'+directory_name+'/'
    # 処理後のフォルダ
    output_dir = input_dir[0:-1]+'_extend/'
    os.makedirs(output_dir, exist_ok=True)
    file = os.listdir(input_dir)
    image_num = len(file)

    datagen = ImageDataGenerator(rotation_range=30,
                                 width_shift_range=20,
                                 height_shift_range=0.,
                                 zoom_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=True)

    # # 読み込んだ画像を順に拡張
    for i in range(image_num):
        img = load_img(input_dir+file[i])
        img = img.resize((150, 150))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        draw_images(datagen, x, output_dir, i, output_dir)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='input images location (following ./imgs/ )', type=str, required=True)
    args = parser.parse_args()
    directory_name = args.directory
    extend_img(directory_name)

if __name__ == '__main__':
    main()
