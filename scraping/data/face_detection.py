# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import argparse

cascade_path = './haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

def face_detect(directory_name):
    # 処理前のフォルダ
    input_dir = './imgs/'+directory_name+'/'
    # 処理後のフォルダ
    output_dir = input_dir[0:-1]+'_face/'
    os.makedirs(output_dir, exist_ok=True)

    file = os.listdir(input_dir)
    image_num = len(file)

    error = 0
    counter = 0
    file_num = 1
    error_max = 100
    while counter < image_num:
        try:
            img = cv2.imread(input_dir + str(file_num).zfill(4) + '.jpg', cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 画像がなければここでexceptへ
            face = faceCascade.detectMultiScale(gray, 1.1, 3)
            img2 = img.copy()
            if len(face) > 0:
                for face_num, rect in enumerate(face):
                    # 切り取り箇所表示（img2のコメントも取る）
                    os.makedirs('./imgs/detected_'+directory_name+'/', exist_ok=True)
                    cv2.rectangle(img2, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=1)
                    cv2.imwrite('./imgs/detected_'+directory_name+'/'+str(file_num)+'.jpg', img2)

                    print('-> Cutting image', str(file_num).zfill(4) + '_' + str(face_num+1) + '.jpg', end=' ')
                    x = rect[0]
                    y = rect[1]
                    w = rect[2]
                    h = rect[3]
                    cv2.imwrite(output_dir + str(file_num).zfill(4) + '_' + str(face_num+1) + '_cutted.jpg', img[y:y+h, x:x+w])
                    print('successful')
            else:
                print('-> ' + str(file_num).zfill(4) + '.jpg: No Face')
            counter += 1
            file_num += 1
        except:
            print('-> ' + str(file_num).zfill(4) + '.jpg: No image')
            file_num += 1
            error += 1
            if error > error_max:
                print('Probably unsupported format images are included.')
                break
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='cutting images location (following ./imgs/ )', type=str, required=True)
    args = parser.parse_args()
    directory_name = args.directory

    face_detect(directory_name)


if __name__ == '__main__':
    main()
