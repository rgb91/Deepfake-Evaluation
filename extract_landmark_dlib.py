import os
import cv2
import csv
import dlib
import numpy as np
from tqdm import tqdm
from threading import Thread

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')
frames_path = r'./frames'
landmarks_path = r'./landmarks-dlib'


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords


if __name__ == '__main__':
    for rf_dir_name in os.listdir(frames_path):
        rf_dir_path = os.path.join(frames_path, rf_dir_name)

        for image_dir_name in os.listdir(rf_dir_path):
            image_dir_path = os.path.join(rf_dir_path, image_dir_name)
            if not os.path.isdir(image_dir_path): continue
            
            lmk_dir_path = os.path.join(landmarks_path, rf_dir_name, image_dir_name+'_lmk')
            if not os.path.exists(lmk_dir_path): os.mkdir(lmk_dir_path)
            
            for image_name in tqdm(os.listdir(image_dir_path)):
                image_path = os.path.join(image_dir_path, image_name)
                l_txt_path = os.path.join(lmk_dir_path, image_name.split('.')[0]+'.txt')
                image = cv2.imread(image_path)
                rects = detector(image, 1)
                if len(rects) < 1: 
                    with open('dlib_face_not_found_v2.csv','a') as fd:
                        writer=csv.writer(fd)
                        writer.writerow([os.path.basename(image_dir_path), image_name])
                    continue
                shape = predictor(image, rects[0])
                shape_np = shape_to_np(shape)
                np.savetxt(l_txt_path, shape_np, fmt="%d")
