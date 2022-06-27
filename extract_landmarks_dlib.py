import os
import cv2
import csv
import dlib
import numpy as np
from tqdm import tqdm

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')
images_dir_paths = [
    r'./frames/D4_v2/fake/deepfacelab/1st_experiment/obama_1m',
    # r'./frames/D4_v2/fake/deepfacelab/1st_experiment/obama_2m',
    # r'./frames/D4_v2/fake/deepfacelab/1st_experiment/obama_3m',
    # r'./frames/D4_v2/fake/deepfacelab/1st_experiment/obama_many2many',
    # r'./frames/D4_v2/fake/deepfacelab/2nd_experiment/biden_200',
    # r'./frames/D4_v2/fake/deepfacelab/2nd_experiment/obama_200',
    # r'./frames/D4_v2/fake/deepfacelab/3rd_experiment/biden_da',
    # r'./frames/D4_v2/fake/deepfacelab/3rd_experiment/obama_da',
    # r'./frames/D4_v2/fake/fom/biden_fom',
    # r'./frames/D4_v2/fake/fom/obama_sixty_fom',
    # r'./frames/D4_v2/fake/fom/obama_wage_fom',
    # r'./frames/D4_v2/real/biden',
    # r'./frames/D4_v2/real/obama_sixty',
    # r'./frames/D4_v2/real/obama_wage',
    # r'./frames/D3/fake/deepfacelab',
    # r'./frames/D3/fake/fom',
    # r'./frames/D3/real/hanwei'
]

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords


if __name__ == '__main__':

    for i, image_dir in enumerate(images_dir_paths):
        print(f'>> {i+1}/{len(images_dir_paths)}:', image_dir)
        landmark_dir = os.path.join(image_dir.replace('frames', 'landmarks'), 'dlib')
        if not os.path.exists(landmark_dir): os.makedirs(landmark_dir)
        
        for image_name in tqdm(os.listdir(image_dir)):
            image_path = os.path.join(image_dir, image_name)
            landmark_path = os.path.join(landmark_dir, image_name.split('.')[0]+'.txt')
            image = cv2.imread(image_path)
            rects = detector(image, 1)
            if len(rects) < 1: 
                with open(r'./results/dlib_face_not_found_v3.csv','a') as fd:
                    writer=csv.writer(fd)
                    writer.writerow([os.path.basename(image_dir), image_name])
                continue
            shape = predictor(image, rects[0])
            shape_np = shape_to_np(shape)
            np.savetxt(landmark_path, shape_np, fmt="%d")
