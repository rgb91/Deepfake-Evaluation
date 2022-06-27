import os
import csv
import numpy as np
from tqdm import tqdm

LANDMARK_MODE = 'insightface'  # options: 'dlib', 'insightface'
# normalizing_factor = 106 if LANDMARK_MODE=='insightface' else 68  # number of landmark points
normalizing_factor = 1
result_file = f'./results/results_{LANDMARK_MODE}_obama1m.csv'
dir_map = [
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D3/fake/deepfacelab', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D3/real/hanwei', LANDMARK_MODE),
    #     'remark': 'd3'
    # },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D3/fake/fom', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D3/real/hanwei', LANDMARK_MODE),
    #     'remark': 'd3'
    # },
    {
        'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/deepfacelab/1st_experiment/obama_1m', LANDMARK_MODE), 
        'real_dir': os.path.join(r'./landmarks/D4_v2/real/obama_sixty', LANDMARK_MODE),
        'remark': 'd4v2_dfl_exp1'
    },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/deepfacelab/1st_experiment/obama_2m', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D4_v2/real/obama_sixty', LANDMARK_MODE),
    #     'remark': 'd4v2_dfl_exp1'
    # },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/deepfacelab/1st_experiment/obama_3m', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D4_v2/real/obama_sixty', LANDMARK_MODE),
    #     'remark': 'd4v2_dfl_exp1'
    # },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/deepfacelab/1st_experiment/obama_many2many', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D4_v2/real/obama_sixty', LANDMARK_MODE),
    #     'remark': 'd4v2_dfl_exp1'
    # },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/deepfacelab/2nd_experiment/obama_200', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D4_v2/real/obama_wage', LANDMARK_MODE),
    #     'remark': 'd4v2_dfl_exp2'
    # },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/deepfacelab/2nd_experiment/biden_200', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D4_v2/real/biden', LANDMARK_MODE),
    #     'remark': 'd4v2_dfl_exp2'
    # },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/deepfacelab/3rd_experiment/obama_da', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D4_v2/real/obama_wage', LANDMARK_MODE),
    #     'remark': 'd4v2_dfl_exp3'
    # },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/deepfacelab/3rd_experiment/biden_da', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D4_v2/real/biden', LANDMARK_MODE),
    #     'remark': 'd4v2_dfl_exp3'
    # },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/fom/obama_sixty_fom', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D4_v2/real/obama_sixty', LANDMARK_MODE),
    #     'remark': 'd4v2_fom'
    # },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/fom/obama_wage_fom', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D4_v2/real/obama_wage', LANDMARK_MODE),
    #     'remark': 'd4v2_fom'
    # },
    # {
    #     'fake_dir': os.path.join(r'./landmarks/D4_v2/fake/fom/biden_fom', LANDMARK_MODE), 
    #     'real_dir': os.path.join(r'./landmarks/D4_v2/real/biden', LANDMARK_MODE),
    #     'remark': 'd4v2_fom'
    # },
]

if __name__ == '__main__':
    for d_map in dir_map:
        fake_dir, real_dir, remark = d_map['fake_dir'], d_map['real_dir'], d_map['remark']
        
        print(f'\n>> {os.path.basename(os.path.dirname(fake_dir))} <> {os.path.basename(os.path.dirname(real_dir))}')

        sum, count = 0, 0
        for f1_name in os.listdir(fake_dir):
            f1_path = os.path.join(fake_dir, f1_name)
            f2_path = os.path.join(real_dir, f1_name)
            
            if not os.path.exists(f2_path): 
                print(f'Skipped \t{f1_name}')
                continue

            shape1, shape2 = np.genfromtxt(f1_path), np.genfromtxt(f2_path)
            dist = (np.linalg.norm((shape1[:,0] - shape2[:,0]), ord=1) + np.linalg.norm((shape1[:,1] - shape2[:,1]), ord=1)) / normalizing_factor
            sum = sum + dist
            count = count + 1
        final_l1_dist = sum/count if count>0 else -1
        print(f'.. {os.path.basename(os.path.dirname(fake_dir))} <> {os.path.basename(os.path.dirname(real_dir))} -> {final_l1_dist}')

        try:
            with open(result_file, 'a') as fd:
                writer=csv.writer(fd)
                writer.writerow([remark, os.path.basename(os.path.dirname(fake_dir)), os.path.basename(os.path.dirname(real_dir)), final_l1_dist])
            print(f'>> Done.')
        except Exception as e:
            print('Error in writing file.')
            if hasattr(e, 'message'): print(e.message)
