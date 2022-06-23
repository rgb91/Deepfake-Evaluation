import os
import csv
import numpy as np
from tqdm import tqdm

f_base_dir = r'landmarks/D4_v2/fake'
r_base_dir = r'landmarks/D4_v2/real'
dir_map = [
    {
        'fake_dir': os.path.join(f_base_dir, 'fom/obama_sixty_fom/dlib'), 
        'real_dir': os.path.join(r_base_dir, 'obama_sixty/dlib'),
        'remark': 'd4v2_fom'
    },
    {
        'fake_dir': os.path.join(f_base_dir, 'fom/obama_wage_fom/dlib'), 
        'real_dir': os.path.join(r_base_dir, 'obama_wage/dlib'),
        'remark': 'd4v2_fom'
    },
    {
        'fake_dir': os.path.join(f_base_dir, 'fom/biden_fom/dlib'), 
        'real_dir': os.path.join(r_base_dir, 'biden/dlib'),
        'remark': 'd4v2_fom'
    },
]
result_file = r'./results/results_d4v2_dlib_all.csv'

if __name__ == '__main__':
    for d_map in dir_map:
        fake_dir, real_dir, remark = d_map['fake_dir'], d_map['real_dir'], d_map['remark']
        
        print(f'\n>> {os.path.basename(fake_dir)} <> {os.path.basename(real_dir)}')

        sum, count = 0, 0
        for f1_name in os.listdir(fake_dir):
            f1_path = os.path.join(fake_dir, f1_name)
            f2_path = os.path.join(real_dir, f1_name)
            
            if not os.path.exists(f1_path) or not os.path.exists(f2_path): 
                print(f'Skipped \t{f1_name}')
                continue

            shape1, shape2 = np.genfromtxt(f1_path), np.genfromtxt(f2_path)
            dist = np.linalg.norm((shape1[:,0] - shape2[:,0]), ord=1) + np.linalg.norm((shape1[:,1] - shape2[:,1]), ord=1)
            sum = sum + dist
            count = count + 1
        final_l1_dist = sum/count if count>0 else -1
        print(f'\t... {os.path.basename(os.path.dirname(fake_dir))} <> {os.path.basename(os.path.dirname(real_dir))} -> {final_l1_dist}')

        try:
            with open(result_file, 'a') as fd:
                writer=csv.writer(fd)
                writer.writerow([os.path.basename(fake_dir), os.path.basename(real_dir), final_l1_dist])
            print(f'<< {os.path.basename(os.path.dirname(fake_dir))} <> {os.path.basename(os.path.dirname(real_dir))} Done.')
        except Exception as e:
            print('Error in writing file.')
            if hasattr(e, 'message'): print(e.message)
