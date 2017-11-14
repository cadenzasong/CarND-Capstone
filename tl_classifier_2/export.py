import pickle
import os
import math
import numpy as np
import yaml
import scipy
import scipy.misc

def export(data_dir, data_desc):
    print("loading yaml: ", end='', flush=True)
    data_list = yaml.load(open(os.path.join(data_dir, data_desc), 'r'))
    print("done")
    crop_idx = 0
    for i in range(0, 3):
        for subdir in ['crop', 'res']:
            dir_ = '{}/{}'.format(subdir, i)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    images = []
    types = []
    for i, data in enumerate(data_list):
        path = os.path.join(data_dir, data['path'])
        boxes = data['boxes']
        image = scipy.misc.imread(path)
        for box in boxes:
            type_ = None
            if box['label'] == 'Red':
                type_ = 0
            elif box['label'] == 'Yellow':
                type_ = 1
            elif box['label'] == 'Green':
                type_ = 2
            else:
                continue
            x_min = int(math.floor(box['x_min']))
            x_max = int(math.ceil(box['x_max']))
            y_min = int(math.floor(box['y_min']))
            y_max = int(math.ceil(box['y_max']))
            crop = image[max(0, y_min):min(image.shape[0], y_max+1), max(0, x_min):min(image.shape[1], x_max+1)]
            scipy.misc.imsave('crop/{}/{}.png'.format(type_, crop_idx), crop)
            res = scipy.misc.imresize(crop, (96, 32))
            scipy.misc.imsave('res/{}/{}.png'.format(type_, crop_idx), res)
            images.append(res)
            types.append(type_)
            crop_idx += 1
        print(i + 1, ' / ', len(data_list), end='\r', flush=True)
    print('\n')
    with open('test.pickle', 'wb') as handle:
        pickle.dump((np.array(images), np.array(types)), handle)


export("../bosch-dataset/dataset_test_rgb/", "test.yaml")
