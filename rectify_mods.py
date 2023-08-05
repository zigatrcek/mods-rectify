import copy
import glob
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_calibration(filename: str):
    assert os.path.isfile(filename)

    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    M1 = fs.getNode("M1").mat()
    M2 = fs.getNode("M2").mat()
    D1 = fs.getNode("D1").mat()
    D2 = fs.getNode("D2").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    width = int(fs.getNode("imageSize").at(0).real())
    height = int(fs.getNode("imageSize").at(1).real())
    return M1, M2, D1, D2, R, T, width, height


def rectify_obstacle(obstacle: dict, M, D, R, P, mode):
    new_obstacle = copy.deepcopy(obstacle)
    new_obstacle['bbox'] = []
    new_obstacle['area'] = None

    if mode == 'mods':
        x1, y1, w, h = obstacle['bbox']
        x2 = x1 + w
        y2 = y1 + h
    else:
        x1, y1, x2, y2 = obstacle['bbox']
        w = x2 - x1
        h = y2 - y1

    points = np.array([[x1, y1], [x2, y2]]).astype(np.float32)
    remapped = cv2.undistortPoints(np.expand_dims(
        points, axis=1), M, D, R=R, P=P).squeeze()
    new_bbox = np.rint(remapped).astype(int)

    new_x1 = new_bbox[0, 0]
    new_y1 = new_bbox[0, 1]
    new_x2 = new_bbox[1, 0]
    new_y2 = new_bbox[1, 1]

    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1

    if mode == 'mods':
        new_bbox = [new_x1, new_y1, new_w, new_h]
    else:
        new_bbox = [new_x1, new_y1, new_x2, new_y2]

    new_obstacle['bbox'] = new_bbox
    new_obstacle['area'] = new_w * new_h
    return new_obstacle


def rectify_water_edge(edge: dict, M, D, R, P):
    x_axis = edge['x_axis']
    y_axis = edge['y_axis']
    new_water_edges_dict = copy.deepcopy(edge)
    new_water_edges_dict['x_axis'] = []
    new_water_edges_dict['y_axis'] = []

    zipped = np.array(list(zip(x_axis, y_axis))).astype(np.float32)
    remapped = cv2.undistortPoints(np.expand_dims(
        zipped, axis=1), M, D, R=R, P=P).squeeze()
    new_water_edges = np.rint(remapped).astype(int)
    new_water_edges_x = new_water_edges[:, 0]
    new_water_edges_y = new_water_edges[:, 1]
    new_water_edges_dict['x_axis'] = new_water_edges_x.tolist()
    new_water_edges_dict['y_axis'] = new_water_edges_y.tolist()
    return new_water_edges_dict


def main():
    config_path = sys.argv[1]
    config = yaml.safe_load(open(config_path))

    output_imgs = config['output_imgs']
    visualise = config['visualise']
    dataset_path = config['dataset_path']
    out_path = config['out_path']



    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    sequences = sorted(glob.glob(dataset_path+'sequences/*'))

    annotations_file = os.path.join(dataset_path, 'mods.json')
    annotations = json.load(open(annotations_file, 'r'))
    ann_sequences = annotations['dataset']['sequences']
    new_annotations = copy.deepcopy(annotations)
    new_annotations['dataset']['sequences'] = []

    dextr_coverage_file = os.path.join(dataset_path, 'dextr_coverage.json')
    dextr_coverage = json.load(open(dextr_coverage_file, 'r'))
    new_dextr_coverage = copy.deepcopy(dextr_coverage)
    new_dextr_coverage['sequences'] = []
    dextr_sequences = dextr_coverage['sequences']

    for seq, ann, dextr in zip(sequences, ann_sequences, dextr_sequences):
        assert seq.split(
            '/')[0] == ann['path'].split('/')[-1], f'Sequence mismatch: {seq} != {ann["path"]}'
        assert len(ann['frames']) == len(
            dextr['frames']), f'Frame mismatch: {len(ann["frames"])} != {len(dextr["frames"])}'

        new_ann = copy.deepcopy(ann)
        new_ann['frames'] = []

        new_dextr = copy.deepcopy(dextr)
        new_dextr['frames'] = []

        seq_base = seq.split('/')[-1].split('-')[0]

        calib_file = os.path.join(dataset_path, 'calibration', f'calibration-{seq_base}.yaml')
        M1, M2, D1, D2, R, T, width, height = load_calibration(calib_file)

        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            M1, D1, M2, D2, (width, height), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        mapL, map_int_l = cv2.initUndistortRectifyMap(
            M1, D1, R1, P1, (width, height), cv2.CV_16SC2)
        mapR, map_int_r = cv2.initUndistortRectifyMap(
            M2, D2, R2, P2, (width, height), cv2.CV_16SC2)

        images = sorted(glob.glob(f'{seq}/frames/*L.jpg'))

        for fn_left, img_ann, img_dextr in zip(images, ann['frames'], dextr['frames']):
            ann_name = img_ann['image_file_name'][:-5]
            name = fn_left.split('/')[-1][:-5]

            assert ann_name == name, f'Ann name: {ann_name} != name: {name}!'

            new_frame = copy.deepcopy(img_ann)
            new_frame['water_edges'] = []
            new_frame['obstacles'] = []

            for edge in img_ann['water_edges']:
                new_water_edges_dict = rectify_water_edge(edge, M1, D1, R1, P1)
                new_frame['water_edges'].append(new_water_edges_dict)

            for obstacle in img_ann['obstacles']:
                new_obstacle = rectify_obstacle(
                    obstacle, M1, D1, R1, P1, mode='mods')
                new_frame['obstacles'].append(new_obstacle)

            new_ann['frames'].append(new_frame)

            new_dextr_frame = copy.deepcopy(img_dextr)
            new_dextr_frame['obstacles'] = []

            for obstacle in img_dextr['obstacles']:
                new_obstacle = rectify_obstacle(
                    obstacle, M1, D1, R1, P1, mode='dextr')
                new_dextr_frame['obstacles'].append(new_obstacle)

            new_dextr['frames'].append(new_dextr_frame)

            fn_right = os.path.join(seq, 'frames', f'{name}R.jpg')

            im_l = cv2.imread(fn_left)
            im_r = cv2.imread(fn_right)

            im_l = cv2.remap(im_l, mapL, map_int_l, cv2.INTER_LINEAR)
            im_r = cv2.remap(im_r, mapR, map_int_r, cv2.INTER_LINEAR)

            if visualise:
                if len(new_frame['obstacles']) > 0 and len(new_frame['water_edges']) > 0:
                    # test obstacle rectification
                    for obstacle in new_frame['obstacles']:
                        x1, y1, w, h = obstacle['bbox']
                        print(f'Obstacle: {x1}, {y1}, {w}, {h}')
                        # draw the rectangle
                        cv2.rectangle(im_l, (x1, y1),
                                      (x1+w, y1+h), (0, 255, 0), 2)

                    # test water edge rectification
                    for edge in new_frame['water_edges']:
                        x_axis = edge['x_axis']
                        y_axis = edge['y_axis']
                        for x, y in zip(x_axis, y_axis):
                            cv2.circle(im_l, (x, y), 1, (0, 0, 255), 2)

                    # test dextr obstacle rectification
                    for obstacle in new_dextr_frame['obstacles']:
                        x1, y1, x2, y2 = obstacle['bbox']
                        w = x2 - x1
                        h = y2 - y1
                        print(f'Obstacle: {x1}, {y1}, {w}, {h}')
                        # draw the rectangle
                        cv2.rectangle(im_l, (x1, y1),
                                      (x1+w, y1+h), (255, 0, 0), 2)

                    plt.imshow(im_l)
                    plt.show()

            if output_imgs:
                out_dir = os.path.join(out_path, 'sequences', f'{seq.split("/")[-1]}', 'frames')
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)

                cv2.imwrite(os.path.join(out_dir, f'{name}L.jpg'), im_l)
                cv2.imwrite(os.path.join(out_dir, f'{name}R.jpg'), im_r)

        new_annotations['dataset']['sequences'].append(new_ann)
        new_dextr_coverage['sequences'].append(new_dextr)
        print(f'Converted sequence {seq}')

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    with open(os.path.join(out_path, 'mods.json'), 'w') as outfile:
        json.dump(new_annotations, outfile, indent=4, cls=NpEncoder)

    with open(os.path.join(out_path, 'dextr_coverage.json'), 'w') as outfile:
        json.dump(new_dextr_coverage, outfile, indent=4, cls=NpEncoder)


if __name__ == '__main__':
    main()
