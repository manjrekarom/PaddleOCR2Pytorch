"""
Converts the json returned by tools/infer/predict_system.py (--output_json) to label 
studio formats.

The template is Optical Character Recognition with value of image changed to "$image" instead of "$ocr".

Depending upon the name/references made in label-studio template, one might have to
change the keys of the json. Arguments are provided for to_name, from_name and 
from_name for textarea.
"""
import os
import copy
import json
import argparse
import numpy as np

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", type=str)
    parser.add_argument("output_json_dir", type=str)
    parser.add_argument("--basepath", type=str, default=None)
    parser.add_argument("--from_name", type=str, default='label')
    parser.add_argument("--to_name", type=str, default='image')
    parser.add_argument("--from_name_text", type=str, default='transcription')
    return parser.parse_args()


def check_keys(dictionary, keys=['boxes', 'texts', 'scores', 'img_info']):
    return all(d_keys in keys for d_keys in dictionary)


def change_basepath(og_path, base_path):
    filename = os.path.basename(og_path)
    return os.path.join(base_path, filename)


def convert_box2xywh(box: np.ndarray, center=False):
    """
    Converts boxes (x1,y1,x2,y2,x3,y3,x4,y4) to rotated box (kx,ky,w,h,r) where
    kx, ky can be left top or center.
    """
    rbox = cv2.minAreaRect(box)
    cx, cy = rbox[0][0], rbox[0][1]
    w, h = rbox[1][0], rbox[1][1]
    x, y = cx - w/2.0, cy - h/2.0
    # shift center
    x_s, y_s = x - cx, y - cy
    # angle in radians
    angle = rbox[2] * np.pi / 180
    # perform rotation
    xd_s = x_s * np.cos(angle) - y_s * np.sin(angle)
    yd_s = y_s * np.cos(angle) + x_s * np.sin(angle)
    # reshift origin
    xd, yd = xd_s + cx, yd_s + cy
    if center:
        return rbox
    return ((xd, yd), (w, h), rbox[2])


# def convert_box2xywh(box: np.ndarray, center=False):
#     """
#     Converts boxes (x1,y1,x2,y2,x3,y3,x4,y4) to rotated box (kx,ky,w,h,r) where
#     kx, ky can be left top or center.
#     """
#     # get 0th coordinates
#     point0 = box[0]
#     rbox = cv2.minAreaRect(box)
#     x, y = point0
#     w, h = rbox[1][0], rbox[1][1]
#     if center:
#         return rbox
#     return ((x, y), (w, h), rbox[2])
 

def normalize_rbox(rbox, im_width, im_height):
    """
    Converts rbox x,y,w,h to percantages of image width and height.
    """
    x, y = rbox[0][0], rbox[0][1]
    w, h = rbox[1][0], rbox[1][1]
    nrbox = ((x/im_width*100, y/im_height*100), (w/im_width*100, h/im_height*100), rbox[2])
    return nrbox


def make_ls_result(result, from_name, to_name, from_name_text):
    """
    Convert to label studio result format.

    {
        "id": "result1",
        "type": "rectanglelabels",
        "from_name": "label",
        "to_name": "image",
        "original_width": 4128,
        "original_height": 3096,
        "image_rotation": 0,
        "value": {
            "rotation": 2.70222,
            "x": 40.81,
            "y": 31.76,
            "width": 8.23,
            "height": 3.1033,
            "rectanglelabels": [
                "Airplane"
            ]
        }
    }
    """
    ls_results = []
    for idx, rbox in enumerate(result['rboxes']):
        det_result = {}
        det_result['id'] = 'result' + str(idx + 1)
        det_result['type'] = 'rectanglelabels'
        det_result['from_name'] = from_name
        det_result['to_name'] = to_name
        det_result['original_width'] = result['img_info']['width']
        det_result['original_height'] = result['img_info']['height']
        det_result['image_rotation'] = 0
        det_result['value'] = {}
        det_result['value']['rotation'] = rbox[2]
        # ls_result['value']['transcription'] = result['texts'][idx]
        det_result['value']['x'] = rbox[0][0]
        det_result['value']['y'] = rbox[0][1]
        det_result['value']['width'] = rbox[1][0]
        det_result['value']['height'] = rbox[1][1]
        det_result['value']['rectanglelabels'] = []
        ls_results.append(det_result)
        # add result for the transcription
        rec_result = copy.deepcopy(det_result)
        rec_result['type'] = 'textarea'
        rec_result['from_name'] = from_name_text
        rec_result['to_name'] = to_name
        rec_result['value']['text'] = [result['texts'][idx]]
        ls_results.append(rec_result)
    return ls_results


def make_ls_json(image_path, ls_results, score):
    ls_json = []
    ls_dict = {}
    ls_json.append(ls_dict)
    ls_dict['data'] = {}
    ls_dict['data']['image'] = image_path
    ls_dict['predictions'] = []
    ls_dict_pred = {}
    ls_dict['predictions'].append(ls_dict_pred)
    ls_dict_pred['result'] = ls_results
    ls_dict_pred['score'] = score
    return ls_json


def main(args):
    json_dir = args.json_dir
    if os.path.isfile(json_dir):
        json_dir, filename = os.path.split(json_dir)
        json_files = [filename]
    else:
        json_files = [file for file in os.listdir(json_dir) if file.endswith('.json')]
    for filename in json_files:
        with open(os.path.join(json_dir, filename)) as f:
            result = json.load(f)
        # checks
        if not check_keys(result):
            print(f'{filename} doesn\'t contain required keys.')
            continue
        if len(result['boxes']) < 1:
            print(f'{filename} doesn\'t contain boxes.')
            continue
        filepath = result['img_info']['filepath']
        if args.basepath is not None:
            filepath = change_basepath(result['img_info']['filepath'], args.basepath)
        rboxes = []
        for box in result['boxes']:
            rbox = convert_box2xywh(np.array(box, dtype=np.float32))
            width, height = result['img_info']['width'], result['img_info']['height']
            nrbox = normalize_rbox(rbox, width, height)
            rboxes.append(nrbox)
        result['rboxes'] = rboxes
        ls_results = make_ls_result(result, args.from_name, args.to_name, 
        args.from_name_text)
        score = np.array(result['scores']).mean()
        ls_json = make_ls_json(filepath, ls_results, score)
        # make output directory
        if not os.path.exists(args.output_json_dir):
            os.makedirs(args.output_json_dir)
        output_path = os.path.join(args.output_json_dir, filename)
        with open(output_path, 'w+') as f:
            json.dump(ls_json, f)


if __name__ == "__main__":
    main(parse_args())
