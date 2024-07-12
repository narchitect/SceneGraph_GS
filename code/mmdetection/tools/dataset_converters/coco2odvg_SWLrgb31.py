import argparse
import json
import os.path

import jsonlines
from pycocotools.coco import COCO
from tqdm import tqdm

custom_id_map = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21,
    21: 22,
    22: 23,
    23: 24,
    24: 25,
    25: 26,
    26: 27
}

custom_ori_map = {
    '1': 'wall',
    '2': 'ceiling',
    '3': 'lighting',
    '4': 'speaker',
    '5': 'door',
    '6': 'smoke alarm',
    '7': 'floor',
    '8': 'trash bin',
    '9': 'elevator button',
    '10': 'escape sign',
    '11': 'board',
    '12': 'fire extinguisher',
    '13': 'door sign',
    '14': 'light switch',
    '15': 'emergency switch button',
    '16': 'elevator',
    '17': 'handrail',
    '18': 'show window',
    '19': 'pipes',
    '20': 'staircase',
    '21': 'window',
    '22': 'radiator',
    '23': 'stecker',
    '24': 'monitor',
    '25': 'whiteboard',
    '26': 'chair',
    '27': 'table'
}

key_list_custom = list(custom_id_map.keys())
val_list_custom = list(custom_id_map.values())

def dump_custom_label_map(args):
    new_map = {}
    for key, value in custom_ori_map.items():
        label = int(key)
        ind = val_list_custom.index(label)
        label_trans = key_list_custom[ind]
        new_map[label_trans] = value
    if args.output is None:
        output = os.path.dirname(args.input) + '/custom_label_map.json'
    else:
        output = os.path.dirname(args.output) + '/custom_label_map.json'
    with open(output, 'w') as f:
        json.dump(new_map, f)

def coco2odvg(args):
    coco = COCO(args.input)
    cats = coco.loadCats(coco.getCatIds())
    nms = {cat['id']: cat['name'] for cat in cats}
    metas = []
    if args.output is None:
        out_path = args.input[:-5] + '_od.json'
    else:
        out_path = args.output

    key_list = key_list_custom
    val_list = val_list_custom
    dump_custom_label_map(args)

    for img_id, img_info in tqdm(coco.imgs.items()):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        instance_list = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue

            if ann.get('iscrowd', False):
                continue

            bbox_xyxy = [x1, y1, x1 + w, y1 + h]
            label = ann['category_id']
            category = nms[label]
            ind = val_list.index(label)
            label_trans = key_list[ind]
            instance_list.append({
                'bbox': bbox_xyxy,
                'label': label_trans,
                'category': category
            })
        metas.append({
            'filename': img_info['file_name'],
            'height': img_info['height'],
            'width': img_info['width'],
            'detection': {
                'instances': instance_list
            }
        })

    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(metas)

    print('save to {}'.format(out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('coco to odvg format.', add_help=True)
    parser.add_argument('input', type=str, help='input json file name')
    parser.add_argument(
        '--output', '-o', type=str, help='output json file name')
    parser.add_argument(
        '--dataset',
        '-d',
        required=True,
        type=str,
        choices=['custom'],
    )
    args = parser.parse_args()

    coco2odvg(args)
