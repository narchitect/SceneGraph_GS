_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

o365v1_od_dataset = dict(
    type='ODVGDataset',
    data_root='data/objects365v1/',
    ann_file='o365v1_train_odvg.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None,
)

flickr30k_dataset = dict(
    type='ODVGDataset',
    data_root='data/flickr30k_entities/',
    ann_file='final_flickr_separateGT_train_vg.json',
    label_map_file=None,
    data_prefix=dict(img='flickr30k_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

gqa_dataset = dict(
    type='ODVGDataset',
    data_root='data/gqa/',
    ann_file='final_mixed_train_no_coco_vg.json',
    label_map_file=None,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

v3d_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        # change this
        label_map_file='data/V3Det/annotations/v3det_2023_v1_label_map.json',
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]
v3det_dataset = dict(
    type='ODVGDataset',
    data_root='data/V3Det/',
    ann_file='annotations/v3det_2023_v1_train_od.json',
    label_map_file='annotations/v3det_2023_v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,  # change this
    pipeline=v3d_train_pipeline,
    return_classes=True,
    backend_args=None)

grit_dataset = dict(
    type='ODVGDataset',
    data_root='grit_processed/',
    ann_file='grit20m_vg.json',
    label_map_file=None,
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

train_dataloader = dict(
    sampler=dict(
        _delete_=True,
        type='CustomSampleSizeSampler',
        dataset_size=[-1, -1, -1, -1, 500000]),
    dataset=dict(datasets=[
        o365v1_od_dataset, flickr30k_dataset, gqa_dataset, v3det_dataset,
        grit_dataset
    ]))
# --------------------------- testing ---------------------------#
class_name = (
    "lighting", "ceiling speaker", "wall", "ceiling", "door",
    "floor", "smoke detector", "trash bin", "elevator call button", "light switch",
    "exit sign", "board", "fire extinguisher", "manual call point", "elevator",
    "handrail", "pipe", "display case", "staircase", "radiator",
    "socket", "door sign", "window"
)

num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255), (199, 100, 0)
    ]
)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text', 'custom_entities', 'tokens_positive'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='../../data/SWL/',
        ann_file='annotations/new_020724/SWL_new_100_020724_polygon.json',
        data_prefix=dict(img='images/'),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file='../../data/SWL/annotations/new_020724/SWL_new_100_020724_polygon.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric'
)

test_evaluator = val_evaluator