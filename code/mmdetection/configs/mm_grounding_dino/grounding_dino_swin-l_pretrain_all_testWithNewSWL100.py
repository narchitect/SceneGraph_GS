_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth'  # noqa

num_levels = 5
model = dict(
    use_autocast=True,
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=None),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))))

# --------------------------- object365v2 od dataset---------------------------
# objv2_backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/objects365v2/': 'yudong:s3://wangyudong/obj365_v2/',
#         'data/objects365v2/': 'yudong:s3://wangyudong/obj365_v2/'
#     }))
objv2_backend_args = None

objv2_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=objv2_backend_args),
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
        label_map_file='data/objects365v2/annotations/o365v2_label_map.json',
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

o365v2_dataset = dict(
    type='ODVGDataset',
    data_root='data/objects365v2/',
    ann_file='annotations/zhiyuan_objv2_train_od.json',
    label_map_file='annotations/o365v2_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=objv2_train_pipeline,
    return_classes=True,
    need_text=False,
    backend_args=None,
)

# --------------------------- openimagev6 od dataset---------------------------
# oi_backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
oi_backend_args = None

oi_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=oi_backend_args),
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
        label_map_file='data/OpenImages/annotations/openimages_label_map.json',
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

oiv6_dataset = dict(
    type='ODVGDataset',
    data_root='data/OpenImages/',
    ann_file='annotations/oidv6-train-annotations_od.json',
    label_map_file='annotations/openimages_label_map.json',
    data_prefix=dict(img='OpenImages/train/'),
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,
    pipeline=oi_train_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- v3det od dataset---------------------------
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
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/V3Det/',
        ann_file='annotations/v3det_2023_v1_train_od.json',
        label_map_file='annotations/v3det_2023_v1_label_map.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False),
        need_text=False,
        pipeline=v3d_train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- lvis od dataset---------------------------
lvis_train_pipeline = [
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
        label_map_file='data/coco/annotations/lvis_v1_label_map.json',
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]
lvis_dataset = dict(
    type='ClassBalancedDataset',
    oversample_thr=1e-3,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='annotations/lvis_v1_train_od.json',
        label_map_file='annotations/lvis_v1_label_map.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False),
        need_text=False,  # change this
        pipeline=lvis_train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- coco2017 od dataset---------------------------
coco2017_train_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='annotations/instance_train2017_norefval_od.json',
        label_map_file='annotations/coco2017_label_map.json',
        data_prefix=dict(img='train2017'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- flickr30k vg dataset---------------------------
flickr30k_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/flickr30k_entities/',
        ann_file='final_flickr_separateGT_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='flickr30k_images/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- gqa vg dataset---------------------------
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

# --------------------------- coco2014 vg dataset---------------------------
coco2014_vg_dataset = dict(
    type='ODVGDataset',
    data_root='data/coco/',
    ann_file='mdetr_annotations/final_mixed_train_only_coco_vg.json',
    label_map_file=None,
    data_prefix=dict(img='train2014/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- refcoco vg dataset---------------------------
refcoco_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_refcoco_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- refcoco+ vg dataset---------------------------
refcoco_plus_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_refcoco+_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- refcocog vg dataset---------------------------
refcocog_dataset = dict(
    type='RepeatDataset',
    times=3,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_refcocog_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- grefcoco vg dataset---------------------------
grefcoco_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_grefcoco_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- grit vg dataset---------------------------
# grit_backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/grit/': 'yichen:s3://chenyicheng/grit/',
#         'data/grit/': 'yichen:s3://chenyicheng/grit/'
#     }))
grit_backend_args = None

grit_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=grit_backend_args),
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
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

grit_dataset = dict(
    type='ODVGDataset',
    data_root='data/grit/',
    ann_file='grit20m_vg.json',
    label_map_file=None,
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=grit_train_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- dataloader---------------------------
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(
        _delete_=True,
        type='CustomSampleSizeSampler',
        ratio_mode=True,
        # OD ~ 1.74+1.67*0.5+0.18*2+0.12*2+0.1=3.2
        # vg ~ 0.15*2+0.62*1+0.49*1+0.12*2+0.12*2+0.08*3+0.19*2+9*0.09=3.3
        dataset_size=[-1, 0.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.09]),
    dataset=dict(datasets=[
        o365v2_dataset,  # 1.74M
        oiv6_dataset,  # 1.67M
        v3det_dataset,  # 0.18M
        coco2017_train_dataset,  # 0.12M
        lvis_dataset,  # 0.1M
        flickr30k_dataset,  # 0.15M
        gqa_dataset,  # 0.62M
        coco2014_vg_dataset,  # 0.49M
        refcoco_dataset,  # 0.12M
        refcoco_plus_dataset,  # 0.12M
        refcocog_dataset,  # 0.08M
        grefcoco_dataset,  # 0.19M
        grit_dataset  # 9M
    ]))

# 4NODES * 8GPU
optim_wrapper = dict(optimizer=dict(lr=0.0001))

max_iter = 250000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=13000)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[210000],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=13000, max_keep_ckpts=30))
log_processor = dict(by_epoch=False)

# --------------------------- testing ---------------------------#
data_root = '../../data/SWL/'
class_name = (
    "lighting", "ceiling speaker", "wall", "ceiling", "door",
    "floor", "smoke detector", "trash bin", "elevator call button", "light switch",
    "exit sign", "board", "fire extinguisher", "manual call point", "elevator",
    "handrail", "pipe", "display case", "staircase", "radiator",
    "socket", "door sign", "window"
)

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette = [
    (0, 255, 255),  # lighting
    (34, 139, 34),  # ceiling speaker
    (220, 20, 60),  # wall
    (255, 215, 0),  # ceiling
    (0, 0, 255),  # door
    (75, 0, 130),  # floor
    (255, 69, 0),  # smoke detector
    (128, 128, 0),  # trash bin
    (0, 255, 127),  # elevator call button
    (255, 192, 203),  # light switch
    (0, 128, 128),  # exit sign
    (255, 165, 0),  # board
    (255, 0, 255),  # fire extinguisher
    (60, 179, 113),  # manual call point
    (70, 130, 180),  # elevator
    (199, 21, 133),  # handrail
    (210, 105, 30),  # pipe
    (144, 238, 144),  # display case
    (255, 20, 147),  # staircase
    (0, 100, 0),  # radiator
    (47, 79, 79),  # socket
    (138, 43, 226),  # door sign
    (128, 0, 128)  # window
])




test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=objv2_backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='RandomChoiceResize',
#         scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                 (736, 1333), (768, 1333), (800, 1333)],
#         keep_ratio=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'flip', 'flip_direction', 'text',
#                    'custom_entities', 'tokens_positive', 'dataset_mode')),
# ]


val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/new_020724/SWL_new_100_020724_test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/new_020724/SWL_new_100_020724_test_polygon.json')
test_evaluator = val_evaluator
