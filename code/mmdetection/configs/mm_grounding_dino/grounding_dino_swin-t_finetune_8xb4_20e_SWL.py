_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

data_root = 'data/SWL/'
class_name = (
    'wall',
    'ceiling',
    'lighting',
    'speaker',
    'door',
    'smoke alarm',
    'floor',
    'trash bin',
    'elevator button',
    'escape sign',
    'board',
    'fire extinguisher',
    'door sign',
    'light switch',
    'emergency switch button',
    'elevator',
    'handrail',
    'show window',
    'pipes',
    'staircase',
    'window',
    'radiator',
    'stecker'
)

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette = [
    (220, 20, 60),   # wall
    (255, 215, 0),   # ceiling
    (0, 255, 255),   # lighting
    (34, 139, 34),   # speaker
    (0, 0, 255),     # door
    (255, 69, 0),    # smoke alarm
    (75, 0, 130),    # floor
    (128, 128, 0),   # trash bin
    (0, 255, 127),   # elevator button
    (0, 128, 128),   # escape sign
    (255, 165, 0),   # board
    (255, 0, 255),   # fire extinguisher
    (138, 43, 226),  # door sign
    (255, 192, 203), # light switch
    (60, 179, 113),  # emergency switch button
    (70, 130, 180),  # elevator
    (199, 21, 133),  # handrail
    (144, 238, 144), # show window
    (210, 105, 30),  # pipes
    (255, 20, 147),  # staircase
    (128, 0, 128),   # window
    (0, 100, 0),     # radiator
    (47, 79, 79)     # stecker
])

model = dict(bbox_head=dict(num_classes=num_classes))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='RandomChoice', #remove randomCrop, reduce randomChoiceResize
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(1728, 2592), (1600, 2400)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations/SWL_coco_trainval.json',
        data_prefix=dict(img='images/')),
    batch_size=2,  # 4 to 2
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler')
)

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/SWL_coco_test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/SWL_coco_test.json')
test_evaluator = val_evaluator

max_epoch = 20 #original 8 gpu, our case 2gpu (rtx8000) keep it or 25

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0)
        }))

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
