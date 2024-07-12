_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

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

model = dict(bbox_head=dict(num_classes=num_classes))

train_pipeline = [
    dict(type='LoadImageFromFile'),
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
        ann_file='annotations/new_020724/SWL_new_100_020724_trainval_polygon.json',
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
        ann_file='annotations/new_020724/SWL_new_100_020724_test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/new_020724/SWL_new_100_020724_test_polygon.json')
test_evaluator = val_evaluator

max_epoch = 100 #original 8 gpu, our case 2gpu (rtx8000) keep it or 25

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
