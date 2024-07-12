_base_ = [
    './grounding_dino_swin-t_finetune_16xb2_1x_coco.py',
]

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'  # noqa

data_root = '../Data/'
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
    'stecker'#SWL
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
    (47, 79, 79)    # stecker
])

model = dict(
    type='GroundingDINO',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        patch_norm=True),
    neck=dict(in_channels=[256, 512, 1024]),
    bbox_head=dict(num_classes=num_classes)
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='Coco2017_SWL/SWL_coco_trainval.json',
        data_prefix=dict(img='Data_Fiona_SL/')),
    batch_size=2,  # 4 to 2
    num_workers=4,
)

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='Coco2017_SWL/SWL_coco_test.json',
        data_prefix=dict(img='Data_Fiona_SL/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'Coco2017_SWL/SWL_coco_test.json')
test_evaluator = val_evaluator

max_epoch = 20  # Adjust the number of epochs as needed

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)
