# The new config inherits a base config to highlight the necessary modification
_base_ = './atss_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(
        num_classes=16))

# Modify dataset related settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data_root = 'data/boat/'
classes = ('general_cargo_ship', 'container_ship', 'bulk_carrier', 'battleship', 'fishing_vessel',
          'cruise_ship' ,'high_speed_vessel' ,'yacht', 'pilot_ship', 'tug', 'others', 
           'Aircraft_carrier', 'crane_ship', 'landing_ship', 'oil_tanker', 'submarine', )
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline))