_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    neck=dict(
        type='PSAFPN'),
    roi_head=dict(
        bbox_head=dict(
            num_classes=3),
        mask_head=dict(
            num_classes=3)))

# Modify dataset related settings
data_root = 'data/qiangjicoco/'
classes = ('pig', 'pighead', 'Pig_feet')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        classes=classes,        
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/'),
    val=dict(
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'),
    test=dict(
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'))
evaluation = dict(metric=['bbox', 'segm'])