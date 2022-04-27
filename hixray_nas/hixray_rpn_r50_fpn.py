_base_ = [
    '../_base_/models/rpn_r50_fpn.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[219.9792, 227.6950, 230.7076], std=[57.3258, 43.7994, 50.0889], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes']),
]

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop', 'Mobile_Phone', 'Tablet', 'Cosmetic',
           'Nonmetallic_Lighter')

data_root = '/home/cl-51/PycharmProjects/mmdetection/data/'
data = dict(
    samples_per_gpu=256,  # batchsize
    workers_per_gpu=12,  # num_workers
    train=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=data_root + 'HiXray/train/train_image',
        ann_file=data_root + 'HiXray/train/train.json',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=data_root + 'HiXray/test/test_image',
        ann_file=data_root + 'HiXray/test/test.json'
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=data_root + 'HiXray/test/test_image',
        ann_file=data_root + 'HiXray/test/test.json'
    ),
)

evaluation = dict(interval=10, metric='proposal_fast')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=10)

# python tools/train.py configs/hixray_nas/hixray_rpn_r50_fpn.py --work-dir work_output/_train_hixray_rpn_r50_fpn
