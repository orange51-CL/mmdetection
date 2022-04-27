_base_ = [
    '../_base_/models/ssd300.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'CocoDataset'
classes = ('Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop', 'Mobile_Phone', 'Tablet', 'Cosmetic',
           'Nonmetallic_Lighter')

img_norm_cfg = dict(
    mean=[219.9792, 227.6950, 230.7076], std=[1, 1, 1], to_rgb=True)
data_root = '/home/cl-51/PycharmProjects/mmdetection/data/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=128,  # batchsize
    workers_per_gpu=12,  # num_workers
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            img_prefix=data_root + 'HiXray/train/train_image',
            ann_file=data_root + 'HiXray/train/train.json',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=data_root + 'HiXray/test/test_image',
        ann_file=data_root + 'HiXray/test/test.json',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=data_root + 'HiXray/test/test_image',
        ann_file=data_root + 'HiXray/test/test.json',
        pipeline=test_pipeline
    ),
)
evaluation = dict(interval=10, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
# optimizer_config = dict(_delete_=True)
# Modify runtime related settings
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# Modify runtime related settings
checkpoint_config = dict(interval=10)
# python tools/train.py configs/hixray_nas/hixray_ssd.py --work-dir work_output/_train_hixray_ssd


# 2022-04-18 13:30:58,118 - mmdet - INFO -
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.391
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.739
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.374
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.138
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.397
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.497
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.497
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.497
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.244
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.514
#
# 2022-04-18 13:30:58,391 - mmdet - INFO - Exp name: hixray_ssd.py
# 2022-04-18 13:30:58,392 - mmdet - INFO - Epoch(val) [20][9069]  bbox_mAP: 0.3910, bbox_mAP_50: 0.7390, bbox_mAP_75: 0.3740, bbox_mAP_s: 0.0000, bbox_mAP_m: 0.1380, bbox_mAP_l: 0.3970, bbox_mAP_copypaste: 0.391 0.739 0.374 0.000 0.138 0.397
