_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop', 'Mobile_Phone', 'Tablet', 'Cosmetic',
           'Nonmetallic_Lighter',)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1.0, 1.0, 1.0], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
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
        img_scale=(300, 300),
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
data_root = '/home/cl-51/PycharmProjects/mmdetection/data/'
data = dict(
    samples_per_gpu=220,  # batchsize
    workers_per_gpu=12,  # num_workers
    train=dict(
        type=dataset_type,
        img_prefix=data_root + 'HiXray/train/train_image/',
        classes=classes,
        ann_file=data_root + 'HiXray/train/train.json',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + 'HiXray/test/test_image/',
        classes=classes,
        ann_file=data_root + 'HiXray/test/test.json',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'HiXray/test/test_image/',
        classes=classes,
        ann_file=data_root + 'HiXray/test/test.json',
        pipeline=test_pipeline
    ),
)
evaluation = dict(interval=10)

# optimizer
optimizer = dict(lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)

# Modify runtime related settings
checkpoint_config = dict(interval=10)

# python tools/train.py configs/hixray_nas/hixray_fcos_resnet50.py --work-dir work_output/_train_hixray_fcos_resnet50

# python tools/analysis_tools/analyze_logs.py plot_curve work_output/_train_hixray_nas/*.log.json --keys loss_cls loss_bbox --legend loss_cls loss_bbox

# 2022-04-18 17:33:58,563 - mmdet - INFO -
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.314
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.614
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.289
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.059
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.321
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.455
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.455
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.212
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.482
#
# 2022-04-18 17:33:59,103 - mmdet - INFO - Exp name: hixray_fcos_resnet50.py
# 2022-04-18 17:33:59,103 - mmdet - INFO - Epoch(val) [20][9069]  bbox_mAP: 0.3140, bbox_mAP_50: 0.6140, bbox_mAP_75: 0.2890, bbox_mAP_s: 0.0000, bbox_mAP_m: 0.0590, bbox_mAP_l: 0.3210, bbox_mAP_copypaste: 0.314 0.614 0.289 0.000 0.059 0.321
