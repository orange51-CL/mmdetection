input_size = 300
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='SSDVGG',
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://vgg16_caffe')),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256),
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=8,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=300,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True
dataset_type = 'CocoDataset'
data_root = '/home/cl-51/PycharmProjects/mmdetection/data/'
img_norm_cfg = dict(
    mean=[219.9792, 227.695, 230.7076], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[219.9792, 227.695, 230.7076],
        to_rgb=True,
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
    dict(
        type='Normalize',
        mean=[219.9792, 227.695, 230.7076],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(
                type='Normalize',
                mean=[219.9792, 227.695, 230.7076],
                std=[1, 1, 1],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=12,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='CocoDataset',
            classes=('Portable_Charger_1', 'Portable_Charger_2', 'Water',
                     'Laptop', 'Mobile_Phone', 'Tablet', 'Cosmetic',
                     'Nonmetallic_Lighter'),
            img_prefix=
            '/home/cl-51/PycharmProjects/mmdetection/data/HiXray/train/train_image',
            ann_file=
            '/home/cl-51/PycharmProjects/mmdetection/data/HiXray/train/train.json',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Expand',
                    mean=[219.9792, 227.695, 230.7076],
                    to_rgb=True,
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
                dict(
                    type='Normalize',
                    mean=[219.9792, 227.695, 230.7076],
                    std=[1, 1, 1],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/cl-51/PycharmProjects/mmdetection/data/HiXray/test/test.json',
        img_prefix=
        '/home/cl-51/PycharmProjects/mmdetection/data/HiXray/test/test_image',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(300, 300),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[219.9792, 227.695, 230.7076],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop',
                 'Mobile_Phone', 'Tablet', 'Cosmetic', 'Nonmetallic_Lighter')),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/cl-51/PycharmProjects/mmdetection/data/HiXray/test/test.json',
        img_prefix=
        '/home/cl-51/PycharmProjects/mmdetection/data/HiXray/test/test_image',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(300, 300),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[219.9792, 227.695, 230.7076],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop',
                 'Mobile_Phone', 'Tablet', 'Cosmetic', 'Nonmetallic_Lighter')))
evaluation = dict(interval=10, metric='bbox')
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.01,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
classes = ('Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop',
           'Mobile_Phone', 'Tablet', 'Cosmetic', 'Nonmetallic_Lighter')
total_epochs = 20
work_dir = 'work_output/_train_hixray_ssd'
auto_resume = False
gpu_ids = [0]
