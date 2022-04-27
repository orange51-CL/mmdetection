# hixray_faster_rcnn_resnet50_fpn
# The new config inherits a base config to highlight the necessary modification
# hixray_nas
_base_ = [
    # '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='MobileNetV2',
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU6')
    ),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 1280],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
)

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop', 'Mobile_Phone', 'Tablet', 'Cosmetic',
           'Nonmetallic_Lighter')

img_norm_cfg = dict(
    mean=[219.9792, 227.6950, 230.7076], std=[57.3258, 43.7994, 50.0889], to_rgb=True)

data_root = '/home/cl-51/PycharmProjects/mmdetection/data/'
data = dict(
    samples_per_gpu=128,  # batchsize
    workers_per_gpu=12,  # num_workers
    train=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=data_root + 'HiXray/train/train_image',
        ann_file=data_root + 'HiXray/train/train.json'
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

# Modify schedule related settings
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=5, metric='bbox')
# Modify runtime related settings
checkpoint_config = dict(interval=10)

# We can use the pre-trained model to obtain higher performance
# load_from = 'checkpoints/*.pth'


# python tools/train.py configs/hixray_nas/hixray_faster_rcnn_mobilenetv2_fpn.py --work-dir work_output/_train_hixray_faster_rcnn_mobilenetv2_fpn

# python tools/analysis_tools/analyze_logs.py plot_curve work_output/_train_hixray_nas/*.log.json --keys loss_cls loss_bbox --legend loss_cls loss_bbox