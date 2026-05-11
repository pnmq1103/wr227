"""
Fair Baseline: RALA ViT + RetinaNet + FPN on COCO, 1× schedule.

This is the control group: same backbone architecture with w=1.0 always
(no MARL routing). Same pretrained weights, same training schedule.

Differences from marl_rala_retinanet_1x_coco.py:
- freeze_router=True permanently (no FreezeRouterHook, no PPORouterHook)
- No custom hooks

This gives a clean A/B comparison to measure the impact of MARL routing.
"""

custom_imports = dict(
    imports=['mmdet_marl'],
    allow_failed_imports=False,
)

# Same model, but router is permanently frozen
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='MARLRALABackbone',
        patch_size=16,
        in_chans=3,
        d_model=256,
        depth=16,
        num_heads=8,
        chunk_size=16,
        drop_path_rate=0.1,
        out_channels=(256, 256, 256, 256),
        fpn_scales=(2.0, 1.0, 0.5, 0.25),
        out_indices=(3, 7, 11, 15),
        freeze_router=True,  # ALWAYS frozen — this is the baseline
        init_cfg=dict(
            type='Pretrained',
            checkpoint='vit_v2_stage_2_router.pth',
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 256, 256, 256],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
)

# ──────────────────────────────────────────────────────────────────
# Dataset (same as MARL config)
# ──────────────────────────────────────────────────────────────────
dataset_type = 'CocoDataset'
data_root = 'D:/kagglehub_cache/datasets/awsaf49/coco-2017-dataset/versions/1/coco2017/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        pipeline=test_pipeline,
        test_mode=True,
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ──────────────────────────────────────────────────────────────────
# Schedule: 1× = 12 epochs
# ──────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0, end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

auto_scale_lr = dict(enable=False, base_batch_size=16)

# ──────────────────────────────────────────────────────────────────
# Runtime — NO custom hooks (baseline has no PPO)
# ──────────────────────────────────────────────────────────────────
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

custom_hooks = []  # No router hooks — baseline mode

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
