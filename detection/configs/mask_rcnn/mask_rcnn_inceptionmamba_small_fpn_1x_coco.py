_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    # pretrained=None,
    backbone=dict(
        _delete_=True,
        type='InceptionMamba',
        arch='small',
        pretrained=None, # path to pretrained model
        drop_path_rate=0.2
        ),
        neck=dict(
              type='FPN',
              in_channels=[72, 144, 288, 576],
              out_channels=256,
              num_outs=5,
        ),
)

ratio = 1
# bs_ratio = 2  # 0.0002 for 2 * 8
bs_ratio = 1

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0002 * ratio, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                   'relative_position_bias_table': dict(decay_mult=0.),
                                                   'norm': dict(decay_mult=0.)}),
    clip_grad=dict(max_norm=0.1, norm_type=2), )

max_epochs = 12
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=max_epochs // 2,
        T_max=max_epochs // 2,
        end=max_epochs,
        by_epoch=True,
        eta_min=0)
]


train_dataloader = dict(
    batch_size=2 * bs_ratio * ratio,
    num_workers=min(2 * bs_ratio * ratio, 2),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,)
test_dataloader = val_dataloader