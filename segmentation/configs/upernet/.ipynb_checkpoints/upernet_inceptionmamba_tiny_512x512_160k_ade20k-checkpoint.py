_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type="InceptionMamba",
        arch="tiny",
        pretrained="/hy-tmp/crossnext/inceptionmamba_tiny.pth",
        drop_path_rate=0.1
       ),
    decode_head=dict(in_channels=[72, 144, 288, 576], num_classes=150),
    auxiliary_head=dict(in_channels=288, num_classes=150)
)

ratio = 1
bs_ratio = 4  # 0.00012 for 4 * 8

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00012 * ratio, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

max_iters = 80000 * 2
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='CosineAnnealingLR',
        begin=max_iters // 2,
        T_max=max_iters // 2,
        end=max_iters,
        by_epoch=False,
        eta_min=0)
]

train_dataloader = dict(
    batch_size=2 * bs_ratio * ratio,
    num_workers=min(2 * bs_ratio * ratio, 8),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,)
test_dataloader = val_dataloader

