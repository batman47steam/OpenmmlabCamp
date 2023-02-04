_base_ = ['./mobilenet-v3-small_8xb128_in1k.py']

dataset_type = 'CustomDataset'
data_preprocessor = dict(num_classes=5)
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small_8xb128_in1k_20221114-bd1bfcde.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=5, topk=(1,)))

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root='data',
        data_prefix='train',
        pipeline={{_base_.train_pipeline}}
    ),
)

val_dataloader = dict(
    batch_size = 32,
    num_workers = 2,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root='data',
        data_prefix='val',
        pipeline={{_base_.test_pipeline}}
    ),

)

val_evaluator = dict(type='Accuracy', topk=(1,))
train_cfg = dict(max_epochs=50, val_interval=5)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)

default_hooks = dict(
    logger=dict(interval=10),
    checkpoint=dict(interval=10),
)

