net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)
featuremap_out_channel = 512

griding_num = 100
num_lanes = 6
classification = True
num_classes = 7
autocast = True

heads = dict(type='LaneCls',
        dim = (griding_num + 1, 56, num_lanes),
        cat_dim =(num_classes, num_lanes))

trainer = dict(
    type='LaneCls'
)

evaluator = dict(
    type='Tusimple',
)

optimizer = dict(
  type = 'SGD',
  lr = 0.025,
  weight_decay = 1e-4,
  momentum = 0.9
)
#optimizer = dict(type='Adam', lr= 0.025, weight_decay = 0.0001)  # 3e-4 for batchsize 8

epochs = 40
batch_size = 16
total_training_samples = 3626
total_iter = (total_training_samples // batch_size + 1) * epochs 

import math

scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)
img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

ori_img_h = 720
ori_img_w = 1280
img_h = 288
img_w = 800
cut_height= 0
sample_y = range(710, 150, -10)

dataset_type = 'TuSimple'
dataset_path = './data/tusimple'
row_anchor = 'tusimple_row_anchor'

train_process = [
    dict(type='RandomRotation', degree=(-6, 6)),
    dict(type='RandomUDoffsetLABEL', max_offset=100),
    dict(type='RandomLROffsetLABEL', max_offset=200),
    dict(type='GenerateLaneCls', row_anchor=row_anchor,
        num_cols=griding_num, num_lanes=num_lanes),
    dict(type='Resize', size=(img_w, img_h)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img', 'cls_label']),
]

val_process = [
    dict(type='GenerateLaneCls', row_anchor=row_anchor,
        num_cols=griding_num, num_lanes=num_lanes),
    dict(type='Resize', size=(img_w, img_h)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img', 'cls_label']),
]

infer_process = [
    dict(type='Resize', size=(img_w, img_h)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img']),
]

dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='trainval',
        processes=train_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='val',
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='val',
        processes=val_process,
    )
)


workers = 8
ignore_label = 255
log_interval = 200
eval_ep = 1
save_ep = epochs
row_anchor='tusimple_row_anchor'
test_json_file='data/tusimple/test_label.json'
lr_update_by_epoch = False
