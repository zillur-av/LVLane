net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
)
featuremap_out_channel = 128
featuremap_out_stride = 8
num_classes = 3
num_lanes = 6 + 1
classification = True

aggregator = dict(
    type='RESA',
    direction=['d', 'u', 'r', 'l'],
    alpha=2.0,
    iter=4,
    conv_stride=9,
)

sample_y=range(710, 150, -10)
heads = dict(
    type='LaneSeg',
    decoder=dict(type='BUSD'),
    thr=0.6,
    sample_y=sample_y,
    cat_dim = (num_lanes - 1, num_classes)
)

optimizer = dict(
  type = 'SGD',
  lr = 0.025,
  weight_decay = 1e-4,
  momentum = 0.9
)


epochs = 15
batch_size = 4
total_iter = (3216 // batch_size + 1) * epochs 
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

bg_weight = 0.4

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

img_height = 368
img_width = 640
cut_height = 160
ori_img_h = 720
ori_img_w = 1280

train_process = [
    dict(type='RandomRotation'),
    dict(type='RandomHorizontalFlip'),
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor'),
] 

val_process = [
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor'),
] 

infer_process = [
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img']),
] 

dataset_path = './data/tusimple'
dataset = dict(
    train=dict(
        type='TuSimple',
        data_root=dataset_path,
        split='trainval',
        processes=train_process,
    ),
    val=dict(
        type='TuSimple',
        data_root=dataset_path,
        split='val',
        processes=val_process,
    ),
    test=dict(
        type='TuSimple',
        data_root=dataset_path,
        split='test',
        processes=val_process,
    )
)


workers = 8
ignore_label = 255
log_interval = 200
eval_ep = 1
save_ep = epochs
#test_json_file='data/tusimple/label_data_0531_small.json'
test_json_file='data/tusimple/label_data_0531.json'
lr_update_by_epoch = False