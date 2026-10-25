import argparse
import json

import cv2
from PIL import Image
import numpy as np
import os
import time
import torch.nn.functional as F
from matplotlib import pyplot as plt
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS, get_tokenizer, create_model_and_transforms
import torch
from torch import nn
import torchvision.transforms as T
from util.misc import nested_tensor_from_tensor_list
from models import build_model



# if torch.backends.mps.is_available():
#     device = torch.device("mps")  # 使用 MPS 加速
# else:
#     device = torch.device("cpu")  # 回退到 CPU
device = torch.device("cuda")
print("[INFO] 当前使用{}做推断".format(device))

# 图像数据处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--test_path', type=str)

    # data parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


# plot box by opencv
def plot_result(pil_img, prob, boxes, save_dir, save_name=None, imshow=False, imwrite=False):
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    LABEL = ['N/A', 'mass']
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
        cl = p.argmax()
        # if cl == 0:
        # print("1111111111111111111111111")
        label_text = '{}: {}%'.format(LABEL[cl], round(p[cl] * 100, 2))
        # print(round(p[cl] * 100, 2))
        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)

    if imwrite:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_dir + '/{}'.format(save_name), opencvImage)


# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b


# def load_model(model_path, args):
#     clip_model_name = "biomedclip_local"
#
#     with open(r"D:\PycharmProjects\detr_mv\patch_detr\openclip\checkpoints\open_clip_config.json", "r") as f:
#         config = json.load(f)
#         model_cfg = config["model_cfg"]
#         preprocess_cfg = config["preprocess_cfg"]
#
#     if (not clip_model_name.startswith(HF_HUB_PREFIX)
#             and clip_model_name not in _MODEL_CONFIGS
#             and config is not None):
#         _MODEL_CONFIGS[clip_model_name] = model_cfg
#
#     clip_model, _, preprocess = create_model_and_transforms(
#         model_name=clip_model_name,
#         pretrained=r"D:\PycharmProjects\detr_mv\patch_detr\openclip\checkpoints\open_clip_pytorch_model.bin",
#         **{f"image_{k}": v for k, v in preprocess_cfg.items()},
#     )
#     model, _, _ = build_model(args, clip_model)
#     model.to(device)
#     model.eval()
#     state_dict = torch.load(model_path, map_location=torch.device('cuda'))  # <-----------修改加载模型的路径
#     model.load_state_dict(state_dict["model"], strict=False)
#     model.to(device)
#     print("load model sucess")
#     return model

def load_model(model_path, args):

    # clip_model_name = "biomedclip_local"

    # with open("/home/sun/hjc/detr-mv/openclip/checkpoints/open_clip_config.json", "r") as f:
    #     config = json.load(f)
    #     model_cfg = config["model_cfg"]
    #     preprocess_cfg = config["preprocess_cfg"]

    # if (not clip_model_name.startswith(HF_HUB_PREFIX)
    #         and clip_model_name not in _MODEL_CONFIGS
    #         and config is not None):
    #     _MODEL_CONFIGS[clip_model_name] = model_cfg

    # clip_model, _, preprocess = create_model_and_transforms(
    #     model_name=clip_model_name,
    #     pretrained="/home/sun/hjc/detr-mv/openclip/checkpoints/open_clip_pytorch_model.bin",
    #     **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    # )
    model, _, _ = build_model(args)
    model.to(device)
    model.eval()
    state_dict = torch.load(model_path,map_location=torch.device('cuda'))  # <-----------修改加载模型的路径
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model

def plot_gradcam(self, cam):
    """ Plot Grad-CAM heatmap. """
    plt.imshow(cam.cpu().detach().numpy(), cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.savefig("out.png", bbox_inches='tight', dpi=300)  # 保存图像，设置高分辨率


# 图像的推断
def detect(im, model, transform, prob_threshold=0.85):
    with torch.no_grad():  # 禁用梯度计算
        # mean-std normalize the input image (batch-size: 1)
        model.train()
        img = transform(im).unsqueeze(0)
        # for name, param in model.named_parameters():
        #     print(f"Parameter {name} is on device: {param.device}")
        # propagate through the model
        img = img.to(device)
        start = time.time()

        samples = img
        if isinstance(img, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(img)

        features, pos = model.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = model.transformer(model.input_proj(src), mask, model.query_embed.weight, pos[-1])[0]

        outputs_class = model.class_embed(hs)
        outputs_coord = model.bbox_embed(hs).sigmoid()
        outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        pred_class_idx = torch.argmax(outputs_class[-1], dim=-1)

        # 确保仅在需要计算梯度的部分启用梯度
        outputs_class[-1].requires_grad_()  # 启用梯度计算

        # 反向传播之前，确保没有额外的梯度计算
        model.zero_grad()
        print(outputs_class[-1].grad_fn)  # 检查这个张量的计算图是否正常

        print(outputs_class[-1].requires_grad)

        # 保留梯度
        outputs_class[-1].retain_grad()


        # 执行反向传播以计算梯度
        outputs_class[-1].backward(torch.ones_like(outputs_class[-1]))  # 对分类部分执行反向传播

        # 获取特征图和梯度
        grads = hs.grad
        pooled_grads = F.adaptive_avg_pool2d(grads, 1)

        # 获取最后一层特征图并计算加权热力图
        feature_map = hs[0]  # 假设选择第一个解码器层的输出特征图
        weights = pooled_grads * feature_map
        cam = torch.sum(weights, dim=1).squeeze()

        # 归一化热力图
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        # 将热力图显示在原始图像上
        plot_gradcam(cam)

        # outputs, cc_merge_text, cc_encoder_fea, ml_out, ml_merge_text, ml_encoder_fea = model(img,fake_img,text_embedding)
        # # print(outputs)

        # keep only predictions with 0.7+ confidence
        # print(outputs['pred_logits'].softmax(-1)[0, :, :-1])
        probas = outputs['pred_logits'].softmax(-1)
        probas = probas[0, :, :-1]
        keep = probas.max(-1).values > prob_threshold

        # print("keep{}".format(keep))
        score = probas[keep]

        probas = probas.cpu().detach().numpy()
        keep = keep.cpu().detach().numpy()

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        end = time.time()

    return probas[keep], bboxes_scaled, end - start


if __name__ == "__main__":
    # 阈值
    prob_threshold = 0
    main_args = get_args_parser().parse_args()
    # 加载模型
    save_dir = r"./results/1"
    dfdetr = load_model(r"detr-cbis-3/checkpoint0099.pth", main_args)
    # img_path = r'D:\PycharmProjects\patch_detr\huiying_val\images2'
    # img_path = r'D:\PycharmProjects\patch_detr\huiying_nonetest'
    img_path = main_args.test_path
    # img_path = r"D:\Desktop\1"
    files = os.listdir(main_args.test_path)  # <--修改为待预测图片所在文件夹路径

    cn = 0
    waste = 0
    json_list = []
    img_path=r"/home/km/fsdownload/cbis_test/images2/Mass-Test_P_00017_LEFT_CC.png"
    im = Image.open(img_path).convert('RGB')
    scores, boxes, waste_time = detect(im, dfdetr, transform, prob_threshold)

