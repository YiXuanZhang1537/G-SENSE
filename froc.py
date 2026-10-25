import json
from collections import OrderedDict
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
from natsort import natsorted
import numpy as np


def compute_overlap(a, b):
    # a [N,4]
    # b [M,4]
    area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - \
         np.maximum(np.expand_dims(a[:, 0], axis=1), b[:, 0]) + 1
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - \
         np.maximum(np.expand_dims(a[:, 1], axis=1), b[:, 1]) + 1

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1),
                        axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih
    return intersection / ua  # (N,M)


def _compute_froc(fps, rec, begin=0.125, end=2):
    # to average recall of fps within [begin, end]
    index = [i for i, fp in enumerate(list(fps)) if (begin <= fp <= end)]
    if len(index) == 0:
        return None
    froc_auc = np.sum(rec[index]) / float(len(index))
    return froc_auc


def _standardize(dts, gts):
    """ Convert gt/dt from dict to list[list[ndarray]] (if in dict format).
    Args:
        dts (dict or list[list[ndarray]]): detected bounding boxes for
            different labels in a set of images, each bbox is of shape (n, 5).
        gts (dict or list[list[ndarray]]): ground truth bounding boxes
            for different labels in a set of images, each bbox is of
            shape (n, 4).
            gts[img_id][label_id] = bboxes (for a specific label in an image).
    """
    assert len(gts) == len(dts), "dts and gts must have the same length"

    if isinstance(dts, dict) and isinstance(gts, dict):
        assert set(dts.keys()) == set(gts.keys()), \
            "dts and gts must have the same key set"
        dts = [value for _, value in natsorted(dts.items())]
        gts = [value for _, value in natsorted(gts.items())]

    return dts, gts


def eval_det(dts, gts, num_classes, iou_thr):
    """ Evaluate a given dataset by comparing DT with GT.

    Args:
        dts (dict or list[list[ndarray]]): detected bounding boxes for
            different labels in a set of images, each bbox is of shape (n, 5).
            Example:
                {
                    "1.2.45212.0455554.3.01555":  # first image
                        [
                            [
                                [128.3, 127.5, 156.8, 140.5, 0.993],  # first box, [xmin, ymin, xmax, ymax, probability]
                                [122.3, 122.4, 152.1, 142.0, 0.681],  # second box
                                [22.3, 22.4, 52.1, 42.0, 0.048],      # third box
                                ...
                            ],  # bounding boxes of class 1, numpy.ndarray
                            [], # bounding boxes of class 2, numpy.ndarray
                            ...
                        ]
                    ...
                }
        gts (dict or list[list[ndarray]]): ground truth bounding boxes
            for different labels in a set of images, each bbox is of
            shape (n, 4).
            gts[img_id][label_id] = bboxes (for a specific label in an image).
            Example:
                {
                    "1.2.45212.0455554.3.01555":  # first image
                        [
                            [
                                [128.3, 127.5, 156.8, 140.5],  # first box, [xmin, ymin, xmax, ymax]
                                [122.3, 122.4, 152.1, 142.0],  # second box
                                [22.3, 22.4, 52.1, 42.0],      # third box
                                ...
                            ],  # bounding boxes of class 1, numpy.ndarray
                            [], # bounding boxes of class 2, numpy.ndarray
                            ...
                        ]
                    ...
                }
        num_classes (int): number of classes to detect.
        iou_thr (float): threshold to determine whether a detection is
            positive or negative.

    Returns:
        (OrderedDict): AP, number of GT bboxes, FROC curve for each label.
    """
    print("dts,gts", len(dts), len(gts))
    dts, gts = _standardize(dts, gts)
    
    num_imgs = len(dts)
    num_classes = len(dts[0])
    results = OrderedDict()
    for label in range(num_classes):
        fps = np.zeros((0,), np.float32)
        tps = np.zeros((0,), np.float32)
        scores = np.zeros((0,), np.float32)
    num_imgs = len(gts)

    results = OrderedDict()

    # match ground truths and detection results
    for label in range(num_classes):
        fps = np.zeros((0,), np.float32)
        tps = np.zeros((0,), np.float32)
        scores = np.zeros((0,), np.float32)
        num_anns = 0.0

        for i in range(num_imgs):
            gt = gts[i][label]
            dt = dts[i][label]
            # print(dt)
            a = np.argsort(-dt[:, -1])
            dt = dt[a]
            num_anns += len(gt)
            matched_anns = []

            for d in dt:
                scores = np.append(scores, d[-1])

                if len(gt) == 0:
                    fps = np.append(fps, 1)
                    tps = np.append(tps, 0)
                    continue

                overlaps = compute_overlap(
                    np.expand_dims(d, axis=0).astype(np.float32),
                    gt.astype(np.float32)
                )
                matched_ann = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, matched_ann]

                if (max_overlap > iou_thr and
                        matched_ann not in matched_anns):
                    fps = np.append(fps, 0)
                    tps = np.append(tps, 1)
                    matched_anns.append(matched_ann)
                else:
                    fps = np.append(fps, 1)
                    tps = np.append(tps, 0)

        # no annotations -> AP for this class is None
        if num_anns == 0:
            results[label] = OrderedDict()
            results[label]['ap'] = None
            results[label]['num_gt_bboxes'] = 0
            results[label]['froc'] = None
            continue

        # sort by score
        indices = np.argsort(-scores)
        fps, tps, scores = fps[indices], tps[indices], scores[indices]

        # compute false positives and true positives at retrieval cutoff
        # of k bboxes (k in [1, n], n is the total number of detected bboxes)
        fps, tps = np.cumsum(fps), np.cumsum(tps)

        # compute recall and precision
        recall = tps / num_anns

        # compute AP, number of ground truth bboxes and FROC
        results[label] = OrderedDict()
        results[label]['num_gt_bboxes'] = num_anns
        results[label]['froc'] = fps / num_imgs, recall  # recall = sensitivity
        results[label]['froc_auc'] = _compute_froc(fps / num_imgs, recall)
    return results


def draw_froc_curve(average_fps,
                    sensitivity,
                    save_path=None,
                    bLogPlot=True,
                    **kwargs):
    """ Plot the FROC curve. """

    plt.figure(figsize=(16, 10), dpi=200)
    plt.plot(average_fps, sensitivity, **kwargs)

    plt.xlabel('Average number of false positives per scan')
    plt.ylabel('Sensitivity')
    plt.title('FROC Curve')
    plt.xlim(0, 5)
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    ax = plt.gca()
    xaxis = [0, 0.25, 0.5, 1, 2, 3, 4, 5]
    ax.xaxis.set_ticks(xaxis)
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.tight_layout()

    # ----------- 新增：在指定点计算 sensitivity -----------
    target_fps = [0.125, 0.25, 0.3, 0.5, 1.0, 2.0]
    interp_sens = np.interp(target_fps, average_fps, sensitivity)

    print("\n指定点 sensitivity:")
    for f, s in zip(target_fps, interp_sens):
        print(f"FPs={f:.3f} -> Sensitivity={s:.4f}")

    # 如果想在图上打点
    plt.scatter(target_fps, interp_sens, color="red", marker="x", label="指定点")
    for f, s in zip(target_fps, interp_sens):
        plt.text(f, s, f"{s:.2f}", fontsize=8, ha='left', va='bottom')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()



def divide_by_two(num, xaxis):
    xaxis.append(int(num))
    # print(num)  # 打印当前数值
    if num < 1:
        return num  # 如果数已小于1，则结束递归
    return divide_by_two(num / 2, xaxis)  # 否则继续递归除以2


# def convert_lists_to_arrays(data):
#     """递归转换JSON中的所有列表到NumPy数组"""
#     if isinstance(data, dict):
#         return {key: convert_lists_to_arrays(value) for key, value in data.items()}
#     elif isinstance(data, list):
#         return np.array(data)
#     else:
#         return data


if __name__ == '__main__':
    save_dir = "froc-5-0009.png"

    with open(r"predict_mv_box-5-0009.json", 'r') as file:
        # 加载 JSON 数据到一个字典
        predict_data = json.load(file)

    with open(r"/mnt/gemlab_data_3/zhangyixuan/cbis_test/cbis_boxes_test.json", 'r') as file:
        # 加载 JSON 数据到一个字典
        true_data = json.load(file)
    # predict_data = convert_lists_to_arrays(predict_data)
    # true_data = convert_lists_to_arrays(true_data)

    # 创建一个空字典来存储合并后的结果
    predict_dict = {}
    true_dict = {}

    # for i in range(len(predict_data)):
    #     for key, value in predict_data[i].items():
    #         # print(key, value)
    #         predict_data[key] = [value]
    #
    # for i in range(len(true_data)):
    #     for key, value in true_data[i].items():
    #         # print(key, value)
    #         true_data[key] = [value]

    # 遍历列表中的每个字典，更新到 combined_dict 中
    for sub_dict in predict_data:
        predict_dict.update(sub_dict)
    for key, value in predict_dict.items():
        if isinstance(value, list):
            if value == []:
                value = [[0, 0, 0, 0, 0]]
                print(value)
            predict_dict[key] = np.array([value])

        # for i in range(len(value)):
        #     if isinstance(value[i], list):
        #         predict_dict[key][i] = np.array(value[i])

    for sub_dict in true_data:
        true_dict.update(sub_dict)

    for key, value in true_dict.items():
        if isinstance(value, list):
            true_dict[key] = np.array([value])
        # for i in range(len(value)):
        #     if isinstance(value[i], list):
        #         true_dict[key][i] = np.array(value[i])
    print(true_dict)

    # -----------------------------------------------------
    gts = true_dict  # 预测框及概率，dict or list[list[ndarray]]，详见eval_det函数说明
    dts = predict_dict  # 真值框，dict or list[list[ndarray]]，详见eval_det函数说明
    out_dir = ''
    # -----------------------------------------------------

    result = eval_det(dts, gts, num_classes=1, iou_thr=0.1)
    # print(result)
    froc_label = "FROC-AUC: %.3f" % (result[0]["froc_auc"])
    draw_froc_curve(result[0]['froc'][0], result[0]['froc'][1],
                    label=froc_label, lw=2,
                    save_path=os.path.join(out_dir, save_dir))
