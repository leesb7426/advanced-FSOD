import os
import cv2
import torch
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou


def is_flip(og_image_size, og_boxes, refined_gt_box):
    flipped_refined_gt_box = refined_gt_box[[2, 1, 0, 3]]
    flipped_refined_gt_box[0] = og_image_size[1] - flipped_refined_gt_box[0]
    flipped_refined_gt_box[2] = og_image_size[1] - flipped_refined_gt_box[2]
    count = 0
    flip = False
    for og_box in og_boxes:
        if torch.sum(torch.round(og_box) == torch.round(refined_gt_box)) == 4:
            count += 1
        elif torch.sum(torch.round(og_box) == torch.round(flipped_refined_gt_box)) == 4:
            count += 1
            flip = True

    if count == 1:
        return flip
    else:
        print('plz check is_flip function')
        exit()


def get_overlap_bbox(gt_boxes, anchors):
    stack_box = torch.stack((gt_boxes, anchors), dim=0)
    overlap_bbox_1, _ = torch.max(stack_box, dim=0)
    overlap_bbox_2, _ = torch.min(stack_box, dim=0)

    overlap_bbox = torch.stack((overlap_bbox_1[:, 0],
                                overlap_bbox_1[:, 1],
                                overlap_bbox_2[:, 2],
                                overlap_bbox_2[:, 3],), dim=1)

    return overlap_bbox


def pairwise_iou_by_UOM(gt_boxes, anchors):
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    gt_boxes_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    anchors_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])

    width_height = torch.min(gt_boxes[:, None, 2:], anchors[:, 2:]) - torch.max(
        gt_boxes[:, None, :2], anchors[:, :2]
    )  # [N,M,2]
    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    maximum_intersection, _ = torch.max(intersection, dim=0)

    iou_based_anchors = maximum_intersection / anchors_area
    iou_based_gt_boxes = intersection / gt_boxes_area.unsqueeze(dim=-1)
    iou_based_gt_boxes, _ = torch.max(iou_based_gt_boxes, dim=0)

    return iou_based_anchors, iou_based_gt_boxes


def scoring_voting(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, sigma_t=0.025):
    additional_gt_labels = []
    additional_gt_boxes = []
    for cls in torch.unique(classes):
        boxes_of_cls = boxes[classes == cls]
        scores_of_cls = torch.exp(scores[classes == cls])
        iou_of_cls = pairwise_iou(Boxes(boxes_of_cls), Boxes(boxes_of_cls))

        for i, iou in enumerate(iou_of_cls):
            p_i = torch.exp(-torch.pow(1 - iou, 2) / sigma_t)
            denominator = torch.sum(p_i * scores_of_cls)
            numerator = torch.sum((p_i * scores_of_cls).unsqueeze(dim=-1) * boxes_of_cls, dim=0)
            updated_box = numerator / denominator
            additional_gt_labels.append(cls)
            additional_gt_boxes.append(updated_box)

    return additional_gt_labels, additional_gt_boxes


def averaging(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, weighted=False):
    while True:
        scores = torch.exp(scores)
        num_instance = boxes.shape[0]
        averaging_target_1 = pairwise_iou(Boxes(boxes), Boxes(boxes)) > 0.3
        averaging_target_2 = classes.unsqueeze(dim=0) == classes.unsqueeze(dim=-1)
        averaging_target = (averaging_target_1 * averaging_target_2).unsqueeze(dim=-1)
        if num_instance == torch.sum(averaging_target):
            break

        if weighted:
            scores = scores.unsqueeze(dim=0).unsqueeze(dim=-1)
            boxes = boxes.unsqueeze(dim=0)

            scores = averaging_target * scores
            boxes = torch.sum(boxes * scores, dim=1) / torch.sum(scores, dim=1)
            scores = (torch.sum(scores, dim=1) / torch.sum(averaging_target, dim=1)).squeeze(dim=-1)

            boxes_classes_concat = torch.cat((boxes, classes.unsqueeze(dim=-1)), dim=-1)
            _, inverse_indices = torch.unique(boxes_classes_concat, return_inverse=True, dim=0)

        else:
            boxes = boxes.unsqueeze(dim=0)
            boxes = torch.sum(boxes * averaging_target, dim=1) / torch.sum(averaging_target, dim=1)

            boxes_classes_concat = torch.cat((boxes, classes.unsqueeze(dim=-1)), dim=-1)
            _, inverse_indices = torch.unique(boxes_classes_concat, return_inverse=True, dim=0)

        scores_list = []
        boxes_list = []
        classes_list = []
        for i in torch.unique(inverse_indices):
            scores_list.append(torch.mean(scores[inverse_indices == i]))
            boxes_list.append(torch.unique(boxes[inverse_indices == i], dim=0))
            classes_list.append(torch.unique(classes[inverse_indices == i]))
        scores = torch.stack(scores_list, dim=0)
        boxes = torch.cat(boxes_list, dim=0)
        classes = torch.cat(classes_list, dim=0)

    return classes, boxes, scores


def passing(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor):
    if classes.shape[0] != 0:
        scores_list = []
        boxes_list = []
        classes_list = []
        for cls in torch.unique(classes):
            scores_list.append(scores[classes == cls])
            boxes_list.append(boxes[classes == cls])
            classes_list.append(classes[classes == cls])
        scores = torch.cat(scores_list, dim=0)
        boxes = torch.cat(boxes_list, dim=0)
        classes = torch.cat(classes_list, dim=0)

    return classes, boxes, scores


def visulaization(dir_name,
                  file_name,
                  og_image_size,
                  ratio,
                  og_boxes,
                  additional_gt_boxes,
                  additional_gt_labels,
                  flip,
                  anchors=None,
                  iou=None):
    if not(os.path.exists(dir_name)):
        os.makedirs(dir_name)

    img = cv2.imread(file_name)
    for box in og_boxes:
        # if flip:
        #     box = box * ratio
        #     box = [og_image_size[1] - max(int(box[0]), 0),
        #            max(int(box[1]), 0),
        #            og_image_size[1] - min(int(box[2]), img.shape[1]),
        #            min(int(box[3]), img.shape[0])]
        #
        # else:
        #     box = box * ratio
        #     box = [max(int(box[0]), 0),
        #            max(int(box[1]), 0),
        #            min(int(box[2]), img.shape[1]),
        #            min(int(box[3]), img.shape[0])]
        #
        # cv2.rectangle(img,
        #               (box[0], box[1]), (box[2], box[3]),
        #               (0, 255, 0), 2)

        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        cv2.rectangle(img,
                      (box[0], box[1]), (box[2], box[3]),
                      (0, 255, 0), 2)

    if iou is not None:
        for box, gt_label, iou in zip(additional_gt_boxes, additional_gt_labels, iou):
            if flip:
                box = box * ratio
                box = [og_image_size[1] - max(int(box[0]), 0),
                       max(int(box[1]), 0),
                       og_image_size[1] - min(int(box[2]), img.shape[1]),
                       min(int(box[3]), img.shape[0])]

            else:
                box = box * ratio
                box = [max(int(box[0]), 0),
                       max(int(box[1]), 0),
                       min(int(box[2]), img.shape[1]),
                       min(int(box[3]), img.shape[0])]

            cv2.rectangle(img,
                          (box[0], box[1]), (box[2], box[3]),
                          (0, 0, 255), 2)

            category = ["bird", "bus", "cow", "motorbike", "sofa"]
            cv2.putText(img,
                        category[gt_label] + str(round(iou.item(), 2)),
                        (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2
                        )

    else:
        for box, gt_label in zip(additional_gt_boxes, additional_gt_labels):
            if flip:
                box = box * ratio
                box = [og_image_size[1] - max(int(box[0]), 0),
                       max(int(box[1]), 0),
                       og_image_size[1] - min(int(box[2]), img.shape[1]),
                       min(int(box[3]), img.shape[0])]

            else:
                box = box * ratio
                box = [max(int(box[0]), 0),
                       max(int(box[1]), 0),
                       min(int(box[2]), img.shape[1]),
                       min(int(box[3]), img.shape[0])]

            cv2.rectangle(img,
                          (box[0], box[1]), (box[2], box[3]),
                          (0, 0, 255), 2)

            category = ["bird", "bus", "cow", "motorbike", "sofa"]
            cv2.putText(img,
                        category[gt_label],
                        (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2
                        )

    # if anchors is not None:
    #     for box in anchors:
    #         if flip:
    #             box = box * ratio
    #             box = [og_image_size[1] - max(int(box[0]), 0),
    #                    max(int(box[1]), 0),
    #                    og_image_size[1] - min(int(box[2]), img.shape[1]),
    #                    min(int(box[3]), img.shape[0])]
    #
    #         else:
    #             box = box * ratio
    #             box = [max(int(box[0]), 0),
    #                    max(int(box[1]), 0),
    #                    min(int(box[2]), img.shape[1]),
    #                    min(int(box[3]), img.shape[0])]
    #
    #         cv2.rectangle(img,
    #                       (box[0], box[1]), (box[2], box[3]),
    #                       (255, 0, 0), 2)

    if not(os.path.exists(os.path.join('temp', file_name.split('/')[-1]))):
        cv2.imwrite(os.path.join('temp', file_name.split('/')[-1]), img)