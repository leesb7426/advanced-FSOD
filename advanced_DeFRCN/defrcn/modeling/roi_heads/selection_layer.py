import cv2
import torch
import logging
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import batched_nms
from defrcn.dataloader import build_detection_test_loader
from defrcn.evaluation.archs import resnet101
import torch.nn.functional as F
from .util import *

logger = logging.getLogger(__name__)


class PrototypicalSelectionBlock:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.imagenet_model = self.build_model()
        for p in self.imagenet_model.parameters():
            p.requires_grad = False

        self.roi_pooler = ROIPooler(output_size=(1, 1),
                                    scales=(1 / 32,),
                                    sampling_ratio=(0),
                                    pooler_type="ROIAlignV2")

        self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
        self.mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
        self.std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)

        self.predefined_dict()

    def build_model(self):
        logger.info("Loading ImageNet Pre-train Model from {}".format(self.cfg.TEST.PCB_MODELPATH))
        if self.cfg.TEST.PCB_MODELTYPE == 'resnet':
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model

    def extract_conv_features(self, img):
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255. - self.mean) / self.std]
        images = ImageList.from_tensors(images, 0)
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW

        return conv_feature

    def extract_activation_vectors(self, conv_feature, boxes):
        """
        :param img:
        :param boxes:
        :return:
        """

        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)
        activation_vectors = self.imagenet_model.fc(box_features)

        return activation_vectors

    def predefined_dict(self):

        image_size_dict, boxes_dict, conv_feature_dict = {}, {}, {}
        all_activation_vectors, all_labels = [], []
        for index in range(len(self.dataloader.dataset)):
            inputs = [self.dataloader.dataset[index]]
            inst = inputs[0]['instances']
            file_name = inputs[0]['file_name']
            assert len(inputs) == 1
            # load support images and gt-boxes
            img = cv2.imread(file_name)  # BGR
            img_size = img.shape[:-1]
            ratio = img_size[0] / inst.image_size[0]

            inst.gt_boxes.tensor = inst.gt_boxes.tensor * ratio
            boxes = [x["instances"].gt_boxes.to(self.device) for x in inputs]
            # for x in inputs:
            #     for box in x["instances"].gt_boxes.tensor:
            #         cv2.rectangle(img,
            #                       (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            #                       (0, 0, 255), 2)
            #     cv2.imwrite('temp.png', img)
            # exit()
            gt_classes = [x['instances'].gt_classes.to(self.device) for x in inputs]

            # extract roi features
            conv_feature = self.extract_conv_features(img)
            activation_vectors = self.extract_activation_vectors(conv_feature, boxes)

            if image_size_dict.get(file_name) is None:
                image_size_dict[file_name] = img_size
                conv_feature_dict[file_name] = conv_feature
            if boxes_dict.get(file_name) is None:
                boxes_dict[file_name] = Boxes.cat(boxes).tensor
            else:
                boxes_dict[file_name] = torch.cat((boxes_dict.get(file_name),
                                                   Boxes.cat(boxes).tensor), dim=0)

            all_activation_vectors.append(activation_vectors)
            all_labels.append(torch.cat(gt_classes, dim=0))

        # concat
        all_activation_vectors = torch.cat(all_activation_vectors, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        assert all_activation_vectors.shape[0] == all_labels.shape[0]

        _, indices = all_labels.sort()
        all_activation_vectors = all_activation_vectors[indices]

        self.image_size_dict = image_size_dict
        self.boxes_dict = boxes_dict
        self.conv_feature_dict = conv_feature_dict
        self.activation_vectors = all_activation_vectors.unsqueeze(dim=1)
        self.num_shot = int(torch.sum(all_labels == 0))

    def get_similarity_and_indices(self, conv_feature, boxes):
        activation_vectors = self.extract_activation_vectors(
            conv_feature, [Boxes(boxes)]).unsqueeze(dim=0)
        similarity = F.cosine_similarity(self.activation_vectors, activation_vectors, dim=-1)
        del activation_vectors
        max_similarity, indices = torch.max(similarity, dim=0)
        del similarity

        return max_similarity, indices

    # def get_index(self,
    #               file_name,
    #               img,
    #               anchors,
    #               matched_gt_boxes,
    #               criteria_value=0.7):
    #
    #     image_size = self.image_size_dict.get(file_name)
    #     boxes = self.boxes_dict.get(file_name)
    #     conv_feature = self.conv_feature_dict.get(file_name)
    #
    #     ratio = image_size[0] / img.shape[1]
    #     refined_anchors = anchors * ratio
    #     refined_matched_gt_boxes = matched_gt_boxes * ratio
    #
    #     flip = is_flip(image_size, boxes, refined_matched_gt_boxes[0])
    #     if flip:
    #         refined_anchors = refined_anchors[:, [2, 1, 0, 3]]
    #         refined_anchors[:, 0] = image_size[1] - refined_anchors[:, 0]
    #         refined_anchors[:, 2] = image_size[1] - refined_anchors[:, 2]
    #
    #     iou_based_anchors, iou_based_gt_boxes = pairwise_iou_by_UOM(boxes, refined_anchors)
    #     max_similarity, indices_for_labels = self.get_similarity_and_indices(conv_feature, refined_anchors)
    #     _, indices_for_labels_of_gt_boxes = self.get_similarity_and_indices(conv_feature, refined_matched_gt_boxes)
    #
    #     index_by_iou_for_in = torch.sum(torch.stack((iou_based_anchors == 1.,
    #                                                  iou_based_gt_boxes == 1.), dim=0), dim=0) == 1
    #     index_by_iou_for_out = torch.sum(torch.stack((iou_based_anchors == 0.,
    #                                                   iou_based_gt_boxes == 0.), dim=0), dim=0) == 2
    #     index_by_iou_for_vague = torch.sum(torch.stack((index_by_iou_for_in,
    #                                                     index_by_iou_for_out), dim=0), dim=0) == 0
    #     index_by_similarity = max_similarity > criteria_value
    #     index_by_labels = (indices_for_labels // self.num_shot) != \
    #                       (indices_for_labels_of_gt_boxes // self.num_shot)
    #     index_by_indices = indices_for_labels != indices_for_labels_of_gt_boxes
    #
    #     index_in = torch.sum(torch.stack((index_by_iou_for_in,
    #                                       index_by_similarity,
    #                                       index_by_labels), dim=0), dim=0) == 3
    #     index_out = torch.sum(torch.stack((index_by_iou_for_out,
    #                                        index_by_similarity), dim=0), dim=0) == 2
    #     index_vague = torch.sum(torch.stack((index_by_iou_for_vague,
    #                                          index_by_similarity,
    #                                          index_by_indices,
    #                                          iou_based_gt_boxes <= 0.3), dim=0), dim=0) == 4
    #
    #     return index_in, index_out, index_vague, indices_for_labels
    #
    # def get_additional(self, anchors,
    #                    pred_objectness_logits,
    #                    indices_for_labels,
    #                    index,
    #                    criteria_value=0.7):
    #     keep = batched_nms(anchors[index],
    #                        pred_objectness_logits[index],
    #                        indices_for_labels[index], criteria_value)
    #     keep = keep[:int((keep.shape[0] + 1) / 2)]
    #     additional_gt_labels, additional_gt_boxes, additional_gt_scores = \
    #         averaging(anchors[index][keep], pred_objectness_logits[index][keep],
    #                   indices_for_labels[index][keep], self.cfg.DDA.WEIGHTED)
    #     additional_gt_labels = additional_gt_labels // self.num_shot
    #
    #     return additional_gt_labels, additional_gt_boxes, additional_gt_scores
    #
    # def generate_gt(self,
    #                 file_names,
    #                 images,
    #                 anchors,
    #                 gt_labels,
    #                 matched_gt_boxes,
    #                 pred_objectness_logits,
    #                 criteria_value=0.7):
    #
    #     additional_gt_labels = []
    #     additional_gt_boxes = []
    #     additional_gt_scores = []
    #
    #     for file_name_i, img_i, anchors_i, gt_labels_i, matched_gt_boxes_i, pred_objectness_logits_i in \
    #             zip(file_names, images, anchors, gt_labels, matched_gt_boxes, pred_objectness_logits):
    #
    #         anchors_i.clip(img_i.shape[1:])
    #         area_i = anchors_i.area()
    #         anchors_i = anchors_i.tensor
    #
    #         index_by_area = area_i > 0
    #         index_by_logits = pred_objectness_logits_i > 0
    #         index = torch.sum(torch.stack((index_by_area,
    #                                        index_by_logits), dim=0), dim=0) == 2
    #
    #         anchors_i = anchors_i[index]
    #         matched_gt_boxes_i = matched_gt_boxes_i[index]
    #         pred_objectness_logits_i = pred_objectness_logits_i[index]
    #
    #         if torch.sum(index) == 0:
    #             gt_labels_i = gt_labels_i[index]
    #             gt_boxes_i = matched_gt_boxes_i
    #             gt_scores_i = pred_objectness_logits_i
    #
    #         else:
    #             index_in, index_out, index_vague, indices_for_labels = \
    #                 self.get_index(file_name_i, img_i, anchors_i, matched_gt_boxes_i)
    #
    #             gt_labels_in, gt_boxes_in, gt_scores_in = \
    #                 self.get_additional(anchors_i, pred_objectness_logits_i, indices_for_labels, index_in)
    #             gt_labels_out, gt_boxes_out, gt_scores_out = \
    #                 self.get_additional(anchors_i, pred_objectness_logits_i, indices_for_labels, index_out)
    #             gt_labels_vague, gt_boxes_vague, gt_scores_vague = \
    #                 self.get_additional(anchors_i, pred_objectness_logits_i, indices_for_labels, index_vague)
    #
    #             gt_labels_i = torch.cat((gt_labels_in, gt_labels_out, gt_labels_vague), dim=0)
    #             gt_boxes_i = torch.cat((gt_boxes_in, gt_boxes_out, gt_boxes_vague), dim=0)
    #             gt_scores_i = torch.cat((gt_scores_in, gt_scores_out, gt_scores_vague), dim=0)
    #
    #         additional_gt_labels.append(gt_labels_i)
    #         additional_gt_boxes.append(gt_boxes_i)
    #         additional_gt_scores.append(gt_scores_i)
    #
    #     return additional_gt_labels, additional_gt_boxes, additional_gt_scores

    def get_index(self,
                  file_name,
                  img,
                  anchors,
                  matched_gt_boxes,
                  criteria_value=0.9):

        image_size = self.image_size_dict.get(file_name)
        boxes = self.boxes_dict.get(file_name)
        conv_feature = self.conv_feature_dict.get(file_name)

        ratio = image_size[0] / img.shape[1]
        refined_anchors = anchors * ratio
        refined_matched_gt_boxes = matched_gt_boxes * ratio

        flip = is_flip(image_size, boxes, refined_matched_gt_boxes[0])
        if flip:
            refined_anchors = refined_anchors[:, [2, 1, 0, 3]]
            refined_anchors[:, 0] = image_size[1] - refined_anchors[:, 0]
            refined_anchors[:, 2] = image_size[1] - refined_anchors[:, 2]

        iou_based_anchors, iou_based_gt_boxes = pairwise_iou_by_UOM(boxes, refined_anchors)
        max_similarity, indices_for_labels = self.get_similarity_and_indices(conv_feature, refined_anchors)
        _, indices_for_labels_of_gt_boxes = self.get_similarity_and_indices(conv_feature, refined_matched_gt_boxes)

        index_by_iou_based_anchors = iou_based_anchors < 0.3
        index_by_iou_based_gt_boxes = iou_based_gt_boxes < 0.3
        index_by_similarity = max_similarity > criteria_value

        index = torch.sum(torch.stack((index_by_iou_based_anchors,
                                       index_by_iou_based_gt_boxes,
                                       index_by_similarity), dim=0), dim=0) == 3

        return index, max_similarity, indices_for_labels

    def get_additional(self, anchors,
                       pred_objectness_logits,
                       indices_for_labels,
                       max_similarity,
                       index,
                       criteria_value=0.7):
        keep = batched_nms(anchors[index],
                           pred_objectness_logits[index] + torch.sigmoid(max_similarity)[index],
                           indices_for_labels[index], criteria_value)
        keep = keep[:int((keep.shape[0] + 1) / 2)]
        gt_labels, gt_boxes, gt_scores = \
            averaging(anchors[index][keep],
                      pred_objectness_logits[index][keep] + torch.sigmoid(max_similarity)[index][keep],
                      indices_for_labels[index][keep], self.cfg.DDA.WEIGHTED)
        gt_labels = gt_labels // self.num_shot

        return gt_labels, gt_boxes, gt_scores

    def generate_gt(self,
                    file_names,
                    images,
                    anchors,
                    gt_labels,
                    matched_gt_boxes,
                    pred_objectness_logits,
                    criteria_value=0.7):

        additional_gt_labels = []
        additional_gt_boxes = []
        additional_gt_scores = []

        for file_name_i, img_i, anchors_i, gt_labels_i, matched_gt_boxes_i, pred_objectness_logits_i in \
                zip(file_names, images, anchors, gt_labels, matched_gt_boxes, pred_objectness_logits):

            anchors_i.clip(img_i.shape[1:])
            area_i = anchors_i.area()
            anchors_i = anchors_i.tensor

            index_by_area = area_i > 0
            index_by_logits = pred_objectness_logits_i > 0
            index = torch.sum(torch.stack((index_by_area,
                                           index_by_logits), dim=0), dim=0) == 2

            anchors_i = anchors_i[index]
            matched_gt_boxes_i = matched_gt_boxes_i[index]
            pred_objectness_logits_i = pred_objectness_logits_i[index]

            if torch.sum(index) == 0:
                gt_labels_i = gt_labels_i[index]
                gt_boxes_i = matched_gt_boxes_i
                gt_scores_i = pred_objectness_logits_i

            else:
                index_i, max_similarity, indices_for_labels = \
                    self.get_index(file_name_i, img_i, anchors_i, matched_gt_boxes_i)
                gt_labels_i, gt_boxes_i, gt_scores_i = \
                    self.get_additional(anchors_i, pred_objectness_logits_i, indices_for_labels, max_similarity, index_i)

            additional_gt_labels.append(gt_labels_i)
            additional_gt_boxes.append(gt_boxes_i)
            additional_gt_scores.append(gt_scores_i)

        return additional_gt_labels, additional_gt_boxes, additional_gt_scores


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

