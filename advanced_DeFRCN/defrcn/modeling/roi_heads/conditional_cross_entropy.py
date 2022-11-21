import torch
import torch.nn.functional as F


def conditional_cross_entropy(pred_logits,
                              gt_classes,
                              similarity_dict,
                              fg_similarity_dict,
                              reduction='mean'
                              ):
    if not(reduction in ['mean', 'sum']):
        print('plz check reduction type')
        exit()

    else:
        _, cls_num = pred_logits.shape
        bg_cls = cls_num - 1

        fg_indices = gt_classes != bg_cls
        bg_indices = gt_classes == bg_cls
        fg_classes = gt_classes[fg_indices]
        unique_of_fg_classes = torch.unique(fg_classes, sorted=True).tolist()

        index = []
        for cls in unique_of_fg_classes:
            similarity_with_cls = similarity_dict.get(cls)
            fg_similarity_with_cls = fg_similarity_dict.get(cls)
            fg_num = fg_similarity_with_cls.shape[0]
            fg_similarity_with_cls = fg_similarity_with_cls.unsqueeze(dim=-1)
            bg_similarity_with_cls = similarity_with_cls[:, bg_indices]
            # index.append(torch.sum((fg_similarity_with_cls > bg_similarity_with_cls), dim=0) > 0)
            index.append(torch.sum((fg_similarity_with_cls > bg_similarity_with_cls), dim=0) == fg_num)

        index = torch.sum(torch.stack(index, dim=1), dim=-1) == len(unique_of_fg_classes)

        fg_pred_logits = pred_logits[fg_indices]
        fg_gt_classes = gt_classes[fg_indices]
        fg_cross_entropy_loss = F.cross_entropy(fg_pred_logits, fg_gt_classes, reduction="sum")

        bg_pred_logits = pred_logits[bg_indices]
        bg_gt_classes = gt_classes[bg_indices]

        bg_pred_logits = bg_pred_logits[index]
        bg_gt_classes = bg_gt_classes[index]
        bg_cross_entropy_loss = F.cross_entropy(bg_pred_logits, bg_gt_classes, reduction="sum")
        total_batch = fg_gt_classes.shape[0] + bg_gt_classes.shape[0]
        print(pred_logits.shape[0],
              fg_gt_classes.shape[0],
              bg_gt_classes.shape[0],
              pred_logits.shape[0] - total_batch)
        print('\n')

        loss = fg_cross_entropy_loss + bg_cross_entropy_loss

        if reduction == "mean":
            loss = loss / total_batch

        return loss