from functools import partial
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torch
import torchvision


class adaptor_loss_KL(torch.nn.Module):
    def __init__(self, logit_mask=None, t_weights=None, l_weights=None) -> None:
        super().__init__()
        self.logit_mask = logit_mask
        weights_matrix = torch.mul(l_weights.unsqueeze(0), t_weights.unsqueeze(0).T)
        self.weights_matrix = weights_matrix / weights_matrix.sum() * len(weights_matrix.view(-1))

    def forward(
        self,
        f_logits_nomask,
        layer_logits,
        t_valid_mask,
        token_valid=True,
        not_embd_valid=False,
        confidence_weighting=False,
    ):
        f_logits = f_logits_nomask
        counted_layer_num = layer_logits.shape[2]
        if self.logit_mask is not None and not not_embd_valid:
            logit_mask_e = self.logit_mask[:, : f_logits.shape[1]].to(f_logits.device)
            f_logits = f_logits + logit_mask_e
            layer_logits = layer_logits + repeat(logit_mask_e, "1 T D -> 1 T L D", L=counted_layer_num)
        available_token_num = t_valid_mask.sum(dim=1).max()
        target_logits = repeat(f_logits, "B T D -> B T L D", L=counted_layer_num)

        kl_loss = F.kl_div(layer_logits.log_softmax(dim=-1), target_logits.softmax(dim=-1), reduction="none")
        kl_loss = kl_loss.sum(dim=-1)

        if token_valid:
            t_valid_mask_e = repeat(t_valid_mask, "B T -> B T L", L=counted_layer_num).int()
            kl_loss = kl_loss * t_valid_mask_e
        weights_matrix = self.weights_matrix.to(layer_logits.device)
        kl_loss = kl_loss[:, :available_token_num] * weights_matrix[:available_token_num].unsqueeze(0)
        if confidence_weighting:
            confidence = f_logits.softmax(dim=-1).max(dim=-1)[0]
            kl_loss = kl_loss * confidence.unsqueeze(-1)
        return kl_loss.mean()


class adaptor_loss_CEL(torch.nn.Module):
    def __init__(self, logit_mask=None, t_weights=None, l_weights=None) -> None:
        super().__init__()
        self.logit_mask = logit_mask
        weights_matrix = torch.mul(l_weights.unsqueeze(0), t_weights.unsqueeze(0).T)
        self.weights_matrix = weights_matrix / weights_matrix.sum() * len(weights_matrix.view(-1))

    def forward(self, gt, layer_logits):
        counted_layer_num = layer_logits.shape[2]
        gt_expand = repeat(gt, "B T -> B T L", L=counted_layer_num)
        loss_fcn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        if self.logit_mask is not None:
            logits_mask = repeat(self.logit_mask, "1 T D -> 1 T L D", L=counted_layer_num).to(layer_logits.device)
            layer_logits += logits_mask
        weights_matrix = self.weights_matrix.to(layer_logits.device)
        loss = loss_fcn(rearrange(layer_logits, "B T L D -> (B T L) D"), gt_expand.reshape(-1)).view(gt_expand.shape)
        loss = loss * weights_matrix.unsqueeze(0)
        return loss.mean()


# def adaptor_loss_KL(
#     f_logits_nomask,
#     layer_logits,
#     t_valid_mask,
#     logit_mask=None,
#     token_valid=True,
#     t_weights=None,
#     l_weights=None,
#     confidence_weighting=False,
# ):
#     f_logits = f_logits_nomask
#     counted_layer_num = layer_logits.shape[2]
#     if logit_mask is not None:
#         logits_mask = logit_mask[:, : f_logits.shape[1]].to(f_logits.device)
#         f_logits = f_logits_nomask + logit_mask[:, : f_logits.shape[1]].to(f_logits.device)
#         layer_logits = layer_logits + repeat(logits_mask, "1 T D -> 1 T L D", L=counted_layer_num)
#     available_token_num = t_valid_mask.sum(dim=1).max()
#     target_logits = repeat(f_logits, "B T D -> B T L D", L=counted_layer_num)

#     kl_loss = F.kl_div(layer_logits.log_softmax(dim=-1), target_logits.softmax(dim=-1), reduction="none")
#     kl_loss = kl_loss.sum(dim=-1)

#     if token_valid:
#         t_valid_mask_e = repeat(t_valid_mask, "B T -> B T L", L=counted_layer_num).int()
#         kl_loss = kl_loss * t_valid_mask_e
#     weights_matrix = torch.mul(l_weights.unsqueeze(0), t_weights.unsqueeze(0).T).to(layer_logits.device)
#     weights_matrix = weights_matrix / weights_matrix.sum() * len(weights_matrix.view(-1))
#     kl_loss = kl_loss[:, :available_token_num] * weights_matrix[:available_token_num].unsqueeze(0)
#     if confidence_weighting:
#         confidence = f_logits.softmax(dim=-1).max(dim=-1)[0]
#         kl_loss = kl_loss * confidence.unsqueeze(-1)
#     return kl_loss.mean()


# def adaptor_loss_CEL(gt, layer_logits, logit_mask=None, t_weights=None, l_weights=None):
#     counted_layer_num = layer_logits.shape[2]
#     gt_expand = repeat(gt, "B T -> B T L", L=counted_layer_num)
#     loss_fcn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
#     if logit_mask is not None:
#         logits_mask = repeat(logit_mask, "1 T D -> 1 T L D", L=counted_layer_num).to(layer_logits.device)
#         layer_logits += logits_mask
#     weights_matrix = torch.mul(l_weights.unsqueeze(0), t_weights.unsqueeze(0).T).to(layer_logits.device)
#     weights_matrix = weights_matrix / weights_matrix.sum() * len(weights_matrix.view(-1))
#     loss = loss_fcn(rearrange(layer_logits, "B T L D -> (B T L) D"), gt_expand.reshape(-1)).view(gt_expand.shape)
#     loss = loss * weights_matrix.unsqueeze(0)

#     return loss.mean()


def calc_FCL_loss(score, target_score, t_valid_mask, t_weights=None):
    flatten_to_2d = lambda x: x.reshape(-1, x.shape[-1])
    target_mask_flat = t_valid_mask.view(-1)
    # prophet_type = self.args.classifier_type
    # if prophet_type == "prophet_KLD":
    ### KLD prophecy loss
    target_score = flatten_to_2d(target_score)[target_mask_flat]
    threshold = target_score.mean(0) - target_score.std(0)
    negative_sample = torch.le(target_score, threshold)
    # target_score[negative_sample] = 0

    pos_weight = torch.ones_like(target_score)
    pos_weight[negative_sample] = (1 - (negative_sample.sum() / len(negative_sample.view(-1)))) * 4
    pos_weight[~negative_sample] = negative_sample.sum() / len(negative_sample.view(-1))

    # loss_fcn = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss_fcn = partial(torchvision.ops.sigmoid_focal_loss, reduction="none")
    loss = loss_fcn(flatten_to_2d(score)[target_mask_flat], target_score.float().detach())

    # ### CEL version
    # loss_fcn = torch.nn.CrossEntropyLoss(reduction="none")
    # score = score.unsqueeze(-1)

    if t_weights is not None:
        t_weights = t_weights.unsqueeze(-1).unsqueeze(0).expand(score.shape[0], -1, score.shape[-1])
        t_weights = flatten_to_2d(t_weights)[target_mask_flat]
        loss = loss * t_weights
    loss = (loss.sum(dim=1) * 1).mean()
    return loss
