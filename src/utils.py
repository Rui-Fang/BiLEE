import torch
from collections import OrderedDict


def js_div(p, q):
    non_zero_p = (p.sum(dim=-1) != 0.0) | (q.sum(dim=-1) != 0.0)
    if non_zero_p.sum() == 0:
        return torch.zeros_like(non_zero_p).to(p.device).float()
    p, q = p[non_zero_p], q[non_zero_p]
    kl = torch.nn.KLDivLoss(reduction="none", log_target=True)
    ps, qs = p.view(-1, p.size(-1)).softmax(-1), q.view(-1, q.size(-1)).softmax(-1)
    m = (0.5 * (ps + qs)).log()
    jsd = 0.5 * (kl(m, p.log_softmax(-1)) + kl(m, q.log_softmax(-1))).sum(dim=-1)
    zeros = torch.zeros_like(non_zero_p).to(p.device).float()
    zeros[non_zero_p] = jsd
    return zeros


def calc_skip_position(skip_position):
    token_num = (skip_position != -1.0).int().sum(dim=0)
    minous_one_num = (skip_position == -1.0).int().sum(dim=0)
    sp_mean = (skip_position.sum(dim=0) + minous_one_num) / token_num
    sp_total_mean = (sp_mean[~sp_mean.isnan()] * token_num[~sp_mean.isnan()]).sum() / token_num.sum()
    return {"mean": sp_mean[~sp_mean.isnan()], "total_mean": sp_total_mean}


def skip_position_analysis(layer_prediction, skip_position, target_mask, final_logits):
    device = layer_prediction.device
    final_logits = final_logits.to(device)
    layer_num = layer_prediction.size(-1)
    target_mask_expended = target_mask.unsqueeze(-1).expand(-1, -1, layer_num)
    layer_prediction = layer_prediction.masked_fill_(~target_mask_expended.bool(), -1)
    true_predict = final_logits.argmax(dim=-1).masked_fill(~target_mask.bool(), -1)
    true_predict_expanded = true_predict.unsqueeze(-1).expand_as(layer_prediction)
    correct_predict = (layer_prediction == true_predict_expanded).float().detach()  # QUESTION： true_predict or labels

    skip_position[~target_mask.bool()] = -1
    sp_dict = calc_skip_position(skip_position)

    oracle_skip = torch.ones_like(skip_position) * layer_num
    for i in range(layer_num):
        o_skipped = correct_predict[:, :, i]
        skip_mask = o_skipped.bool() & (oracle_skip == layer_num).bool()
        oracle_skip[skip_mask] = i
    oracle_skip[~target_mask.bool()] = -1

    mask = oracle_skip != -1
    oracle_mean = (oracle_skip * mask).sum(dim=0) / mask.sum(dim=0)  # [batch_size]
    output_dict = {
        "skip_position": sp_dict["total_mean"],
        "oracle_mean": oracle_mean[~oracle_mean.isnan()],
        "oracle_skip_position": calc_skip_position(oracle_skip[:, :5])["total_mean"],
    }
    return oracle_skip, correct_predict, output_dict


def jsd_per_layer(layer_logits, target_logits, valid_indices=None):
    seq_length, counted_layer_num = layer_logits.shape[1:3]
    jsd_loss = js_div_loss()
    jsd_list = []
    for i in range(counted_layer_num):
        layer_logit = layer_logits[:, :, i, :]
        target_logit = target_logits[:, :, i, :] if len(target_logits.shape) == 4 else target_logits
        if valid_indices is not None:
            jsd_layer = [jsd_loss(layer_logit[:, j, valid_indices[j]], target_logit[:, j, valid_indices[j]].detach()) for j in range(seq_length)]
        else:
            jsd_layer = [jsd_loss(layer_logit[:, j, :], target_logit[:, j, :].detach()) for j in range(seq_length)]
        jsd_list.append(torch.stack(jsd_layer))
    result = torch.stack(jsd_list).permute(2, 1, 0)
    result[result.isnan()] = 0.0
    return result


class training_buffers:
    def __init__(self, item_list) -> None:
        self.buffer = [[] for item in item_list]

    def add(self, item_lists):
        for i, item in enumerate(item_lists):
            self.buffer[i].append(item.detach().clone().cpu())

    def get_all(self):
        result = [torch.cat(items, dim=0) for items in self.buffer]
        del self.buffer
        return result


class js_div_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, q):
        return js_div(p, q)


class embedding_switch:
    """
    Class to switch between different symentic embedding
    """

    def __init__(self, oneD_vocab_length, twoD_vocab_length, max_seq_length=10) -> None:
        bz = 1
        seq_length = max_seq_length
        vocab_size = twoD_vocab_length
        self.twoD_length = twoD_vocab_length

        self.oneD_length = oneD_vocab_length
        output_vocab_size = oneD_vocab_length
        valid_indices = torch.arange(output_vocab_size).view(1, -1)
        pos_indices = torch.arange(seq_length).view(-1, 1) * output_vocab_size

        # valid_indices = torch.arange(10).view(1, -1)
        # pos_indices = torch.arange(seq_length).view(-1, 1) * 10
        valid_indices = valid_indices + pos_indices + 2  # [seq_length, 10]
        ones_indices = torch.ones(seq_length, 1).to(valid_indices.device)
        valid_indices = torch.cat((valid_indices, ones_indices), dim=-1).long()
        self.valid_indices = valid_indices

        valid_indices[-1, :] = torch.ones(1, output_vocab_size + 1)
        # valid_indices[-1,:] = torch.ones(1, 11)
        valid_indices = valid_indices.unsqueeze(0).repeat([1, 1, 1])  # [bz, 10, sl]
        zero_mask = torch.zeros(1, seq_length, vocab_size)
        mask = zero_mask - 1e9
        self.logit_mask = mask.scatter_(-1, valid_indices, zero_mask)  # scatter explained: https://zhuanlan.zhihu.com/p/339043454

    def one_to_two(self, one_D_embedding):
        """
        oneD_embedding: [... , seq_length, self.oneD_length]
        return: [... , seq_length, self.twoD_length]
        """
        one_D_shape = one_D_embedding.shape
        two_D_shape = one_D_shape
        two_D_shape[-1] = self.twoD_length
        one_D_embedding = one_D_embedding.view(-1, one_D_shape[-2], one_D_shape[-1])
        two_D_embedding = torch.zeros_like(two_D_shape, device=one_D_embedding.device).view(-1, one_D_shape[-2], self.twoD_length)
        two_D_embedding.scatter_(-1, self.valid_indices, one_D_embedding)
        two_D_embedding = two_D_embedding.view(two_D_shape)
        # twoD_embedding = torch.softmax(oneD_embedding, dim=-1)
        return two_D_embedding

    def two_to_one(self, twoD_embedding):
        """
        twoD_embedding: [... , seq_length, vocab_size]
        """
        oneD_embedding = twoD_embedding - self.logit_mask
        # oneD_embedding = torch.softmax(twoD_embedding, dim=-1)
        return oneD_embedding
