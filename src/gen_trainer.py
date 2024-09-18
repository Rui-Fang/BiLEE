import os
import numpy as np
import pandas as pd
import lightning as pl
import torch
from torch.utils.data import DataLoader
from loguru import logger

from utils import training_buffers
from matplotlib import pyplot as plt
from exit_models import T5FineTunerSmall
from einops import rearrange


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = []
        for dataset in datasets:
            if dataset is not None:
                self.datasets.append(dataset)

    def __getitem__(self, i):
        return list(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class GTDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path):
        logger.info("Using GT data for training.")
        self.data_path = data_path
        self.args = args
        self.file_names = os.listdir(self.data_path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data, target_mask, label = torch.load(os.path.join(self.data_path, self.file_names[idx]), map_location=lambda storage, loc: storage)
        return data.type(torch.float32), target_mask, label


class savedDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path):
        logger.info("Using gen data for training.")
        self.data_path = data_path
        self.args = args
        self.file_names = os.listdir(self.data_path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.data_path, self.file_names[idx]), map_location=lambda storage, loc: storage).type(torch.float32)


class adaptive_T5(T5FineTunerSmall):
    # def _set_trainer_params(self):
    #     self.token_weights = torch.tensor([1, 1, 1, 1, 0.1, 0, 0, 0, 0, 0], device=self.device)
    #     self.token_weights = self.token_weights / self.token_weights.sum() * self.args.max_output_length
    #     layer_weights = torch.arange(self.args.num_decoder_layers, device=self.device) + 1
    #     layer_weights = layer_weights.flip(dims=[0])
    #     self.layer_weights = layer_weights / layer_weights.sum() * self.args.num_decoder_layers
    #     logger.info(f"token_weights: {self.token_weights}")
    #     logger.info(f"layer_weights: {self.layer_weights}")

    def __init__(self, args, train=True):
        super().__init__(args, train)

    def train_dataloader(self):
        logger.info("load training data and create training loader.")
        train_dataset = ConcatDataset(
            savedDataset(self.args, data_path="src/all_hs") if self.args.gen_alpha != 0.0 else None,
            GTDataset(self.args, data_path="src/hs_gq") if self.args.gen_alpha != 1.0 else None,
        )
        dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            drop_last=False,
            shuffle=True,
            num_workers=2,
            prefetch_factor=1,
        )
        return dataloader

    def on_train_epoch_start(self):
        self.finished_sanity_check = True
        self.set_threshold(torch.zeros_like(self.get_threshold()))
        if self.args.decoder_skipping:
            self.buffer = training_buffers(["skip_score", "skip_decision", "layer_prediction", "correct_predict", "jsd"])

    def training_step(self, batch, batch_idx):
        alpha = self.args.gen_alpha
        beta = self.args.cel_beta
        device = self.device
        decoder = self.model.decoder
        adaptor_loss_gt = 0
        adaptor_loss_gen = 0
        self.counted_layer_num = len(decoder.block)

        def shared_process(hidden_states, target_mask, confidence_weighting=False):
            final_hidden_states = hidden_states[:, :, -1:, :]
            final_hidden_states = decoder.dropout(decoder.final_layer_norm(final_hidden_states)) * (self.args.d_model**-0.5)
            final_logits_unmask = self.model.lm_head(final_hidden_states)
            if "HW" in self.args.classifier_type or "single" in self.args.classifier_type:
                layer_logits = torch.stack(
                    [decoder.skip_adaptor(all_hidden_states[:, :, i, :], i).squeeze(1) for i in range(len(decoder.block))],
                    dim=2,
                )
            else:
                layer_logits = torch.stack(
                    [block.skip_fixer(all_hidden_states[:, :, i, :], None).squeeze(1) for i, block in enumerate(decoder.block)],
                    dim=2,
                )
            return (
                self.kl_loss(
                    final_logits_unmask.squeeze(),
                    layer_logits,
                    target_mask,
                    token_valid=True,
                    not_embd_valid=False,
                    confidence_weighting=confidence_weighting,
                ),
                layer_logits,
            )

        if alpha != 0.0:  # gen training
            gen_data = batch.pop(0)
            all_hidden_states = rearrange(gen_data, "MB B L T D -> (MB B) T L D").to(device)  # [batch, seq, layer, d_model]
            target_mask_g = torch.ones(all_hidden_states.shape[:2]).bool().to(device)
            adaptor_loss_gen, _ = shared_process(all_hidden_states, target_mask_g, confidence_weighting=True)
            self.log(f"train/fixer_loss_gen", adaptor_loss_gen, on_step=True, prog_bar=True, sync_dist=True)

        if alpha != 1.0:
            gt_data = batch.pop(0)
            all_hidden_states = rearrange(gt_data[0], "MB B T L D -> (MB B) T L D").to(device)  # [batch, seq, layer, d_model]
            target_mask = rearrange(gt_data[1], "MB B T->(MB B) T").to(device)
            kd_loss, layer_logits = shared_process(all_hidden_states, target_mask)
            cel_loss = self.cel_loss(gt_data[2].squeeze(0), layer_logits)
            adaptor_loss_gt = kd_loss * (1 - beta) + cel_loss * (beta)
            self.log(f"train/kd_loss", kd_loss, on_step=True, prog_bar=False, sync_dist=True)
            self.log(f"train/cel_loss", cel_loss, on_step=True, prog_bar=False, sync_dist=True)
            self.log(f"train/adaptor_loss_gt", adaptor_loss_gt, on_step=True, prog_bar=True, sync_dist=True)

        return {"loss": adaptor_loss_gt * (1 - alpha) + adaptor_loss_gen * alpha}

    def on_train_epoch_end(self):
        return

        ### TEST ###
        # valid_mask = self.model.logit_mask.to(device)

        # seq_length = target_mask_g.shape[1]
        # final_logits_masked = final_logits_unmask.squeeze() + (valid_mask[:, :seq_length].to(self.device))

        # valid_mask_expand = valid_mask[:, :seq_length].unsqueeze(2).expand(-1, -1, self.counted_layer_num, -1)

        # full_predict = final_logits_masked.max(dim=-1)[1]
        # layer_prediction = (layer_logits + valid_mask_expand).max(dim=-1)[1]

        # layer_logits_masked = layer_logits.masked_fill(valid_mask_expand != 0, float("-inf"))
        # layer_logits_masked_softmax = layer_logits_masked.softmax(dim=-1)
        # top2_values, _ = layer_logits_masked_softmax.topk(2, dim=-1)
        # score = (top2_values[:, :, :, 0] - top2_values[:, :, :, 1]).squeeze(1)
        # wrong_prediction = (layer_prediction != full_predict.unsqueeze(-1)).int().detach()
        # wrong_score = score
        # wrong_score[wrong_prediction > 0] = 0
        # wrong_score = wrong_score.max(dim=0)[0]
        # skip_threshold = self.get_threshold().T[:seq_length]
        # skip_threshold[skip_threshold < wrong_score] = wrong_score[skip_threshold < wrong_score]
        # skip_threshold_new = self.get_threshold().T
        # skip_threshold_new[:seq_length] = skip_threshold
        # self.set_threshold(skip_threshold_new.T)

        # skip_threshold = self.get_threshold()
        # skip_desicion = score >= skip_threshold.transpose(0, 1)[:seq_length]  # if i not in [0, 11] else torch.zeros_like(score).bool()
        # position = torch.ones_like(layer_prediction)[:,:,0]*self.counted_layer_num

        ### adjucted fixer loss
        # final_logits_masked = final_logits_masked.unqueeze(2).expand(-1, -1, self.counted_layer_num, -1)
        # inverse sofmetma

        ### score loss
        # jsd_masked = jsd_per_layer(layer_logits, final_logits_masked, self.model.valid_indices)
        # target_score = (1 - jsd_masked) * 0.9

        # mask = valid_mask[:, :seq_length].to(self.device)
        # l_logits_masked = layer_logits.transpose(1, 2).masked_fill(mask != 0, float("-inf")).transpose(1, 2).softmax(dim=-1)
        # top2_scores = l_logits_masked.topk(2, dim=-1)[0]
        # exit_scores = top2_scores[..., 0] - top2_scores[..., 1]
        # # score_loss = self.calc_BCE_loss(exit_scores, target_score, target_mask_g)
        # score_loss = F.mse_loss(exit_scores, target_score, reduction="none").sum(dim=-1).mean()
        # error = (exit_scores - target_score).abs()

        # self.log(f"train/score_loss", score_loss, on_step=True, prog_bar=True, sync_dist=True)
