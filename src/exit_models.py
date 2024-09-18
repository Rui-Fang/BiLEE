import copy
from functools import partial
import os
import pickle
import time
import numpy as np
import pandas as pd
import lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from main_utils import assert_all_frozen, dec_2d, numerical_decoder
from dataset import l1_query
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, T5Config, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F
from loguru import logger
from main_models import TreeBuilder, encode_single_newid, softmax, decode_token
from main_metrics import recall, MRR100
import warnings
from utils import js_div_loss, training_buffers, jsd_per_layer, skip_position_analysis, js_div, calc_skip_position
from matplotlib import pyplot as plt
from einops import rearrange, reduce, repeat
from losses import adaptor_loss_KL, adaptor_loss_CEL, calc_FCL_loss


class T5FineTunerSmall(pl.LightningModule):
    """
    Hand crafted smaller T5 fine-tuner, removed the unused parts for readability.
    """

    def _init_model(self):
        # Initialize Main model
        logger.info("Initializing main model")
        t5_config_student = copy.deepcopy(self.t5_config)
        model = T5ForConditionalGeneration(t5_config_student)
        logger.info("Model initialized, loading parameters")
        if not self.training:
            self.model = model
            return  # load ckpt in infer stage
        # Load ckpt or copy teacher params
        if self.args.ckpt_path is not None:
            # Training from ckpt
            state_dict = torch.load(self.args.ckpt_path, map_location=torch.device("cpu"))

            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]

            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            ### TEMP ###
            # state_dict.pop("decode_embeddings.weight")
            # state_dict.pop("lm_head.weight")
            # state_dict.pop("decoder.embed_tokens.weight")
            # if not same index, remove the mismatched params
            if self.args.semantic_identifier != 2:
                state_dict.pop("decode_embeddings.weight")
                state_dict.pop("lm_head.weight")
                state_dict.pop("decoder.embed_tokens.weight")

            logger.info(model.load_state_dict(state_dict, strict=False))
            del state_dict
            logger.info(f"ckpt loaded from {self.args.ckpt_path}")
        else:
            if "base" in self.args.model_info:
                pretrain_model = T5ForConditionalGeneration.from_pretrained("t5-base")
            elif "large" in self.args.model_info:
                pretrain_model = T5ForConditionalGeneration.from_pretrained("t5-large")
            else:
                raise NotImplementedError
            pretrain_params_state_dict = pretrain_model.state_dict()
            pretrain_params_state_dict = {k.replace("model.", ""): v for k, v in pretrain_params_state_dict.items()}
            for name, param in model.named_parameters():
                with torch.no_grad():
                    if name.startswith(("encoder", "shared")):  # only encoder
                        param.copy_(pretrain_model.state_dict()[name])
            del pretrain_model
            logger.info("Main model encoder initialized from pretrain model ")
        self.model = model
        return

    def _build_tree(self, args):
        tree_save_path = args.output_dir + args.query_info + "tree.pkl"
        if os.path.isfile(tree_save_path):
            logger.info("save loaded tree")
            with open(tree_save_path, "rb") as input_file:
                root = pickle.load(input_file)
            self.root = root
        else:
            logger.info("Begin build tree")
            builder = TreeBuilder()
            phase_list = ["train", "dev", "test"] if args.trivia else ["train", "dev"]
            if args.trivia:
                filename_dict = {phase: f"TriviaQA/{phase}.tsv" for phase in phase_list}
            else:
                filename_dict = {
                    "train": "NQ/nq_train_doc_newid.tsv",
                    "dev": "NQ/nq_dev_doc_newid.tsv",
                }
            column_names = ["query", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4", "bert_k30_c30_5"]
            df_dicts = {
                phase: pd.read_csv(
                    os.path.join(args.data_path, filename_dict[phase]),
                    encoding="utf-8",
                    names=column_names,
                    header=None,
                    sep="\t",
                    dtype={"query": str, "queryid": str, "oldid": str, args.id_class: str},
                ).loc[:, ["query", args.id_class]]
                for phase in phase_list
            }
            if args.trivia:
                df_dicts["dev"] = pd.merge(df_dicts["test"], df_dicts["dev"], how="outer")  ## QUESTION : is it proper here?
            df = pd.merge(df_dicts["train"], df_dicts["dev"], how="outer")

            for _, (_, newid) in tqdm(df.iterrows()):
                if args.label_length_cutoff:
                    newid = newid[: args.max_output_length - 2]
                if args.trivia:
                    newid = newid.split(",")
                    for i in range(len(newid)):
                        toks = encode_single_newid(args, newid[i])
                        builder.add(toks)  # add every newid into one tree
                elif args.nq:
                    newid = str(newid)
                    toks = encode_single_newid(args, newid)
                    builder.add(toks)
            if args.tree == 1:
                root = builder.build()
            else:
                logger.info("No Tree")
                root = None
            return root

    def _set_trainer_params(self):
        self.token_weights = torch.tensor([1, 2, 2, 2, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1], device=self.device)
        self.token_weights = self.token_weights / self.token_weights.sum() * self.args.max_output_length
        layer_weights = torch.arange(self.args.num_decoder_layers, device=self.device) + 1
        layer_weights = layer_weights.flip(dims=[0])
        self.layer_weights = layer_weights / layer_weights.sum() * self.args.num_decoder_layers
        logger.info(f"token_weights: {self.token_weights}")
        logger.info(f"layer_weights: {self.layer_weights}")

    def __init__(self, args, train=True):
        super(T5FineTunerSmall, self).__init__()
        self.args = args
        self.root = self._build_tree(args)
        self.save_hyperparameters(args)
        # assert args.tie_word_embedding is not args.semantic_identifier
        if args.semantic_identifier:
            if self.args.position:
                expand_scale = args.max_output_length if not args.hierarchic_decode else 1
                self.decode_vocab_size = args.output_vocab_size * expand_scale + 2
            else:
                self.decode_vocab_size = args.output_vocab_size + 2
        else:
            self.decode_vocab_size = None
        t5_config = T5Config(
            num_layers=args.num_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_ff=args.d_ff,
            d_model=args.d_model,
            num_heads=args.num_heads,
            decoder_start_token_id=0,  # 1,
            output_past=True,
            d_kv=args.d_kv,
            dropout_rate=args.dropout_rate,
            semantic_identifier=args.semantic_identifier,
            hierarchic_decode=args.hierarchic_decode,
            decode_vocab_size=self.decode_vocab_size,
            output_vocab_size=args.output_vocab_size,
            tie_word_embeddings=args.tie_word_embedding,
            tie_decode_embedding=args.tie_decode_embedding,
            contrastive=args.contrastive,
            Rdrop=args.Rdrop,
            Rdrop_only_decoder=args.Rdrop_only_decoder,
            Rdrop_loss=args.Rdrop_loss,
            adaptor_decode=args.adaptor_decode,
            adaptor_efficient=args.adaptor_efficient,
            adaptor_layer_num=args.adaptor_layer_num,
            embedding_distillation=args.embedding_distillation,
            weight_distillation=args.weight_distillation,
            input_dropout=args.input_dropout,
            denoising=args.denoising,
            multiple_decoder=args.multiple_decoder,
            decoder_num=args.decoder_num,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            max_output_length=args.max_output_length,
            decoder_skipping=args.decoder_skipping,
            per_vocab_length=args.kary,
            classifier_type=args.classifier_type,
            use_skip_logits=self.args.use_skip_logits,
            logits_fixer=self.args.logits_fixer,
            CALM_thresholds=self.args.CALM_thresholds,
            fixer_type=self.args.fixer_type,
            fixer_midlayer_num=self.args.fixer_midlayer_num,
            fixer_midlayer_dim=self.args.fixer_midlayer_dim,
            force_cache=self.args.force_cache,
        )

        self.training = train
        self.sanity = False
        self.t5_config = t5_config
        self._init_model()

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")  # QUESTION : It this matter?

        logger.debug(t5_config)
        logger.debug(self.model)

        if self.args.freeze_embeds:
            self.freeze_embeds()
        if self.args.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
        if self.args.decoder_skipping:
            for name, param in self.model.named_parameters():
                if "skip" not in name or (self.args.fix_fixer and "fixer" in name):
                    param.requires_grad = False

        if self.args.softmax:
            self.fc = torch.nn.Linear(args.d_model, self.args.num_cls)  # [feature size, num cls]

        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=0.5)
        if self.args.disc_loss:
            self.dfc = torch.nn.Linear(args.d_model, 1)
        n_observations_per_split = {
            "train": self.args.n_train,
            "validation": self.args.n_val,
            "test": self.args.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        if train:
            n_samples = self.n_obs["train"]
            train_dataset = l1_query(self.args, self.tokenizer, n_samples)
            self.l1_query_train_dataset = train_dataset
            self.t_total = (
                (len(train_dataset) // (self.args.train_batch_size * max(1, self.args.n_gpu)))
                // self.args.gradient_accumulation_steps
                * float(self.args.num_train_epochs)
            )
        self.last_batch = None
        self.step_no = 0
        # self.switcher = embedding_switch(self.args.output_vocab_size, self.decode_vocab_size, self.args.max_output_length)
        self.first_loss_dict = {}
        self._set_trainer_params()
        self.kl_loss = adaptor_loss_KL(self.model.logit_mask, self.token_weights, self.layer_weights)
        self.cel_loss = adaptor_loss_CEL(self.model.logit_mask, self.token_weights, self.layer_weights)

    def unfreeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = True

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def forward(
        self,
        input_ids,
        aug_input_ids=None,
        aug_attention_mask=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        only_encoder=False,
    ):
        if self.args.Rdrop > 0 and not self.args.Rdrop_only_decoder and self.training:
            if aug_input_ids is not None and self.training:
                input_ids = torch.cat([input_ids, aug_input_ids.clone()], dim=0)
                attention_mask = torch.cat([attention_mask, aug_attention_mask], dim=0)
            elif self.training:
                input_ids = torch.cat([input_ids, input_ids.clone()], dim=0)
                attention_mask = torch.cat([attention_mask, attention_mask.clone()], dim=0)
            if self.args.input_dropout and np.random.rand() < 0.5:
                input_mask = torch.rand(input_ids.shape, device=input_ids.device) < 0.9
                input_ids = torch.where(input_mask, input_ids, torch.zeros_like(input_ids))
            if decoder_attention_mask is not None:
                decoder_attention_mask = torch.cat([decoder_attention_mask, decoder_attention_mask], dim=0)
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)
            if decoder_input_ids is not None:
                decoder_input_ids = torch.cat([decoder_input_ids, decoder_input_ids], dim=0)

        out = self.model(
            input_ids,
            encoder_outputs=encoder_outputs,
            only_encoder=only_encoder,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        return out

    def _step(self, batch):
        self.step_no += 1
        return self._step_i(batch, -1)

    def _step_i(self, batch, i, encoder_outputs=None, input_mask=None):
        # last batch check
        batch["target_ids"][batch["target_ids"][:, :] == self.tokenizer.pad_token_id] = -100
        shared_params = {
            "input_ids": batch["source_ids"],
            "attention_mask": batch["source_mask"],
            "aug_input_ids": batch["aug_source_ids"],
            "aug_attention_mask": batch["aug_source_mask"],
            "decoder_attention_mask": batch["target_mask"],
            "labels": batch["target_ids"],
        }
        outputs = self.forward(**shared_params)
        loss = outputs.loss
        orig_loss = getattr(outputs, "orig_loss", 0.0)
        dist_loss = getattr(outputs, "dist_loss", 0.0)

        output_dict = {
            "loss": loss,
            "orig_loss": orig_loss,
        }
        if self.args.Rdrop > 0:
            output_dict.update({"dist_loss": dist_loss})

        if self.args.decoder_skipping:
            skip_outputs = outputs.skip_outputs
            # for keys in ["skip_score", "skip_decision", "layer_prediction"]:
            #     skip_outputs[keys] = skip_outputs[keys].transpose(1, 2)
            source_id, label, target_mask = batch["source_ids"], batch["target_ids"], batch["target_mask"]
            f_logits, f_logits_nomask = outputs.logits, outputs.logits_before_mask

            loss_gt = 0.0
            layer_num = len(self.model.decoder.block)
            device = f_logits.device
            layer_logits, skip_score, skip_decision, layer_prediction, position = (
                skip_outputs["logits"],  # [batch, seq, layer, vocab]
                skip_outputs["skip_score"],  # [batch, seq, layer]
                skip_outputs["skip_decision"].float(),  # [batch, seq, layer]
                skip_outputs["layer_prediction"],
                skip_outputs["position"],
            )
            batch_size, seq_length, counted_layer_num, vocab_size = layer_logits.shape
            self.counted_layer_num = counted_layer_num  # sometimes the counted layer num is not the same as the layer num
            oracle_skip, correct_predict, anlysis_dict = skip_position_analysis(layer_prediction, position, target_mask, f_logits)
            # oracle_mean = anlysis_dict["oracle_mean"]
            del anlysis_dict["oracle_mean"]
            output_dict.update(anlysis_dict)
            t_valid_mask_e = repeat(target_mask, "B T -> B T L", L=counted_layer_num)
            skip_score[~t_valid_mask_e] = -100
            correct_predict[~t_valid_mask_e] = -100
            skip_decision[~t_valid_mask_e] = -100
            # layer_logits[~t_valid_mask_e] = -float("inf")

            # jsd_unmasked = jsd_per_layer(layer_logits, f_logits_nomask)
            if self.args.log_hard_sample:  # TODO: check order
                for i, query in enumerate(oracle_skip):
                    if query.max() > 6:
                        layer_logit = layer_logits[i, query.max(dim=0)[1], :, ...]
                        final_logit = f_logits_nomask[i, query.max(dim=0)[1], ...]
                        kld = F.kl_div(layer_logit.log_softmax(dim=-1), final_logit.softmax(dim=-1), reduction="none").sum(dim=-1)
                        self.hard_sample.append(
                            (self.tokenizer.decode(source_id[i]), label[i], query, layer_prediction[i], skip_decision[i], skip_score[i], kld)
                        )

            if self.args.logits_fixer:
                fixer_loss = self.kl_loss(
                    f_logits_nomask,
                    layer_logits,
                    target_mask,
                    token_valid=True,
                )
                fixer_loss_cel = self.cel_loss(label, layer_logits)
                loss_gt = fixer_loss * (1 - self.args.cel_beta) + fixer_loss_cel * self.args.cel_beta
                output_dict.update({"fixer_loss": fixer_loss, "fixer_loss_cel": fixer_loss_cel})

            jsd_masked = jsd_per_layer(layer_logits, f_logits, valid_indices=self.model.valid_indices)
            items_list = [skip_score, skip_decision, layer_prediction, correct_predict, jsd_masked]
            self.buffer.add([items_list.detach().cpu() for items_list in items_list])

            loss_g = 0.0

            if self.args.gen_training:
                beam_size = self.args.gen_beam_size
                threshold = self.get_threshold()
                self.set_threshold(torch.ones_like(threshold) * 10)
                outputs_generate = self.model.generate(
                    batch["source_ids"],
                    attention_mask=batch["source_mask"],
                    use_cache=self.args.use_cache,
                    decoder_attention_mask=batch["target_mask"],
                    max_length=self.args.max_output_length,
                    num_beams=beam_size,
                    length_penalty=self.args.length_penalty,
                    num_return_sequences=beam_size,
                    early_stopping=False,
                    semantic_identifier=self.args.semantic_identifier,
                    decode_vocab_size=self.decode_vocab_size,
                    decode_tree=self.root,
                    output_scores=True,
                    generate_kwargs={"exit_analysis": True, "training_mode": False, "return_all": True},
                )
                
                ## save hidden states ##
                # all_hs = [i.skip_outputs["all_hs"] for i in outputs_generate[-3]]
                # all_hs = torch.cat(all_hs, dim=2).type(torch.float16)
                # torch.save(all_hs, f"src/all_hs/{self.step_no}.pt")
                # hs_gq = rearrange(list(outputs.decoder_hidden_states), "L B T E-> (B T) L E").type(torch.float16)
                # torch.save(hs_gq[batch["target_mask"].view(-1).bool()], f"src/hs_gq/{self.step_no}.pt")

                logits_g = torch.cat([i.skip_outputs["logits"] for i in outputs_generate[-3]], dim=1)
                final_logits_g_unmasked = torch.cat([i.final_lm_logits for i in outputs_generate[-3]], dim=1)
                target_mask_g = torch.ones(logits_g.shape[:2]).to(self.device).bool()

                fixer_loss_g = self.kl_loss(
                    final_logits_g_unmasked,
                    logits_g,
                    target_mask_g,
                    token_valid=True,
                    confidence_weighting=self.args.conf_weighting,
                )
                output_dict.update({"fixer_loss_g": fixer_loss_g})  # , "fixer_loss_g_2": fixer_loss_2})
                loss_g += fixer_loss_g
                self.set_threshold(threshold)
            loss = loss_g * self.args.gen_alpha + loss_gt * (1 - self.args.gen_alpha)
            output_dict.update({"loss": loss, "loss_g": loss_g, "loss_exit": loss_gt})
        return output_dict

    def on_train_epoch_start(self):
        self.finished_sanity_check = True
        if self.args.decoder_skipping:
            self.buffer = training_buffers(["skip_score", "skip_decision", "layer_prediction", "correct_predict", "jsd"])

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        show_metric = "loss" if self.args.decoder_skipping else "orig_loss"
        for k in outputs.keys():
            self.log(f"train/{k}", outputs[k], on_step=True, prog_bar=(k == show_metric), sync_dist=True)
        return outputs

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self.args.decoder_skipping:  # naive method of calculating skip threshold
            if getattr(self, "finished_sanity_check", None) is None:  # fuck sanity check
                return

            count_layer_num = self.counted_layer_num
            if getattr(self, "hard_sample", None) is not None and self.hard_sample != []:
                logger.info(f"hard_sample:{len(self.hard_sample)}")
                self.log("hard_sample_times", len(self.hard_sample), on_epoch=True)
                query, target_id, predict, layer_prediction, decision, skip_score, kld = [zipped for zipped in zip(*self.hard_sample)]
                save_dict = {
                    "query": query,
                    "target_id": target_id,
                    "predict": predict,
                    "layer_perdiction": layer_prediction,
                    "decision": decision,
                    "skip_score": skip_score,
                    "kld": kld,
                    # "final_logit": final_logit,
                }
                torch.save(save_dict, f"hard_example{self.current_epoch}.pt")
            [skip_score, skip_decision, layer_prediction, correct_predict, jsd] = self.buffer.get_all()
            seq_threshold = torch.stack([i.skip_params_sequence for i in self.model.decoder.block])
            jsd_currect = jsd[correct_predict == 1]
            jsd_threshold = jsd_currect.std() + jsd_currect.mean()
            correct_predict[jsd < jsd_threshold] = 1
        return

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = []
        if self.args.decoder_skipping:
            self.threshold_buffer = self.get_threshold()
            self.set_threshold(torch.ones_like(self.threshold_buffer))
            if self.args.exit_analysis:
                self.skip_outputs_list = []

    def validation_step(self, batch, batch_idx):
        result = self.infer_step(batch, exit_analysis=self.args.exit_analysis)
        self.val_output_list.append(result)

        if self.args.exit_analysis:
            self.skip_outputs_list.append(result["skip_outputs"])
        return result

    def infer_step(self, batch, exit_analysis: bool = False):
        inf_result_cache = []
        outputs = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=self.args.use_cache,
            decoder_attention_mask=batch["target_mask"],
            max_length=self.args.max_output_length,
            num_beams=self.args.num_return_sequences,
            length_penalty=self.args.length_penalty,
            num_return_sequences=self.args.num_return_sequences,
            early_stopping=False,
            semantic_identifier=self.args.semantic_identifier,
            decode_vocab_size=self.decode_vocab_size,
            decode_tree=self.root,
            output_scores=True,
            generate_kwargs={
                "exit_analysis": exit_analysis,
                "training_mode": False,
                "pruning": self.args.beam_pruning,
                "nq": self.args.nq,
            },
        )
        outs, scores = outputs[:2]
        time_encoder, time_bs = outputs[-2:]
        if self.args.semantic_identifier:
            dec = decode_token(self.args, outs.cpu().numpy())
            dec = dec_2d(dec, self.args.num_return_sequences)
        else:
            dec = [self.tokenizer.decode(ids) for ids in outs]

        texts = [self.tokenizer.decode(ids) for ids in batch["source_ids"]]
        for r in batch["rank"]:
            if self.args.label_length_cutoff:
                gt = [s[: self.args.max_output_length - 2] for s in list(r[0])]
            else:
                gt = list(r[0])
            ra = r[1]
            ra = [str(a.item()) for a in ra]
            for pred, g, text, ran in zip(dec, gt, texts, ra):
                pred = ",".join(pred)
                inf_result_cache.append([text, pred, g, int(ran)])
        # pred = '14-18,1-10-21,25-17-21,25-19-1-1,25-17-1-1,25-19-21-1,25-19-6-1,25-17-1-5,25-19-1-5,25-17-3'
        ret = {
            "inf_result_batch": inf_result_cache,
            "inf_result_batch_prob": scores,
            "time_encoder": time_encoder,  # TODO: time to time_it
            "time_bs": time_bs,
        }

        if self.args.decoder_skipping:
            totensor = lambda x: [torch.tensor(i) for i in x]
            skip_position, _ = totensor(outputs[-4:-2])
            avg_skip_position = skip_position[skip_position != -1]
            ret.update({"avg_skip_position": avg_skip_position})
            if exit_analysis:
                skip_outputs = outputs[2]
                ret.update({"skip_outputs": {k: v.detach() for k, v in skip_outputs.items()}})
        return ret

    def on_validation_epoch_end(self):
        outputs = self.val_output_list
        results = self.process_valid_outputs(outputs)
        self.val_output_list.clear()
        torch.cuda.empty_cache()
        if hasattr(self, "finished_sanity_check"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.log("recall1", results["recall"][5], on_epoch=True, on_step=False)

        if self.args.decoder_skipping:
            sequence_params = torch.stack([i.skip_params_sequence for i in self.model.decoder.block])
            # print(f"sequence_params:{sequence_params}")
            if self.args.exit_analysis:
                self.exit_analysis_eval(self.skip_outputs_list)
                del self.skip_outputs_list

            if hasattr(self, "finished_sanity_check"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.log("val/skip_position", results["skip_dict"]["total_mean"], on_epoch=True, on_step=False)
            self.set_threshold(self.threshold_buffer)

    def on_test_start(self):
        if self.args.flops_profiling:
            from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

            self.prof = FlopsProfiler(self.model)
            self.flop_list = []

        self.time_test_start = time.time()

        super().on_test_start()
        self.test_output_list = []
        self.test_skip_position_list = []
        self.skip_outputs_list = []
        self.time_encoder = 0
        self.time_bs = 0
        return

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if batch_idx > 10 and self.args.flops_profiling:
            self.prof.start_profile()

        result = self.infer_step(batch, exit_analysis=self.args.exit_analysis)

        if batch_idx > 10 and self.args.flops_profiling:
            flops = self.prof.get_total_flops(as_string=False)
            # params = self.prof.get_total_params(as_string=False)
            # self.prof.print_model_profile(profile_step=batch_idx)
            self.prof.end_profile()
            self.flop_list.append(flops)

        self.test_output_list.append(result["inf_result_batch"])
        self.time_encoder += result["time_encoder"]
        self.time_bs += result["time_bs"]
        if self.args.decoder_skipping:
            self.test_skip_position_list.append(result["avg_skip_position"])
            if self.args.exit_analysis:
                self.skip_outputs_list.append({k: v.detach().cpu() for k, v in result["skip_outputs"].items()})

    @torch.no_grad()
    def exit_analysis_eval(self, skip_outputs_list, log=True):
        skip_outputs = skip_outputs_list
        skip_outputs = {
            k: torch.nested.nested_tensor([skip_outputs[i][k].squeeze().cpu() for i in range(len(skip_outputs))]) for k in skip_outputs[0].keys()
        }
        skip_outputs = {k: torch.nested.to_padded_tensor(v, -1.0) for k, v in skip_outputs.items()}
        skip_outputs = {k: v.view(-1, *v.shape[2:]) for k, v in skip_outputs.items()}
        layer_prediction, position, final_logits = skip_outputs["layer_prediction"], skip_outputs["position"], skip_outputs["final_lm_logits"]
        layer_logits = skip_outputs["logits"]
        target_mask = skip_outputs["position"] != -1
        batch_size, seq_length, counted_layer_num, vocab_size = layer_logits.shape
        oracle_skip, correct_predict, anlysis_dict = skip_position_analysis(layer_prediction, position, target_mask, final_logits)
        e_valid_slices = self.model.valid_indices.sort()[0]
        flat_target_mask = target_mask.view(-1).bool()
        decisions = skip_outputs["skip_decision"]
        scores = skip_outputs["skip_score"]
        classifier_loss = calc_FCL_loss(scores, correct_predict, target_mask)

        def count_redundant(position):
            if min(position) == -1:
                return torch.tensor(0).to(position.device)
            count = torch.bincount(position, minlength=13)
            count[count < len(position) / 100] = 0  # TODO: better threshold
            redundant = torch.nonzero(count).min()
            return redundant

        redundant = torch.stack([count_redundant(position[:, i]) for i in range(position.shape[1])])
        logger.info(f"redundant:{redundant}")
        logger.info("exit position:{}".format(anlysis_dict["skip_position"]))
        logger.info("oracle position:{}".format(anlysis_dict["oracle_mean"]))
        logger.info("oracle position mean:{}".format(anlysis_dict["oracle_skip_position"]))
        logger.info("---")
        nonzero_sentenes = position[:, 0] != -1
        if log:
            self.log(f"val/oracle_skip_postion", anlysis_dict["oracle_skip_position"], on_epoch=True, sync_dist=True)
            self.log(f"val/classifier_loss", classifier_loss, on_epoch=True, sync_dist=True)

    def on_test_end(self):
        super().on_test_end()
        if self.args.flops_profiling:
            flops = torch.tensor(self.flop_list).float().mean() // self.args.eval_batch_size
            logger.info(f"Average FLOPs: {flops}")
        self.time_test_end = time.time()
        time_test = self.time_test_end - self.time_test_start

        outputs = self.test_output_list
        inf_result_cache = [item for sublist in outputs for item in sublist]
        dataset_size = len(inf_result_cache)
        result = pd.DataFrame(inf_result_cache, columns=["query", "pred", "gt", "rank"])
        result.sort_values(by=["query", "rank"], ascending=True, inplace=True)
        result.to_csv(self.args.res1_save_path, mode="w", sep="\t", header=None, index=False)
        res1 = result.loc[result["rank"] == 1]
        res1 = res1.values.tolist()
        recall_value = recall(self.args, res1)
        mrr_value = MRR100(self.args, res1)

        if self.args.exit_analysis:
            self.exit_analysis_eval(self.skip_outputs_list, log=False)

        logger.info(f"Inference time:{time_test}, encoder_time:{self.time_encoder}, bs_time:{self.time_bs}")
        logger.info(f"Inference time cost per sample:{time_test / dataset_size},Thorughput:{dataset_size / time_test}")
        logger.info(f"Avg_time_encoder: {self.time_encoder/dataset_size}, Avg_time_bs: {self.time_bs/dataset_size}")
        logger.info(f"encoder_time_ratio: {self.time_encoder/time_test*100:.2f}%, bs_time_ratio: {self.time_bs/time_test*100:.2f}%")
        if self.args.decoder_skipping:
            skip_position = [
                torch.cat([sublist, torch.ones(self.args.max_output_length - len(sublist)) * -1]) for sublist in self.test_skip_position_list
            ]
            skip_dict = calc_skip_position(torch.stack(skip_position))
            logger.info(skip_dict)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.decoder_learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))
                ],
                "weight_decay": 0.0,
                "lr": self.args.decoder_learning_rate,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.t_total)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def train_dataloader(self):
        logger.info("load training data and create training loader.")
        n_samples = self.n_obs["train"]
        if hasattr(self, "l1_query_train_dataset"):
            train_dataset = self.l1_query_train_dataset
        else:
            train_dataset = l1_query(self.args, self.tokenizer, n_samples)
        self.prefix_embedding, self.prefix2idx_dict, self.prefix_mask = (
            train_dataset.prefix_embedding,
            train_dataset.prefix2idx_dict,
            train_dataset.prefix_mask,
        )
        sampler = DistributedSampler(train_dataset)  # QUESTION: May caused ddp error?
        dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.args.train_batch_size,
            drop_last=True,
            shuffle=False,  # QUESTION: ENMMMM
            num_workers=self.args.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        logger.info("load validation data and create validation loader.")
        n_samples = self.n_obs["validation"]
        val_dataset = l1_query(self.args, self.tokenizer, n_samples, task="dev")
        sampler = DistributedSampler(val_dataset)
        dataloader = DataLoader(
            val_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.args.num_workers,
        )
        return dataloader

    def process_valid_outputs(self, outputs, log=True):
        if self.args.decoder_skipping:
            skip_position_list = [sublist["avg_skip_position"] for sublist in outputs]
            skip_position = [torch.cat([sublist, -torch.ones(self.args.max_output_length - len(sublist))]) for sublist in skip_position_list]
            skip_position = torch.stack(skip_position)
            skip_dict = calc_skip_position(skip_position)
            if log:
                logger.info(skip_dict)

        inf_result_cache = [item for sublist in outputs for item in sublist["inf_result_batch"]]
        inf_result_cache_prob = [
            softmax(
                sublist["inf_result_batch_prob"][
                    i
                    * int(len(sublist["inf_result_batch_prob"]) / len(outputs[0]["inf_result_batch"])) : (i + 1)
                    * int(len(sublist["inf_result_batch_prob"]) / len(outputs[0]["inf_result_batch"]))
                ]
            )
            for sublist in outputs
            for i in range(len(sublist["inf_result_batch"]))
        ]

        res = pd.DataFrame(inf_result_cache, columns=["query", "pred", "gt", "rank"])
        res.sort_values(by=["query", "rank"], ascending=True, inplace=True)
        res1 = res.loc[res["rank"] == 1]
        res1 = res1.values.tolist()
        recall_dict = recall(self.args, result=res1, recall_num=[1, 5, 10], log=log)
        mrr = MRR100(self.args, result=res1, log=log)
        return {
            "recall": recall_dict,
            "MRR100": mrr,
            "inf_result_cache": inf_result_cache,
            "inf_result_cache_prob": inf_result_cache_prob,
            "skip_dict": skip_dict if self.args.decoder_skipping else None,
        }

    ### LTT part ###
    def get_threshold(self):
        return torch.stack([i.skip_params_sequence for i in self.model.decoder.block]) if self.args.decoder_skipping else None

    def set_threshold(self, threshold):
        if self.args.decoder_skipping:
            device = f"cuda:{self.args.gpu_no[0]}"
            for i, block in enumerate(self.model.decoder.block):
                block.skip_params_sequence = torch.nn.Parameter(threshold[i].to(device))

    def one_validation_test(self, dataloader, threshold=None, batch_num=-1):
        device = f"cuda:{self.args.gpu_no[0]}"
        if threshold != None:  # backup old threshold
            old_threshold = torch.stack([i.skip_params_sequence for i in self.model.decoder.block])
            self.set_threshold(threshold)
        test_output_list = []
        pbar = tqdm(dataloader, ncols=0, total=batch_num if batch_num != -1 else None)
        for i, batch in enumerate(pbar):
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            test_output_list.append(self.infer_step(batch, exit_analysis=True))
            if i == batch_num:
                break
        results = self.process_valid_outputs(test_output_list, log=False)
        skip_output_list = [sublist["skip_outputs"] for sublist in test_output_list]
        self.exit_analysis_eval(skip_output_list, log=False)
        if threshold != None:
            self.set_threshold(old_threshold)
        return results

    def threshold_compare(self, dataloader, test_threshold, target_threshold, batch_num=-1):
        device = f"cuda:{self.args.gpu_no[0]}"
        old_threshold = torch.stack([i.skip_params_sequence for i in self.model.decoder.block])
        test_output_list = []
        target_output_list = []

        pbar = tqdm(dataloader, ncols=0, total=batch_num if batch_num != -1 else None)
        for i, batch in enumerate(pbar):
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            self.set_threshold(test_threshold)
            test_output_list.append(self.infer_step(batch, exit_analysis=self.args.exit_analysis))
            self.set_threshold(target_threshold)
            target_output_list.append(self.infer_step(batch, exit_analysis=self.args.exit_analysis))
            if i == batch_num:
                break

        results = self.process_valid_outputs(test_output_list, log=False)
        # skip_output_list = [sublist["skip_outputs"] for sublist in test_output_list]

        results_target = self.process_valid_outputs(target_output_list, log=False)
        # target_output_list = [sublist["skip_outputs"] for sublist in target_output_list]

        self.set_threshold(old_threshold)
        return results, results_target
