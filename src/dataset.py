import numpy as np
import torch
from torch.utils.data import Dataset
from main_utils import load_data_infer, load_data
from loguru import logger


class l1_query(Dataset):
    def __init__(self, args, tokenizer, num_samples, print_text=False, task="train"):
        assert task in ["train", "dev", "test"]
        self.args = args
        input_length = args.max_input_length
        output_length = args.max_output_length * int(np.log10(args.output_vocab_size))
        inf_input_length = args.inf_max_input_length
        random_gen = args.random_gen
        softmax = args.softmax

        if task == "train":
            self.dataset, self.q_emb, self.query_dict, self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict = load_data(args)
        elif task == "test" or task == "dev":
            self.dataset = load_data_infer(args, task)
            self.q_emb, self.query_dict, self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict = None, None, None, None, None
        else:
            raise NotImplementedError("No Corresponding Task.")

        if num_samples:
            self.dataset = self.dataset[:num_samples]

        self.task = task
        self.input_length = input_length
        self.doc_length = self.args.doc_length
        self.inf_input_length = inf_input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
        self.softmax = softmax
        self.random_gen = random_gen
        if random_gen:
            assert len(self.dataset[0]) >= 3
        self.random_min = 2
        self.random_max = 6
        self.vocabs = set(self.tokenizer.get_vocab().keys())
        for token in [
            self.tokenizer.eos_token,
            self.tokenizer.unk_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
            self.tokenizer.cls_token,
            self.tokenizer.mask_token,
        ] + tokenizer.additional_special_tokens:
            if token is not None:
                self.vocabs.remove(token)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def clean_text(text):
        text = text.replace("\n", "")
        text = text.replace("``", "")
        text = text.replace('"', "")

        return text

    def convert_to_features(self, example_batch, length_constraint):
        # Tokenize contexts and questions (as pairs of inputs)

        input_ = self.clean_text(example_batch)
        output_ = self.tokenizer.batch_encode_plus([input_], max_length=length_constraint, padding="max_length", truncation=True, return_tensors="pt")

        return output_

    def __getitem__(self, index):
        inputs = self.dataset[index]
        query_embedding = torch.tensor([0])
        prefix_embedding, prefix_mask = torch.tensor([0]), torch.tensor([0])
        query, targets_text, rank = inputs[:3]

        if len(inputs) >= 6:
            query, targets_text, rank, neg_target, aug_query = inputs[0], inputs[1], inputs[2], inputs[4], inputs[5]
        elif len(inputs) >= 5:
            query, targets_text, rank, neg_target = inputs[0], inputs[1], inputs[2], inputs[4]
        else:
            query, targets_text, rank = inputs[0], inputs[1], inputs[2]

        if hasattr(self, "query_dict") and self.query_dict is not None:
            query_embedding = self.q_emb[self.query_dict[query]]
        neg_targets_list = []
        if self.args.hard_negative:
            neg_targets_list = np.random.choice(neg_target, self.args.sample_neg_num)
        if self.args.aug_query and len(aug_query) >= 1:
            aug_query = np.random.choice(aug_query, 1)[0]
        else:
            aug_query = ""
        if self.args.label_length_cutoff:
            targets_text = targets_text[: self.args.max_output_length - 2]

        source = self.convert_to_features(query, self.input_length if self.task == "train" else self.inf_input_length)
        source_ids = source["input_ids"].squeeze()
        # if "print_token" in self.args.query_type:
        #     logger.info("Input Text: ", query, "\n", "Output Text: ", source_ids)
        src_mask = source["attention_mask"].squeeze()
        aug_source = self.convert_to_features(aug_query, self.input_length if self.task == "train" else self.inf_input_length)
        aug_source_ids = aug_source["input_ids"].squeeze()
        aug_source_mask = aug_source["attention_mask"].squeeze()
        # if self.args.multiple_decoder:
        #     target_ids, target_mask = [], []
        #     for i in range(self.args.decoder_num):
        #         targets = self.convert_to_features(target[i], self.output_length)
        #         target_ids.append(targets["input_ids"].squeeze())
        #         target_mask.append(targets["attention_mask"].squeeze())
        # else:
        targets = self.convert_to_features(targets_text, self.output_length)
        target_ids = targets["input_ids"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        def target_to_prefix_emb(target, tgt_length):
            tgt_prefix_emb = []
            prefix_masks = []
            for i in range(tgt_length):
                if i < len(target):
                    ###### fake data
                    _prefix_emb = np.random.rand(10, 768)
                    ###### real data
                    # _prefix_emb = self.prefix_embedding[self.prefix2idx_dict[target[:i]]]
                    _prefix_emb = torch.tensor(_prefix_emb)
                    tgt_prefix_emb.append(_prefix_emb.unsqueeze(0))
                    ##############################
                    ###### fake data
                    _prefix_mask = np.random.rand(
                        10,
                    )
                    _prefix_mask[_prefix_mask < 0.5] = 0
                    _prefix_mask[_prefix_mask > 0.5] = 1
                    ###### real data
                    # _prefix_mask = self.prefix_mask[self.prefix2idx_dict[target[:i]]]
                    _prefix_mask = torch.LongTensor(_prefix_mask)
                    prefix_masks.append(_prefix_mask.unsqueeze(0))
                    ##############################
                else:
                    tgt_prefix_emb.append(torch.zeros((1, 10, 768)))
                    prefix_masks.append(torch.zeros((1, 10)))
            return torch.cat(tgt_prefix_emb, dim=0), torch.cat(prefix_masks, dim=0)

        if self.prefix_embedding is not None:
            prefix_embedding, prefix_mask = target_to_prefix_emb(targets_text, self.output_length)

        neg_target_ids_list = []
        neg_target_mask_list = []
        neg_rank_list = []

        if self.args.hard_negative:
            for cur_target in neg_targets_list:
                cur_targets = self.convert_to_features(cur_target, self.output_length)
                cur_target_ids = cur_targets["input_ids"].squeeze()
                cur_target_mask = cur_targets["attention_mask"].squeeze()
                neg_target_ids_list.append(cur_target_ids)
                neg_target_mask_list.append(cur_target_mask)
                neg_rank_list.append(999)  # denote hard nagative

        lm_labels = torch.zeros(self.args.max_output_length, dtype=torch.long)

        if self.args.semantic_identifier:
            ## func target_id+target_id2, twice or k
            def decode_embedding_process(target_texts):
                if self.args.kary:
                    idx = 0
                    target_texts = target_texts.split("-")
                    if self.args.position and not self.args.hierarchic_decode:
                        target_id_int = [i * self.args.output_vocab_size + int(c) + 2 for i, c in enumerate(target_texts)]
                    else:
                        target_id_int = [int(c) + 2 for c in target_texts]
                else:  # not used
                    bits = int(np.log10(self.args.output_vocab_size))
                    idx = 0
                    for i in range(0, len(target_texts), bits):
                        if i + bits >= len(target_texts):
                            c = target_texts[i:]
                        c = target_texts[i : i + bits]
                        if self.args.position:
                            temp = idx * self.args.output_vocab_size + int(c) + 2 if not self.args.hierarchic_decode else int(c) + 2
                        else:
                            temp = int(c) + 2
                        target_id_int.append(temp)
                        idx += 1

                lm_labels[: len(target_id_int)] = torch.LongTensor(target_id_int)
                lm_labels[len(target_id_int)] = 1
                decoder_attention_mask = lm_labels.clone()
                decoder_attention_mask[decoder_attention_mask != 0] = 1
                target_ids = lm_labels
                target_mask = decoder_attention_mask
                return target_ids, target_mask

            target_ids, target_mask = decode_embedding_process(targets_text)

            if self.args.hard_negative:
                for i in range(len(neg_target_ids_list)):
                    cur_target_ids = neg_target_ids_list[i]
                    cur_target_ids, cur_target_mask = decode_embedding_process(cur_target_ids)
                    neg_target_ids_list[i] = cur_target_ids
                    neg_target_mask_list[i] = cur_target_mask

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "aug_source_ids": aug_source_ids,
            "aug_source_mask": aug_source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "neg_target_ids": neg_target_ids_list,
            "neg_rank": neg_rank_list,
            "neg_target_mask": neg_target_mask_list,
            "doc_ids": doc_ids if self.args.contrastive_variant != "" else torch.tensor([-1997], dtype=torch.int64),
            "doc_mask": doc_mask if self.args.contrastive_variant != "" else torch.tensor([-1997], dtype=torch.int64),
            "softmax_index": torch.tensor([inputs[-1]], dtype=torch.int64) if self.softmax else torch.tensor([-1997], dtype=torch.int64),
            "rank": rank,
            "query_emb": query_embedding,
            "prefix_emb": prefix_embedding,
            "prefix_mask": prefix_mask,
        }
