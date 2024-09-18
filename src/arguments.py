import argparse
import os
import yaml
from loguru import logger


def parsers_parser():
    parser = argparse.ArgumentParser()

    # general params
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--exp_postfix", type=str)
    parser.add_argument("--project_suffix", type=str)
    parser.add_argument("--tags", type=str, nargs="*")
    parser.add_argument("--mode", type=str, choices=["train", "train_gen", "eval", "calculate", "LTT"])
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--train_pct", type=float)
    parser.add_argument("--gpu_no", type=int, nargs="*")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--model_name_or_path", type=str, default="t5-")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="t5-")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_set", type=str, default="dev")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--model_info", type=str, default="base")
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--log", type=int)
    parser.add_argument("--wandb", type=int)
    parser.add_argument("--profiler", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=-1)  # will be filled by os.cpu_count()
    parser.add_argument("--use_cache", type=int)

    # skip params
    parser.add_argument("--decoder_skipping", type=int)
    parser.add_argument("--conf_weighting", type=int)
    parser.add_argument("--beam_pruning", type=float)
    parser.add_argument("--CALM_thresholds", type=int, default=0)
    parser.add_argument("--use_skip_logits", type=int, default=1)
    parser.add_argument("--flops_profiling", type=int)
    parser.add_argument("--logits_fixer", type=int)
    parser.add_argument("--fix_fixer", type=int)
    parser.add_argument("--exit_analysis", type=int)
    parser.add_argument("--log_hard_sample", type=int, default=0)
    parser.add_argument("--gen_training", type=int, default=0)
    parser.add_argument("--cel_beta", type=float)
    parser.add_argument("--gen_alpha", type=float)
    parser.add_argument("--gen_beam_size", type=int, default=10)
    parser.add_argument("--fixer_type", type=str, default="0")
    parser.add_argument("--fixer_midlayer_num", type=int, default=4)
    parser.add_argument("--fixer_midlayer_dim", type=int, default=1024)
    parser.add_argument(
        "--classifier_type",
        type=str,
        choices=[
            "classifier",
            "last_layer_softmax",
            "fixer_softmax",
            "fixer_softmax_HW",
            "fixer_softmax_single",
            "fixer_softmax_HW_multiple",
            "hs_cos",
        ],
    )
    parser.add_argument("--valid_length", type=int, default=5)
    parser.add_argument("--prophet_type", type=str, default="")
    parser.add_argument("--force_cache", type=int)
    """
    Classifier:             hidden state -> classifier -> decision
    Classifier_B:           hidden state -> oritinal_lm_head -> classifier -> decision
    last_layer_softmax :    hidden state -> oritinal_lm_head -> softmax -> top(2-1) -> decision
    multiple_classifier :   hidden state -> layer_wise_lm_head -> classifier -> decision
    trained_layer_softmax : hidden state -> layer_wise_lm_head -> softmax -> top(2-1) -> decision
    """
    parser.add_argument("--threshold_path", type=str)

    # LTT params
    parser.add_argument("--full_eval", type=int)

    # training params
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--decoder_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument("--early_stop_callback", type=int, default=0, choices=[0, 1])
    parser.add_argument("--fp_16", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--softmax", type=int, default=0, choices=[0, 1])

    # model params
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_cls", type=int, default=1000)
    parser.add_argument("--semantic_identifier", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--output_vocab_size", type=int, default=30)
    parser.add_argument("--hierarchic_decode", type=int, default=0, choices=[0, 1])
    parser.add_argument("--tie_word_embedding", type=int, default=0, choices=[0, 1])
    parser.add_argument("--tie_decode_embedding", type=int, default=1, choices=[0, 1])
    parser.add_argument("--length_penalty", type=int, default=0.8)  # interesting
    parser.add_argument("--freeze_encoder", type=int, default=0, choices=[0, 1])
    parser.add_argument("--freeze_embeds", type=int, default=0, choices=[0, 1])

    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--continue_ckpt", type=str, default=None)
    parser.add_argument("--ckpt_monitor", type=str, default="recall")
    parser.add_argument("--monitor_mode", type=str, default="min", choices=["min", "max"])

    parser.add_argument("--pretrain_encoder", type=int, default=1, choices=[0, 1])
    parser.add_argument("--recall_num", type=list, default=[1, 5, 10, 20, 50, 100], help="[1,5,10,20,50,100]")
    parser.add_argument("--random_gen", type=int, default=0, choices=[0, 1])
    parser.add_argument("--label_length_cutoff", type=int, default=0)

    parser.add_argument("--max_input_length", type=int, default=40)
    parser.add_argument("--inf_max_input_length", type=int, default=40)
    parser.add_argument("--max_output_length", type=int, default=10)
    parser.add_argument("--doc_length", type=int, default=64)
    parser.add_argument("--contrastive_variant", type=str, default="", help="E_CL, ED_CL, doc_Reweight")
    parser.add_argument("--num_return_sequences", type=int, help="generated id num (include invalid)")

    # NCI params
    parser.add_argument("--Rdrop", type=float, default=0.0, help="default to 0-0.3")
    parser.add_argument("--input_dropout", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument(
        "--Rdrop_only_decoder",
        type=int,
        default=0,
        help="1-RDrop only for decoder, 0-RDrop for all model",
        choices=[0, 1],
    )
    parser.add_argument("--Rdrop_loss", type=str, default="Contrast", choices=["KL", "L2", "Contrast"])

    parser.add_argument("--adaptor_decode", type=int, default=0, help="default to 0,1")
    parser.add_argument("--adaptor_efficient", type=int, default=0, help="default to 0,1")
    parser.add_argument("--adaptor_layer_num", type=int, default=0)

    parser.add_argument("--contrastive", type=int, default=0)
    parser.add_argument("--embedding_distillation", type=float, default=0.0)
    parser.add_argument("--weight_distillation", type=float, default=0.0)
    parser.add_argument("--hard_negative", type=int, default=0)
    parser.add_argument("--aug_query", type=int, default=0)
    parser.add_argument("--aug_query_type", type=str, default="corrupted_query", help="aug_query, corrupted_query")
    parser.add_argument("--sample_neg_num", type=int, default=0)
    parser.add_argument("--query_tloss", type=int, default=0)
    parser.add_argument("--weight_tloss", type=int, default=0)
    parser.add_argument("--ranking_loss", type=int, default=0)
    parser.add_argument("--disc_loss", type=int, default=0)
    parser.add_argument("--denoising", type=int, default=0)
    parser.add_argument("--multiple_decoder", type=int, default=0)
    parser.add_argument("--decoder_num", type=int, default=1)
    parser.add_argument("--loss_weight", type=int, default=0)

    parser.add_argument("--test1000", type=int, help="default to 0,1")
    parser.add_argument("--n_val", type=int, default=-1)
    parser.add_argument("--n_train", type=int, default=-1)
    parser.add_argument("--n_test", type=int, default=-1)

    # dataset params
    parser.add_argument("--trivia", type=int)
    parser.add_argument("--nq", type=int)
    parser.add_argument("--kary", type=int, default=30)
    parser.add_argument("--tree", type=int, default=1)
    parser.add_argument(
        "--query_type",
        type=str,
        default="gtq_doc_aug_qg",
        help="gtq -- use ground turth query;qg -- use qg; doc -- just use top64 doc token; doc_aug -- use random doc token. ",
    )
    parser.add_argument("--id_class", type=str, default="bert_k30_c30_4")

    parser_args = parser.parse_args()
    non_default = {
        opt.dest: getattr(parser_args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(parser_args, opt.dest) and opt.default != getattr(parser_args, opt.dest)
    }
    try:
        yaml_args = yaml.load(open(parser_args.config_path, "r"), Loader=yaml.FullLoader)

        yaml_args.update(non_default)  # override yaml args by command line args
        parser_dict = vars(parser_args)
        parser_dict.update(yaml_args)
        parser_args = argparse.Namespace(**parser_dict)
    except:
        raise ValueError("config path not found")

    # args post process
    parser_args.num_workers = parser_args.num_workers if parser_args.num_workers > 0 else os.cpu_count()
    parser_args.use_cache = True if parser_args.use_cache == 1 and parser_args.mode != "train" else False
    parser_args.tokenizer_name_or_path += parser_args.model_info
    parser_args.model_name_or_path += parser_args.model_info
    parser_args.n_gpu = len(parser_args.gpu_no)

    parser_args.position = parser_args.semantic_identifier == 2
    if parser_args.decoder_skipping:
        pass
    else:
        parser_args.exit_analysis = False

    if parser_args.mode == "train" and "doc" in parser_args.query_type:
        assert parser_args.contrastive_variant == ""
        parser_args.max_input_length = parser_args.doc_length
        logger.info("change max input length to", parser_args.doc_length)

    model_params_dict = {
        "large_no_adaptor": {
            "num_layers": 24,
            "num_decoder_layers": 12,
            "d_ff": 4096,
            "d_model": 1024,
            "num_heads": 16,
            "d_kv": 64,
            "adaptor_layer_num": 0,
        },
        "base_no_adaptor": {
            "num_layers": 12,
            "num_decoder_layers": 6,
            "d_ff": 3072,
            "d_model": 768,
            "num_heads": 12,
            "d_kv": 64,
            "adaptor_layer_num": 0,
        },
    }
    parser_args.model_params_keys = [
        "num_layers",
        "num_decoder_layers",
        "d_ff",
        "d_model",
        "num_heads",
        "d_kv",
        "adaptor_layer_num",
    ]
    for param in parser_args.model_params_keys:
        setattr(parser_args, param, model_params_dict[parser_args.model_info][param])
    parser_args.adaptor_efficient = parser_args.adaptor_layer_num > 0
    parser_args.adaptor_decode = parser_args.adaptor_layer_num > 0
    if parser_args.test1000:
        parser_args.n_val = 1000
        parser_args.n_train = 1000
        parser_args.n_test = 1000

    return parser_args
