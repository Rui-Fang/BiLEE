from functools import partial
import json
import os
from pathlib import Path
import sys
import time
import warnings

import lightning.pytorch as pl
import torch
from arguments import parsers_parser
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.profilers import PyTorchProfiler
from loguru import logger
from gen_trainer import adaptive_T5
from main_metrics import MRR100, recall
from exit_models import decode_token, l1_query
from main_utils import dec_2d, numerical_decoder
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from exit_models import T5FineTunerSmall
import yaml
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

YOUR_API_KEY = ""

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

time_str = time.strftime("%Y%m%d-%H:%M")
time_str_short = time.strftime("%m%d_%H%M")


def init_trainer(args):
    if YOUR_API_KEY != "" and args.wandb == 1:
        logger.info("Using wandb logger")
        os.environ["WANDB_API_KEY"] = YOUR_API_KEY
        pl_logger = WandbLogger(
            name=args.exp_postfix + "-" + time_str,
            group=args.exp_name,
            project=f"BiLEE - {args.dataset_name}",
            entity="",
            tags=[*args.tags],
        )
    else:
        pl_logger = None  # TensorBoardLogger(f'/logs/{time_str}/')
    filename_prefix = args.exp_postfix

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename=filename_prefix + "_{epoch}-{recall1:.4f}",
        monitor=args.ckpt_monitor,
        save_on_train_epoch_end=True,
        mode=args.monitor_mode,
        save_top_k=2,
        every_n_epochs=1,
    )
    earlystopping_callback = EarlyStopping(monitor=args.ckpt_monitor, mode=args.monitor_mode, patience=10, verbose=True, check_finite=True)
    lr_monitor = pl.callbacks.LearningRateMonitor()
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        devices=args.gpu_no,
        num_sanity_val_steps=0,
        max_epochs=args.num_train_epochs,
        precision="16-mixed" if args.fp_16 else 32,
        gradient_clip_val=args.max_grad_norm,
        enable_checkpointing=True,
        logger=pl_logger,
        callbacks=[lr_monitor, checkpoint_callback, earlystopping_callback, RichProgressBar()],
        strategy="ddp_find_unused_parameters_true",
        accelerator=args.accelerator,
        limit_val_batches=0.5,
        limit_train_batches=args.train_pct,
        log_every_n_steps=10,
    )
    trainer = pl.Trainer(**train_params)
    return trainer


def train(args, trainer):
    model = T5FineTunerSmall(args)
    trainer.fit(model, ckpt_path=args.continue_ckpt)


def train_gen(args, trainer):
    model = adaptive_T5(args)
    trainer.fit(model, ckpt_path=args.continue_ckpt)


@torch.no_grad()
def learn_than_test(args, trainer):
    def distance(x, target):
        return target["MRR100"] - x["MRR100"]

    dict_no_cache = lambda x: {k: v for k, v in x.items() if "inf" not in k}
    model = T5FineTunerSmall(args)
    model.to(f"cuda:{args.gpu_no[0]}")
    model.eval()
    val_dataset = l1_query(model.args, model.tokenizer, num_samples=False, task="dev")

    delta = 0.01
    epsilon = 0.01
    batch_size = 64
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=model.args.num_workers,
    )

    batch_num = 12
    n_samples = batch_num * batch_size
    from bounds import HB_mu_plus

    ### CITE: https://github.com/aangelopoulos/ltt/tree/77252828e918410406a06e97471460c1994b41d0
    hoeffding_p = lambda x, n=n_samples, d=delta: HB_mu_plus(x + 1e-9, n, d, 1000)

    sequence_params = model.get_threshold()
    default_results = model.one_validation_test(dataloader, sequence_params, batch_num=batch_num * 3)
    logger.info(f"model_thershold_results: {dict_no_cache(default_results)}")

    zero_point = torch.zeros_like(sequence_params)
    zero_results = model.one_validation_test(dataloader, zero_point, batch_num=batch_num * 3)
    logger.info(f"zero_results: {dict_no_cache(zero_results)}")

    full_point = torch.ones_like(sequence_params) * 10
    full_results = model.one_validation_test(dataloader, full_point, batch_num=batch_num * 3)
    logger.info(f"full_results: {dict_no_cache(full_results)}")

    if args.threshold_path and os.path.exists(args.threshold_path):
        saved_params = torch.load(args.threshold_path)
        start_point = saved_params
        logger.info("threshold loaded")
    else:
        logger.info("start from model parameters")
        start_point = torch.ones((12, 10))
    torch.save(start_point, Path(args.output_dir) / "skip_params_sequence.pt")

    ## stage 1
    if not args.CALM_thresholds:
        start_results, start_tagets = model.threshold_compare(dataloader, start_point, full_point, batch_num // 2)
        logger.info(f"start_results: {dict_no_cache(start_results)}")
        logger.info(f"start_targets: {dict_no_cache(start_tagets)}")
        error_rate = distance(start_results, start_tagets)
        results = start_results
        s1_start_point = (start_point + 0.01).clip(0, 1)
        s1_start_point[:, 4:] = 0.0
        max_sequence_length = len(results["skip_dict"]["mean"])
        s1_start_point[:, max_sequence_length:] = 0.0
        logger.info("stage 1 start")
        logger.info(s1_start_point)
        flag = 1
        previous_point = s1_start_point
        failed_layer = torch.zeros(s1_start_point.shape[0])
        failed_token = torch.zeros(s1_start_point.shape[1])
        step = 0.02
        threshold_compare = partial(model.threshold_compare, dataloader=dataloader, target_threshold=full_point)
        while flag != 0:
            flag = 0
            for token_no in range(previous_point.shape[1]):
                if (previous_point[:, token_no] <= 0.01).sum() == len(previous_point[:, token_no]) or failed_token[token_no] >= 7:
                    continue
                test_point = previous_point.clone()
                test_point[:, token_no] = test_point[:, token_no] - (step / (failed_token[token_no] + 1))
                test_magnification = 3 + failed_token[token_no].item()
                results_temp, results_full = threshold_compare(test_threshold=test_point, batch_num=batch_num * test_magnification)
                error_rate = distance(results_temp, results_full)
                logger.info(f"test_results: {dict_no_cache(results_temp)}")
                logger.info(f"test_targets: {dict_no_cache(results_full)}")
                sample_size = int(n_samples * test_magnification)
                p_value = hoeffding_p(error_rate, sample_size)
                if error_rate > 0 and p_value > epsilon:
                    logger.info(f"t{token_no} :: sample_size: {sample_size}, error_rate: {error_rate}, p: {p_value},  failed")
                    failed_token[token_no] += 1
                    if failed_token[token_no] >= 1:
                        flag += 1
                else:
                    logger.info(f"t{token_no} :: sample_size: {sample_size}, error_rate: {error_rate}, p: {p_value},  passed")
                    flag += 1
                    previous_point = test_point.clone()
                    results = results_temp

            logger.info(f"one round finished")
            logger.info(f"results: {dict_no_cache(results)}")
            print(previous_point)
            torch.save(previous_point, Path(args.output_dir) / "skip_params_sequence.pt")

    else:
        error_rate = distance(full_results, full_results)
        threshold = model.model.decoder.block[0].skip_params_sequence[0].item()
        logger.info(f"start from 1.0")
        previous_point = torch.ones((12, 10))
        while hoeffding_p(error_rate) < epsilon:
            test_point = previous_point - 0.01
            results_temp, results_full = model.threshold_compare(dataloader, test_point, full_point, batch_num * 3)
            error_rate = distance(results_temp, results_full)
            p_rate = hoeffding_p(error_rate)
            if error_rate >= 0 and p_rate > epsilon:
                logger.info(f"error_rate: {error_rate}, p={p_rate}, failed")
                break
            else:
                logger.info(f"error_rate: {error_rate}, p={p_rate}, passed")
                previous_point = test_point
            logger.info(f"results: {dict_no_cache(results_temp)}")
            logger.info(f"threshold:{previous_point}")
            torch.save(previous_point, Path(args.output_dir) / "skip_params_sequence.pt")

    logger.info("finished")
    final_results = model.one_validation_test(dataloader, previous_point, batch_num=batch_num * 16)
    logger.info(f"full_results: {dict_no_cache(final_results)}")
    logger.info(f"\n{previous_point}")


def inference(args, trainer):
    model = T5FineTunerSmall(args, train=False)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    for k in list(state_dict.keys()):
        if "teacher_model." in k:
            del state_dict[k]
    logger.info(model.load_state_dict(state_dict, strict=False))
    args.tokenizer_name_or_path = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    num_samples = args.n_test if args.n_test >= 0 else None
    dataset = l1_query(args, tokenizer, num_samples=num_samples, task="test")
    model.to(f"cuda:{args.gpu_no[0]}")
    if args.threshold_path != "" and args.threshold_path is not None:
        threshold = torch.load(args.threshold_path)
        for layer_no in range(len(model.model.decoder.block)):
            model.model.decoder.block[layer_no].skip_params_sequence = torch.nn.Parameter(threshold[layer_no].to(f"cuda:{args.gpu_no[0]}"))
        logger.warning("skip_params_sequence loaded")
    logger.info(model.get_threshold())

    model.eval()
    if args.full_eval:
        model.set_threshold(torch.ones_like(model.get_threshold() * 4396))
        logger.warning("full_eval mode")
    test_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    trainer.test(model, dataloaders=test_dataloader)
    logger.info(f"Peak gpu usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.4f} GiB")
    logger.info(f"output dir: {args.output_dir}")
    return


def calculate(args):  # deprecated
    recall_value = recall(args)
    mrr_value = MRR100(args)
    return recall_value, mrr_value


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=250, profile="short", edgeitems=10)
    args = parsers_parser()
    seed_everything(args.seed, workers=True)
    if args.nq:
        args.dataset_name = "NQ"
        short_name = "NQ"
    elif args.trivia:
        args.dataset_name = "TriviaQA"
        short_name = "TQA"
    else:
        args.dataset_name = "kary"
    time_str_short = time.strftime("%m%d") if args.mode != "train" else time_str_short
    if args.output_dir is None or args.output_dir == "":
        if args.mode in ["train", "train_gen"]:
            args.output_dir = (
                dir_path + f"/logs/{args.dataset_name}/{args.mode}/{time_str_short}-[{args.exp_name.split('|')[0]}]-[{args.exp_postfix}]"
            )
        else:
            args.output_dir = str(Path(args.ckpt_path).parent / args.mode / args.exp_postfix)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    yaml.dump(vars(args), open(args.output_dir + f"/configs_{args.mode}.yaml", "w"))
    important_info_list = [
        args.dataset_name,
        args.query_type,
        args.model_info,
        args.id_class,
        args.test_set,
        args.ckpt_monitor,
        "dem:",
        str(args.semantic_identifier),
        "ada:",
        str(args.adaptor_decode),
        "adaeff:",
        str(args.adaptor_efficient),
        "adanum:",
        str(args.adaptor_layer_num),
        "RDrop:",
        str(args.dropout_rate),
        str(args.Rdrop),
        str(args.Rdrop_only_decoder),
    ]

    args.query_info = "_".join(important_info_list)
    args.tag_info = "{}_lre{}d{}".format(args.query_info, str(float(args.learning_rate * 1e4)), str(float(args.decoder_learning_rate * 1e4)))
    args.res1_save_path = args.output_dir + f"/Beam{args.num_return_sequences}-BS{args.eval_batch_size}.tsv"
    logger.configure(extra={"postfix": args.exp_postfix, "dataset": short_name, "mode": args.mode})
    # Logging
    logger.remove()
    log_name = (
        f"/{args.mode}.log"
        if args.mode != "eval"
        else f"/Beam{args.num_return_sequences}-BS{args.eval_batch_size}F{args.flops_profiling}C{args.use_cache}.log"
    )
    if args.log:
        logger.add(
            args.output_dir + log_name,
            format="|{time:YYYY-MM-DD HH:mm:ss} - <g>{elapsed}</g>|<c>{module}:{line}:{function}</c>" + " - <lvl>{message}</lvl>",
            level="DEBUG",
        )
    logger.add(
        sys.stderr, format="|<y>{extra[dataset]}</y>-<g>{extra[postfix]}</g>|<c>{module}.{function}</c>" + " - <lvl>{message}</lvl>", level="INFO"
    )
    logger.debug(json.dumps(vars(args), indent=4, sort_keys=False))
    logger.warning(args.tag_info)
    logger.info(f"Current working dir : {os.getcwd()}")
    logger.info(f"torch version : {torch.__version__}")  # 1.10.0+cu113
    logger.info(f"pl version : {pl.__version__}")  # 1.4.9
    logger.info(f"CUDA is : {torch.cuda.is_available()}")
    logger.info(f"available GPU : {torch.cuda.device_count()}")
    try:
        logger.info(f"CUDA visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    except:
        logger.info("All CUDA devices are visiable")
    logger.info(f"Using GPU : {args.gpu_no}")
    logger.info(f"dir path : {args.output_dir}")
    logger.info(f"parent path : {parent_path}")
    trainer = init_trainer(args)
    if args.mode == "train":
        train(args, trainer)
    if args.mode == "train_gen":
        logger.warning("train_gen mode")
        train_gen(args, trainer)
    elif args.mode == "eval":
        args.recall_num = [1, 5, 10, 20, 50, 100]
        logger.warning("Early Exitting {}".format("activated" if args.decoder_skipping else "deactivated"))
        logger.warning("Beam Pruning {}".format("activated" if args.beam_pruning else "deactivated"))
        logger.warning("{}".format("Use KV cache" if args.use_cache else "Not use KV cache"))
        inference(args, trainer)
    elif args.mode == "LTT":
        learn_than_test(args, trainer)

    # elif args.mode == "calculate":
    #     args.res1_save_path = args.result_path
    #     args.recall_num = [1, 5, 10, 20, 50, 100]
    #     calculate(args)
