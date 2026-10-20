# main.py 
import os, logging, argparse, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def build_boot_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", required=True)
    p.add_argument("--gpu", type=str, default="3")
    p.add_argument("--threads", type=int, default=5)
    return p

def build_main_parser(boot):
    p = argparse.ArgumentParser(parents=[boot], add_help=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--task", choices=["train","eval","test","vis"], default=None)
    return p

def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "main.log")
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    return logging.getLogger("main")

def get_out_dir(cfg):
    return cfg.get("paths", {}).get("out_dir") or os.path.abspath(os.path.join(cfg["paths"]["log_dir"], ".."))

def main():
    boot = build_boot_parser()
    boot_args, _ = boot.parse_known_args()

    from utils.resource import pin_resources
    pin_resources(gpu=boot_args.gpu, threads=boot_args.threads)

    import torch
    from utils import load_config, dump_config, set_seed

    p = build_main_parser(boot)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42), cfg.get("repro", {}).get("deterministic", False))
    out_dir = get_out_dir(cfg)
    logger = setup_logging(out_dir)
    dump_config(cfg, out_dir)

    logger.info(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    logger.info(f"Num GPUs = {torch.cuda.device_count()}")

    version = str(cfg["model"]["version"]).lower()
    task = (cfg.get("task") or args.task or "train").lower()
    exp_name = cfg.get("experiment_name", cfg.get("exp_name", "exp"))
    logger.info(f"Task={task} | Version={version} | Experiment={exp_name}")

    if task == "train":
        if version == "2d":
            from train_2d import run as run_2d
            run_2d(cfg, out_dir, logger)
        elif version == "3d":
            from train_3d import run as run_3d
            run_3d(cfg, out_dir, logger)
        else:
            from train_3d_uv import run as run_3d_uv
            run_3d_uv(cfg, out_dir, logger)
    else:
        if version == "2d":
            from evaluate_2d import run as eval_2d
            eval_2d(cfg, out_dir, logger)
        elif version == "3d":
            from evaluate_3d import run as eval_3d
            eval_3d(cfg, out_dir, logger)
        else:
            from evaluate_3d_uv import run as eval_3d_uv
            eval_3d_uv(cfg, out_dir, logger)


if __name__ == "__main__":
    main()
