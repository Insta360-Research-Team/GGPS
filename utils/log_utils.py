import os
import wandb
import torch

def safe_log_metrics_pl(log_writer, metrics, step):
    """Lightning TensorBoardLogger.log_metrics 需 Python float/int 或 Tensor；numpy 标量会触发 ValueError。"""
    if log_writer is None:
        return
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            clean[k] = float(v.detach().cpu().item())
        else:
            clean[k] = float(v)
    try:
        log_dir = getattr(log_writer, "log_dir", None)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_writer.log_metrics(clean, step)
    except (FileNotFoundError, OSError) as e:
        print(f"[Warning] TensorBoard log_metrics failed: {e}")

def tensorboard_log_image(log_writer, tag: str, image_tensor, step):
    log_writer.experiment.add_image(
        tag,
        image_tensor,
        step,
    )

def wandb_log_image(log_writer, tag: str, image_tensor, step):
    image_dict = {
        tag: wandb.Image(image_tensor),
    }
    log_writer.experiment.log(
        image_dict,
        step=step,
    )