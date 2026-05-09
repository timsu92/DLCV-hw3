"""DDP training script.

Launch with:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 -m src.train

Or single-GPU smoke test:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m src.train --epochs 1
"""

from __future__ import annotations

import argparse
import contextlib
import os
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.augment import get_train_transform, get_val_transform
from src.dataset import (
    CellDataset,
    build_coco_annotations,
    load_or_build_annotations,
    oversample_rare_classes,
)
from src.model import build_model
from src.utils import cross_class_nms, encode_mask, resize_binary_mask

TRAIN_DIR = Path("data/train")
CACHE_TRAIN = Path("data/train_annotations.json")
CACHE_VAL = Path("data/val_annotations.json")
CHECKPOINT_DIR = Path("checkpoints") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2, help="per-GPU batch size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--score-thresh", type=float, default=0.05)
    p.add_argument(
        "--min-size",
        type=int,
        nargs="+",
        default=[480, 512, 544],
        help="shorter-side targets for multi-scale training (multiples of 32)",
    )
    p.add_argument(
        "--max-size",
        type=int,
        default=640,
        help="max image side after resizing",
    )
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="enable gradient checkpointing on ResNet layer2-4 to save ~30% activation memory",
    )
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def evaluate(model_without_ddp, val_loader, val_coco_json: dict, device):
    """Run COCOeval on val set. Returns AP50."""
    model_without_ddp.eval()
    torch.cuda.empty_cache()

    coco_gt = COCO()
    coco_gt.dataset = val_coco_json
    coco_gt.createIndex()

    # Original image sizes needed to scale predicted masks back to GT resolution.
    # Val images are pre-resized to 640 px (get_val_transform), so model output
    # masks are at ~640 px; COCOeval requires them at the original annotation size.
    img_sizes = {img["id"]: (img["height"], img["width"]) for img in val_coco_json["images"]}

    results = []
    for imgs, targets in val_loader:
        imgs = [img.to(device) for img in imgs]
        preds = model_without_ddp(imgs)
        for pred, target in zip(preds, targets):
            pred = cross_class_nms(pred)
            image_id = target["image_id"].item()
            orig_h, orig_w = img_sizes[image_id]
            for box, label, score, mask in zip(
                pred["boxes"], pred["labels"], pred["scores"], pred["masks"]
            ):
                binary = (mask[0] > 0.5).cpu().numpy()
                binary = resize_binary_mask(binary, orig_h, orig_w)
                rle = encode_mask(binary)
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": label.item(),
                        "score": score.item(),
                        "segmentation": rle,
                        "bbox": box.tolist(),
                    }
                )

    if not results:
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    # Default maxDets=100 caps recall on dense images (up to 772 instances/image).
    # Raise to 1500 to cover the full instance range in this dataset.
    evaluator.params.maxDets = [1, 10, 1500]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[1])  # stats[1] = AP @ IoU=0.50


def save_checkpoint(path: Path, epoch: int, model, optimizer, ap50: float):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ap50": ap50,
        },
        path,
    )


def main():
    args = parse_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    train_coco, val_coco = load_or_build_annotations(TRAIN_DIR, CACHE_TRAIN, CACHE_VAL)

    # Oversample rare classes (class3, class4) × 3 to compensate imbalance
    train_folders = [img["file_name"] for img in train_coco["images"]]
    train_folders_os = oversample_rare_classes(TRAIN_DIR, train_folders, factor=3)
    train_coco_os = build_coco_annotations(TRAIN_DIR, train_folders_os)

    train_ds = CellDataset(
        TRAIN_DIR,
        train_coco_os,
        transforms=get_train_transform(),
    )
    val_ds = CellDataset(TRAIN_DIR, val_coco, transforms=get_val_transform())

    train_sampler = DistributedSampler(train_ds, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    # val loader: no DistributedSampler — rank 0 evaluates full val set
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_model(
        min_size=tuple(args.min_size),
        max_size=args.max_size,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_optim_steps = (len(train_loader) // args.accum_steps) * args.epochs
    warmup_steps = min(args.warmup_steps, total_optim_steps // 5)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps)),
            CosineAnnealingLR(
                optimizer, T_max=max(1, total_optim_steps - warmup_steps), eta_min=1e-6
            ),
        ],
        milestones=[warmup_steps],
    )

    scaler = GradScaler("cuda")
    best_ap50 = 0.0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        epoch_loss = 0.0

        for step, (imgs, targets) in enumerate(train_loader):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            is_last_accum = (step + 1) % args.accum_steps == 0

            ctx = model.no_sync() if not is_last_accum else contextlib.nullcontext()
            with ctx:
                with autocast("cuda"):
                    loss_dict = model(imgs, targets)
                    loss = sum(loss_dict.values()) / args.accum_steps
                scaler.scale(loss).backward()

            if is_last_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.accum_steps

        # Flush remaining gradients if last epoch batch didn't land on an accum boundary
        if len(train_loader) % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if is_main:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}", flush=True)

        # Evaluate on rank 0 only
        if is_main:
            ap50 = evaluate(model.module, val_loader, val_coco, device)
            print(f"  Val AP50: {ap50:.4f}  (best: {best_ap50:.4f})", flush=True)

            save_checkpoint(
                CHECKPOINT_DIR / "last_model.pth",
                epoch + 1,
                model,
                optimizer,
                ap50,
            )
            print("  Saved last checkpoint", flush=True)

            if (epoch + 1) % 3 == 0:
                save_checkpoint(
                    CHECKPOINT_DIR / f"epoch_{epoch + 1:03d}.pth",
                    epoch + 1,
                    model,
                    optimizer,
                    ap50,
                )
                print(f"  Saved periodic checkpoint (epoch {epoch + 1})", flush=True)

            if ap50 > best_ap50:
                best_ap50 = ap50
                save_checkpoint(
                    CHECKPOINT_DIR / "best_model.pth",
                    epoch + 1,
                    model,
                    optimizer,
                    ap50,
                )
                print(f"  Saved best checkpoint (AP50={ap50:.4f})", flush=True)

        dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main()
