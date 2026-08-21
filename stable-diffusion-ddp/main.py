import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image

MODEL_PATH = "/volumes/builtin/stable-diffusion-v1-5"
DATASET_PATH = "/volumes/builtin/landscape-images"
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
NUM_TRAIN_IMAGES = 10
NUM_STEPS = 5000
BATCH_SIZE = 4
SAVE_EVERY = 100
RESOLUTION = 512
MASK_RATIO = 0.25
DTYPE = torch.bfloat16

ARTIFACTS = Path("/artifacts")
ARTIFACTS.mkdir(exist_ok=True)
TENSORBOARD_DIR = ARTIFACTS / "tensorboard"


def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, dist.get_world_size()


def log(msg, rank):
    if rank == 0:
        print(msg, flush=True)


def load_images(path, num_images, device):
    raw = load_dataset(path, split="train")
    raw = raw.select(range(min(num_images, len(raw))))
    transform = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    images = torch.stack([transform(item["image"].convert("RGB")) for item in raw])
    return images.to(device, dtype=DTYPE)


def random_rectangle_mask(batch_size, h, w, ratio):
    masks = torch.zeros(batch_size, 1, h, w)
    mask_h = max(1, int(h * ratio ** 0.5))
    mask_w = max(1, int(w * ratio ** 0.5))
    for i in range(batch_size):
        top = torch.randint(0, h - mask_h + 1, (1,)).item()
        left = torch.randint(0, w - mask_w + 1, (1,)).item()
        masks[i, :, top : top + mask_h, left : left + mask_w] = 1.0
    return masks


def save_unet(unet, path):
    torch.save(unet.module.state_dict(), path)


@torch.no_grad()
def inpaint_demo(unet, vae, noise_scheduler, images, latents, null_embedding, device,
                 output_path=None, num_samples=4):
    unet.eval()
    images = images[:num_samples]
    latents = latents[:num_samples]
    B = images.shape[0]

    latent_h, latent_w = latents.shape[2], latents.shape[3]
    masks = random_rectangle_mask(B, latent_h, latent_w, MASK_RATIO).to(device, dtype=DTYPE)
    null_cond = null_embedding.expand(B, -1, -1)

    pixel_masks = torch.nn.functional.interpolate(masks, size=(RESOLUTION, RESOLUTION), mode="nearest")

    sample = torch.randn_like(latents)
    noise_scheduler.set_timesteps(50)
    for t in noise_scheduler.timesteps:
        t_batch = t.expand(B).to(device)
        noise_pred = unet(sample, t_batch, null_cond).sample
        sample = noise_scheduler.step(noise_pred, t, sample).prev_sample.to(DTYPE)

        noise = torch.randn_like(latents)
        noised_original = noise_scheduler.add_noise(latents, noise, t_batch).to(DTYPE)
        sample = sample * masks + noised_original * (1 - masks)

    decoded = vae.decode(sample / vae.config.scaling_factor).sample.clamp(-1, 1)

    grid = torch.cat([
        images * 0.5 + 0.5,
        (images * (1 - pixel_masks)) * 0.5 + 0.5,
        decoded * 0.5 + 0.5,
    ])
    if output_path is None:
        output_path = ARTIFACTS / "inpainting_demo.png"
    save_image(grid.float(), output_path, nrow=B)
    print(f"Demo saved to {output_path}", flush=True)
    unet.train()


@torch.no_grad()
def eval_loss(unet, noise_scheduler, latents, noise, timesteps_grid, null_embedding, device):
    """Denoising MSE over a fixed noise sample and a fixed grid of timesteps.

    The per-step training loss is noisy because it draws a random timestep each
    step. Holding the noise and the timesteps constant removes that variance, so
    the only thing that moves between evaluations is the model itself.
    """
    unet.eval()
    N = latents.shape[0]
    null_cond = null_embedding.expand(N, -1, -1)
    total = 0.0
    for t in timesteps_grid:
        t_batch = t.expand(N).to(device)
        noisy = noise_scheduler.add_noise(latents, noise, t_batch).to(DTYPE)
        noise_pred = unet(noisy, t_batch, null_cond).sample
        total += torch.nn.functional.mse_loss(noise_pred.float(), noise.float()).item()
    unet.train()
    return total / len(timesteps_grid)


def main():
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    log(f"Mask inpainting | GPUs: {world_size}", rank)
    log(f"model={MODEL_PATH}", rank)
    log(f"dataset={DATASET_PATH}", rank)
    log(f"steps={NUM_STEPS}", rank)
    log(f"lr={LEARNING_RATE}", rank)
    log(f"mask_ratio={MASK_RATIO}", rank)
    log(f"resolution={RESOLUTION}", rank)

    log("Loading models...", rank)
    vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae").to(device, dtype=DTYPE)
    unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    vae.requires_grad_(False)

    null_embedding = torch.zeros(1, 77, unet.config.cross_attention_dim, device=device, dtype=DTYPE)

    unet = unet.to(device, dtype=DTYPE)
    unet = DDP(unet, device_ids=[torch.cuda.current_device()], gradient_as_bucket_view=True)

    optimizer = AdamW(unet.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR)) if rank == 0 else None

    log(f"Loading {NUM_TRAIN_IMAGES} images and precomputing latents ({DATASET_PATH})...", rank)
    train_images = load_images(DATASET_PATH, NUM_TRAIN_IMAGES, device)
    with torch.no_grad():
        train_latents = vae.encode(train_images).latent_dist.mean * vae.config.scaling_factor

    eval_gen = torch.Generator(device=device).manual_seed(0)
    eval_noise = torch.randn(train_latents.shape, generator=eval_gen, device=device, dtype=DTYPE)
    eval_timesteps = torch.linspace(
        0, noise_scheduler.config.num_train_timesteps - 1, 10
    ).long().to(device)

    dataset = TensorDataset(train_latents)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    unet.train()
    global_step = 0
    epoch = 0

    checkpoints = sorted(ARTIFACTS.glob("checkpoint-*/unet.pt"))
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        global_step = int(latest_ckpt.parent.name.split("-")[1])
        state_dict = torch.load(latest_ckpt, map_location="cpu")
        unet.module.load_state_dict(state_dict)
        del state_dict
        log(f"Resumed from checkpoint at step {global_step}", rank)

    log("Training...", rank)
    while global_step < NUM_STEPS:
        sampler.set_epoch(epoch)
        for (latents,) in loader:
            if global_step >= NUM_STEPS:
                break

            B = latents.shape[0]
            null_cond = null_embedding.expand(B, -1, -1)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (B,), device=device
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(DTYPE)

            noise_pred = unet(noisy_latents, timesteps, null_cond).sample
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if global_step % 10 == 0:
                log(f"  step {global_step}/{NUM_STEPS}  loss={loss.item():.4f}", rank)

            if global_step % SAVE_EVERY == 0 and rank == 0:
                eval_value = eval_loss(unet, noise_scheduler, train_latents, eval_noise,
                                       eval_timesteps, null_embedding, device)
                if writer is not None:
                    writer.add_scalar("eval/loss", eval_value, global_step)
                log(f"  step {global_step}/{NUM_STEPS}  eval_loss={eval_value:.4f}", rank)

                ckpt = ARTIFACTS / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                save_unet(unet, ckpt / "unet.pt")
                log(f"Checkpoint saved to {ckpt}/unet.pt", rank)

                inpaint_demo(unet, vae, noise_scheduler, train_images, train_latents, null_embedding, device,
                             output_path=ARTIFACTS / f"inpainting-step-{global_step}.png")

        epoch += 1

    if writer is not None:
        writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
