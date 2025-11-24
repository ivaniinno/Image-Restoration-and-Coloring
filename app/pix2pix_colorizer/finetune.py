"""Finetune / train script for colorization model with WGAN-GP and R1.

Features:
- WGAN-GP gradient penalty (lambda_gp)
- Optional R1 regularization on critic (lambda_r1)
- Multiple critic steps per generator step (critic_steps)
"""
import os
import glob
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import ImageColorizationDataset
from models import build_models
from utils import lab_to_rgb
from imageio import imwrite


def gradient_penalty(critic, real, fake, condition, device):
    # interpolation
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    interpolates_logits = critic(interpolates, condition)
    grad_outputs = torch.ones_like(interpolates_logits, device=device)
    gradients = torch.autograd.grad(
        outputs=interpolates_logits,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def r1_regularization(critic, real, condition, device):
    # r1 regularization: gradient of critic output wrt real images
    real = real.requires_grad_(True)
    real_logits = critic(real, condition)

    # sum outputs to get gradients
    grad_outputs = torch.ones_like(real_logits, device=device)
    gradients = torch.autograd.grad(
        outputs=real_logits,
        inputs=real,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    r1 = (gradients.pow(2).sum(1)).mean()
    return r1


def train(
    generator,
    critic,
    dataloader,
    optim_G,
    optim_C,
    device,
    epochs=10,
    critic_steps=5,
    lambda_gp=10.0,
    lambda_r1=0.0,
    out_dir='checkpoints',
    display_step=100,
):
    os.makedirs(out_dir, exist_ok=True)
    step = 0
    for epoch in range(epochs):
        g_loss_accum = 0.0
        c_loss_accum = 0.0
        for batch_idx, (real, condition) in enumerate(dataloader):
            real = real.to(device)
            condition = condition.to(device)

            # multiple critic steps
            for _ in range(critic_steps):
                optim_C.zero_grad()
                fake = generator(condition).detach()
                fake_logits = critic(fake, condition)
                real_logits = critic(real, condition)

                # maximize real_logits - fake_logits => minimize -(real - fake)
                loss_C = fake_logits.mean() - real_logits.mean()

                if lambda_gp > 0:
                    gp = gradient_penalty(critic, real, fake, condition, device)
                    loss_C = loss_C + lambda_gp * gp

                if lambda_r1 > 0:
                    r1 = r1_regularization(critic, real, condition, device)
                    loss_C = loss_C + lambda_r1 * r1

                loss_C.backward()
                optim_C.step()

            # generator step
            optim_G.zero_grad()
            fake = generator(condition)
            recon_loss = torch.nn.functional.l1_loss(fake, real)

            adv = -critic(fake, condition).mean()
            loss_G = recon_loss + 1e-3 * adv
            loss_G.backward()
            optim_G.step()

            g_loss_accum += loss_G.item()
            c_loss_accum += loss_C.item()
            step += 1

            if step % display_step == 0:
                print(f"Epoch {epoch} Step {step}: G={g_loss_accum/display_step:.4f} C={c_loss_accum/display_step:.4f}")
                g_loss_accum = 0.0
                c_loss_accum = 0.0

        gen_path = os.path.join(out_dir, f"ResUnet_epoch_{epoch}.pt")
        crit_path = os.path.join(out_dir, f"PatchGAN_epoch_{epoch}.pt")
        torch.save(generator.state_dict(), gen_path)
        torch.save(critic.state_dict(), crit_path)
        print(f"Saved checkpoints: {gen_path}, {crit_path}")

        prune_checkpoints(out_dir, prefix='ResUnet', keep=5)
        prune_checkpoints(out_dir, prefix='PatchGAN', keep=5)


def find_latest_weight(path, pattern='*.pt'):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, pattern))
        if not files:
            return None
        files = sorted(files, key=os.path.getmtime)
        return files[-1]
    elif os.path.isfile(path):
        return path
    return None


def prune_checkpoints(out_dir, prefix='ResUnet', keep=5):
    pattern = os.path.join(out_dir, f"{prefix}_*.pt")
    files = glob.glob(pattern)
    if len(files) <= keep:
        return
    files = sorted(files, key=os.path.getmtime)
    to_remove = files[:-keep]
    for f in to_remove:
        try:
            os.remove(f)
            print('Removed old checkpoint', f)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ab_path', default=os.environ.get('AB_PATH', '/kaggle/input/image-colorization/ab/ab/ab1.npy'))
    parser.add_argument('--l_path', default=os.environ.get('L_PATH', '/kaggle/input/image-colorization/l/gray_scale.npy'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--critic_steps', type=int, default=5)
    parser.add_argument('--lambda_gp', type=float, default=10.0)
    parser.add_argument('--lambda_r1', type=float, default=0.0)
    parser.add_argument('--resume_gen', default=None, help='path to generator weights (.pt) or directory')
    parser.add_argument('--resume_crit', default=None, help='path to critic weights (.pt) or directory')
    parser.add_argument('--out_dir', default='checkpoints')
    parser.add_argument('--display_step', type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ab = np.load(args.ab_path)
    l = np.load(args.l_path)

    dataset = ImageColorizationDataset(l, ab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    generator, critic = build_models(device=device)

    if args.resume_gen:
        gen_w = find_latest_weight(args.resume_gen)
        if gen_w:
            print('Loading generator weights from', gen_w)
            generator.load_state_dict(torch.load(gen_w, map_location=device))
    if args.resume_crit:
        crit_w = find_latest_weight(args.resume_crit)
        if crit_w:
            print('Loading critic weights from', crit_w)
            critic.load_state_dict(torch.load(crit_w, map_location=device))

    optim_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optim_C = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9))

    train(
        generator,
        critic,
        loader,
        optim_G,
        optim_C,
        device,
        epochs=args.epochs,
        critic_steps=args.critic_steps,
        lambda_gp=args.lambda_gp,
        lambda_r1=args.lambda_r1,
        out_dir=args.out_dir,
        display_step=args.display_step,
    )


if __name__ == '__main__':
    main()
