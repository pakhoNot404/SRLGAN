import lightning.pytorch as pl
import torch
import torchmetrics.functional as metrics
from lightning.pytorch.tuner import Tuner
from torch import optim, nn
from torchvision import io
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms.functional import convert_image_dtype, resize
from torchvision.utils import save_image

from dataset2d import SR2DDataModule
from lossSaver import LossSaver


class SRLGANModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.netG = SRCNN()
        self.netD = NetD()
        self.lossG = LossG()
        self.lossD = LossD()
        self.automatic_optimization = False  # gan禁用自动optimize

    def training_step(self, batch, batch_idx):
        optimD, optimG = self.optimizers()
        # train d
        putin = batch["putin"]
        target = batch["target"]
        pred = self.netG(putin)
        d_fake = self.netD(pred)
        d_real = self.netD(target)
        lossD = self.lossD(d_real, d_fake)
        optimD.zero_grad()
        self.manual_backward(lossD)
        optimD.step()
        # train g
        putin = batch["putin"]
        target = batch["target"]
        pred = self.netG(putin)
        d_fake = self.netD(pred)
        lossG = self.lossG(target, pred, d_fake)
        optimG.zero_grad()
        self.manual_backward(lossG)
        optimG.step()
        self.log("train/lossG", lossG)
        self.log("train/lossD", lossD)

    def validation_step(self, batch, batch_idx):
        pred = self.netG(batch["putin"])
        d_fake = self.netD(pred)
        d_real = self.netD(batch["target"])
        lossG = self.lossG(batch["target"], pred, d_fake)
        lossD = self.lossD(d_real, d_fake)
        self.log("val/lossG", lossG)
        self.log("val/lossD", lossD)
        self.log(
            "metrics/psnr",
            metrics.peak_signal_noise_ratio(pred, batch["target"]),
            prog_bar=True,
        )
        self.log(
            "metrics/ssim",
            metrics.structural_similarity_index_measure(pred, batch["target"]),
            prog_bar=True,
        )

    def forward(self, putin):
        pred = self.netG(putin)
        return pred

    def configure_optimizers(self):
        optimD = optim.AdamW(self.netD.parameters())
        optimG = optim.AdamW(self.netG.parameters())
        return optimD, optimG


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.width = 128
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(1, self.width, kernel_size=9, padding=9 // 2),
            nn.ReLU(),
            nn.Conv2d(self.width, self.width // 2, kernel_size=5, padding=5 // 2),
            nn.ReLU(),
            nn.Conv2d(self.width // 2, 1, kernel_size=5, padding=5 // 2),
        )

    def forward(self, x):
        pred = self.net(x.float())
        return pred


class NetD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x.float()).view(batch_size))


class LossG(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = nn.Sequential(
            *list(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.children())[:31]
        ).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.l2_loss_fn = nn.MSELoss()
        self.smooth_l1 = LossSmoothL1()
        self.cl_loss = CLLoss()
        self.init_weight = (1, 1e-3, 6e-3, 2e-8)

    def forward(self, target, pred, d_pred):
        device = target.device
        self.vgg.to(device)

        loss_pixel = self.l2_loss_fn(pred, target)
        loss_prec = self.l2_loss_fn(
            self.vgg(pred.repeat(1, 3, 1, 1)), self.vgg(target.repeat(1, 3, 1, 1))
        )
        loss_adv = torch.mean(1 - d_pred)
        loss_smooth = self.smooth_l1(pred)
        loss_cl = self.cl_loss(target, pred)
        return loss_pixel + 1e-3 * loss_adv + 6e-3 * loss_prec + 2e-8 * loss_smooth + 0.1 * loss_cl


class CLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilinear = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.mse = nn.MSELoss()

    def forward(self, sr, hr):
        cl = self.bilinear(sr)
        lr = self.bilinear(hr)
        return self.mse(cl, lr)


class LossSmoothL1(nn.Module):
    @staticmethod
    def forward(i):
        loss_P = torch.mean(torch.abs(i[:, :, :-1, :] - i[:, :, 1:, :]))
        loss_L = torch.mean(torch.abs(i[:, :, :, :-1] - i[:, :, :, 1:]))
        return loss_P + loss_L


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class LossD(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, d_real: torch.Tensor, d_fake: torch.Tensor):
        return 1 - d_real.mean() + d_fake.mean()


if __name__ == "__main__":
    device = 0
    pl.seed_everything(42)
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                filename="srlgan",
                monitor="metrics/psnr",
                mode="max",
                auto_insert_metric_name=False,
            ),
            LossSaver(),
        ],
        max_epochs=200,
        devices=[device],
    )
    dm = SR2DDataModule(32)
    model = SRLGANModule()
    # %% 2 find batch_size
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=dm)
    # %% 3 train
    trainer.fit(model, dm,
                # ckpt_path=f"checkpoints/srlgan.ckpt"
                )
    # %% 4 test
    model = SRLGANModule.load_from_checkpoint(f'checkpoints/srlgan.ckpt')
    model.eval()
    #
    putin1 = convert_image_dtype(
        io.read_image('data/soilct/50p_d0/009.png', io.ImageReadMode.GRAY)[None]).to(f'cuda:{device}')
    pred = model(putin1)
    save_image(pred, '.png')
    #
    putin2 = resize(putin1, [128, 128], antialias=True)
    pred = model(putin2)
    psnr = metrics.peak_signal_noise_ratio(putin1, pred)
    ssim = metrics.structural_similarity_index_measure(putin1, pred)
    print(f'psnr: {psnr:.3f}')
    print(f'ssim: {ssim:.3f}')
    save_image(pred, 'srlgan2.png')
