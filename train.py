import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.generator import Generator
from models.discriminator import Discriminator
from dataset import ImagetoImageDataset
from config import config
import itertools
from torchvision import transforms

class AgingGAN(pl.LightningModule):
    def __init__(self, hparams):
        super(AgingGAN, self).__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

        # Generators for all transitions
        self.genYoung2Middle = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.genMiddle2Young = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.genMiddle2Old = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.genOld2Middle = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        
        # Discriminators for each age group
        self.disYoung = Discriminator(hparams['ndf'])
        self.disMiddle = Discriminator(hparams['ndf'])
        self.disOld = Discriminator(hparams['ndf'])

        # Cache for generated images
        self.generated_middle_from_young = None
        self.generated_young_from_middle = None
        self.generated_old_from_middle = None
        self.generated_middle_from_old = None
        self.real_young = None
        self.real_middle = None
        self.real_old = None

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        dataset = ImagetoImageDataset(
            domainA_dir=config['trainYoung_dir'],
            domainB_dir=config['trainMiddle_dir'],
            domainC_dir=config['trainOld_dir'],
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        real_young, real_middle, real_old = batch

        # Generator Training
        g_opt.zero_grad()
        
        # Identity losses
        same_middle_y = self.genYoung2Middle(real_middle)
        same_young_m = self.genMiddle2Young(real_young)
        same_old_m = self.genMiddle2Old(real_old)
        same_middle_o = self.genOld2Middle(real_middle)
        
        loss_identity = (
            F.l1_loss(same_middle_y, real_middle) +
            F.l1_loss(same_young_m, real_young) +
            F.l1_loss(same_old_m, real_old) +
            F.l1_loss(same_middle_o, real_middle)
        ) * self.hparams['identity_weight']

        # GAN losses
        fake_middle_y = self.genYoung2Middle(real_young)
        fake_young_m = self.genMiddle2Young(real_middle)
        fake_old_m = self.genMiddle2Old(real_middle)
        fake_middle_o = self.genOld2Middle(real_old)

        loss_GAN = (
            F.mse_loss(self.disMiddle(fake_middle_y), torch.ones_like(self.disMiddle(fake_middle_y))) +
            F.mse_loss(self.disYoung(fake_young_m), torch.ones_like(self.disYoung(fake_young_m))) +
            F.mse_loss(self.disOld(fake_old_m), torch.ones_like(self.disOld(fake_old_m))) +
            F.mse_loss(self.disMiddle(fake_middle_o), torch.ones_like(self.disMiddle(fake_middle_o)))
        ) * self.hparams['adv_weight']

        # Cycle losses
        recovered_young = self.genMiddle2Young(fake_middle_y)
        recovered_middle_y = self.genYoung2Middle(fake_young_m)
        recovered_middle_o = self.genOld2Middle(fake_old_m)
        recovered_old = self.genMiddle2Old(fake_middle_o)

        loss_cycle = (
            F.l1_loss(recovered_young, real_young) +
            F.l1_loss(recovered_middle_y, real_middle) +
            F.l1_loss(recovered_middle_o, real_middle) +
            F.l1_loss(recovered_old, real_old)
        ) * self.hparams['cycle_weight']

        # Total generator loss
        g_loss = loss_identity + loss_GAN + loss_cycle
        
        self.manual_backward(g_loss)
        g_opt.step()

        # Store generated images
        self.generated_middle_from_young = fake_middle_y.detach()
        self.generated_young_from_middle = fake_young_m.detach()
        self.generated_old_from_middle = fake_old_m.detach()
        self.generated_middle_from_old = fake_middle_o.detach()

        # Discriminator Training
        d_opt.zero_grad()

        # Young discriminator
        loss_D_young = (
            F.mse_loss(self.disYoung(real_young), torch.ones_like(self.disYoung(real_young))) +
            F.mse_loss(self.disYoung(self.generated_young_from_middle), torch.zeros_like(self.disYoung(self.generated_young_from_middle)))
        ) * 0.5

        # Middle discriminator
        loss_D_middle = (
            F.mse_loss(self.disMiddle(real_middle), torch.ones_like(self.disMiddle(real_middle))) +
            F.mse_loss(self.disMiddle(self.generated_middle_from_young), torch.zeros_like(self.disMiddle(self.generated_middle_from_young))) +
            F.mse_loss(self.disMiddle(self.generated_middle_from_old), torch.zeros_like(self.disMiddle(self.generated_middle_from_old)))
        ) * 0.5

        # Old discriminator
        loss_D_old = (
            F.mse_loss(self.disOld(real_old), torch.ones_like(self.disOld(real_old))) +
            F.mse_loss(self.disOld(self.generated_old_from_middle), torch.zeros_like(self.disOld(self.generated_old_from_middle)))
        ) * 0.5

        d_loss = loss_D_young + loss_D_middle + loss_D_old
        
        self.manual_backward(d_loss)
        d_opt.step()

        # Log losses
        self.log('Loss/Generator', g_loss)
        self.log('Loss/Discriminator', d_loss)

        return {'loss': g_loss + d_loss}

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(
            itertools.chain(
                self.genYoung2Middle.parameters(),
                self.genMiddle2Young.parameters(),
                self.genMiddle2Old.parameters(),
                self.genOld2Middle.parameters()
            ),
            lr=self.hparams['lr'],
            betas=(0.5, 0.999),
            weight_decay=self.hparams['weight_decay']
        )
        
        d_optim = torch.optim.Adam(
            itertools.chain(
                self.disYoung.parameters(),
                self.disMiddle.parameters(),
                self.disOld.parameters()
            ),
            lr=self.hparams['lr'],
            betas=(0.5, 0.999),
            weight_decay=self.hparams['weight_decay']
        )
        
        return [g_optim, d_optim]

# Example command to train
if __name__ == "__main__":
    model = AgingGAN(config)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model)
