import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from dataset import ImagetoImageDataset
from config import config
import itertools
class AgingGAN(pl.LightningModule):

    def __init__(self, hparams):
        super(AgingGAN, self).__init__()
        self.save_hyperparameters(hparams)
        self.genA2B = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.genB2A = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.disGA = Discriminator(hparams['ndf'])
        self.disGB = Discriminator(hparams['ndf'])

        # cache for generated images
        self.generated_A = None
        self.generated_B = None
        self.real_A = None
        self.real_B = None

    def forward(self, x):
        return self.genA2B(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch

        if optimizer_idx == 0:
            # Identity loss
            same_B = self.genA2B(real_B)
            loss_identity_B = F.l1_loss(same_B, real_B) * self.hparams['identity_weight']
            same_A = self.genB2A(real_A)
            loss_identity_A = F.l1_loss(same_A, real_A) * self.hparams['identity_weight']

            # GAN loss
            fake_B = self.genA2B(real_A)
            pred_fake = self.disGB(fake_B)
            loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * self.hparams['adv_weight']

            fake_A = self.genB2A(real_B)
            pred_fake = self.disGA(fake_A)
            loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * self.hparams['adv_weight']

            # Cycle loss
            recovered_A = self.genB2A(fake_B)
            loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * self.hparams['cycle_weight']

            recovered_B = self.genA2B(fake_A)
            loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * self.hparams['cycle_weight']

            # Total loss
            g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            output = {'loss': g_loss, 'log': {'Loss/Generator': g_loss}}
            self.log('Loss/Generator', g_loss)

            self.generated_B = fake_B
            self.generated_A = fake_A

            self.real_B = real_B
            self.real_A = real_A

            return output

        if optimizer_idx == 1:
            # Discriminator loss (as in your existing code)
            pred_real = self.disGA(real_A)
            loss_D_real = F.mse_loss(pred_real, torch.ones(pred_real.shape).type_as(pred_real))

            fake_A = self.generated_A
            pred_fake = self.disGA(fake_A.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake))

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            pred_real = self.disGB(real_B)
            loss_D_real = F.mse_loss(pred_real, torch.ones(pred_real.shape).type_as(pred_real))

            fake_B = self.generated_B
            pred_fake = self.disGB(fake_B.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake))

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            d_loss = loss_D_A + loss_D_B
            output = {'loss': d_loss, 'log': {'Loss/Discriminator': d_loss}}

            self.log('Loss/Discriminator', d_loss)

            return output

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                   lr=self.hparams['lr'], betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(),
                                                   self.disGB.parameters()),
                                   lr=self.hparams['lr'],
                                   betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        return [g_optim, d_optim], []

# Example command to train
if __name__ == "__main__":
    model = AgingGAN(config)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model)