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

        # Cache for generated images
        self.generated_A = None
        self.generated_B = None
        self.real_A = None
        self.real_B = None

    def forward(self, x):
        return self.genA2B(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch

        if optimizer_idx == 0:
            # Identity loss, GAN loss, Cycle loss, and Total loss as before
            # Define the generator loss here
            pass

        if optimizer_idx == 1:
            # Define the discriminator loss here
            pass

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

    def train_dataloader(self):
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        dataset_A = ImagetoImageDataset(
            domainA_dir=self.hparams['trainYoung_dir'], 
            domainB_dir=self.hparams['trainMiddle_dir'], 
            transform=train_transform
        )

        dataset_B = ImagetoImageDataset(
            domainA_dir=self.hparams['trainYoung_dir'], 
            domainB_dir=self.hparams['trainOld_dir'], 
            transform=train_transform
        )
        
        dataset_C = ImagetoImageDataset(
            domainA_dir=self.hparams['trainMiddle_dir'], 
            domainB_dir=self.hparams['trainOld_dir'], 
            transform=train_transform
        )

        datasets = torch.utils.data.ConcatDataset([dataset_A, dataset_B, dataset_C])

        return DataLoader(datasets, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'], shuffle=True)

if __name__ == "__main__":
    model = AgingGAN(config)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model)
