import os
from torch import nn, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from model.model import LitAutoEncoder



if __name__ == "__main__":



    # # Crea la directory se non esiste
    # if not os.path.exists("Autoencoders/checkpoints/lightning_logs"):
    #     os.makedirs("Autoencoders/checkpoints/lightning_logs")

    # define any number of nn.Modules (or use your current ones)
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    # init the autoencoder
    autoencoder = LitAutoEncoder(encoder, decoder)

    # setup data
    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = utils.data.DataLoader(dataset)


    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=1, default_root_dir="checkpoints/lightning_logs")
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


    # # load checkpoint
    # checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    # autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # choose your trained nn.Module
    encoder = autoencoder.encoder
    encoder.eval()

