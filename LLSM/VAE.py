import os, sys, torch, argparse
import pandas as pd
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import math

from dataset_def import HealthMNISTDatasetConv, PhysionetDataset
from parse_model_args import VAEArgs

class ConvVAE(nn.Module):
    """
    Encoders and decoders for variational autoencoder being able to encode and decode missing values as well with convolution and transposed convolution layers.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim_zy, latent_dim_zm, num_dim, covariate_dim, model_name="LVAE-MNAR-full", vy_init=1, vy_fixed=False, p_input=0.2, p=0.5):
        super(ConvVAE, self).__init__()

        self.latent_dim_zy = latent_dim_zy
        self.latent_dim_zm = latent_dim_zm
        self.num_dim = num_dim
        self.covariate_dim = covariate_dim
        self.model_name = model_name
        self.p_input = p_input
        self.p = p

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        # first convolution layer
        self.conv1y = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # encoder y
        self.conv1m = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # encoder m

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d_1_y = nn.Dropout2d(p=self.p)  # spatial dropout
        self.dropout2d_1_m = nn.Dropout2d(p=self.p)  # spatial dropout

        # second convolution layer
        self.conv2y = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # encoder y
        self.conv2m = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # encoder m
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d_2_y = nn.Dropout2d(p=self.p)
        self.dropout2d_2_m = nn.Dropout2d(p=self.p)

        self.fc1y = nn.Linear(32 * 9 * 9, 300) # encoder y
        self.fc1m = nn.Linear(32 * 9 * 9, 300) # encoder m
        self.dropout1_y = nn.Dropout(p=self.p)
        self.dropout1_m = nn.Dropout(p=self.p)
        self.fc21y = nn.Linear(300, 30) # encoder y
        self.fc21m = nn.Linear(300, 30) # encoder m
        self.dropout2_y = nn.Dropout(p=self.p)
        self.dropout2_m = nn.Dropout(p=self.p)
        self.fc211y = nn.Linear(30, self.latent_dim_zy) # encoder y
        self.fc221y = nn.Linear(30, self.latent_dim_zy) # encoder y
        self.fc211m = nn.Linear(30, self.latent_dim_zm) # encoder m
        self.fc221m = nn.Linear(30, self.latent_dim_zm) # encoder m

        # decoder network
        self.fc3concat_m_layer = nn.Linear(self.num_dim, 30) # layer needed for concatenation
        self.fc3y = nn.Linear(self.latent_dim_zy, 30) # decoder y
        self.fc3m1 = nn.Linear(self.latent_dim_zm, 30) # decoder m
        self.fc3m2 = nn.Linear(self.latent_dim_zy, 30) # decoder m
        self.fc3y1 = nn.Linear(self.latent_dim_zm, 30)  # decoder y
        self.fc3y2 = nn.Linear(self.latent_dim_zy, 30)  # decoder y
        self.dropout3_y = nn.Dropout(p=self.p)
        self.dropout3_m = nn.Dropout(p=self.p)
        self.fc31y = nn.Linear(30, 300) # decoder y
        self.fc31m = nn.Linear(30, 300) # decoder m
        self.dropout4_y = nn.Dropout(p=self.p)
        self.dropout4_m = nn.Dropout(p=self.p)
        self.fc4y = nn.Linear(300, 32 * 9 * 9) # decoder y
        self.fc4m = nn.Linear(300, 32 * 9 * 9) # decoder m

        self.dropout2d_3_y = nn.Dropout2d(p=self.p)
        self.dropout2d_3_m = nn.Dropout2d(p=self.p)
        # first transposed convolution
        self.deconv1y = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1) # decoder y
        self.deconv1m = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1) # decoder m

        self.dropout2d_4_y = nn.Dropout2d(p=self.p)
        self.dropout2d_4_m = nn.Dropout2d(p=self.p)
        # second transposed convolution
        self.deconv2y = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1) # decoder y
        self.deconv2m = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1) # decoder m

        self.dropout2d_5 = nn.Dropout2d(p=self.p)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode_y(self, y):
        """
        Encode the passed parameter y conditioned on auxiliary information x and mask m

        :param y: input data
        :return: variational mean and variance
        """
        # convolution
        zy = F.relu(self.conv1y(y))
        zy = self.dropout2d_1_y(self.pool1(zy))
        zy = F.relu(self.conv2y(zy))
        zy = self.dropout2d_2_y(self.pool2(zy))

        # MLP
        zy = zy.view(-1, 32 * 9 * 9)
        h1zy = self.dropout1_y(F.relu(self.fc1y(zy)))
        h2y = self.dropout2_y(F.relu(self.fc21y(h1zy)))
        return self.fc211y(h2y), self.fc221y(h2y)

    def encode_m(self, m):
        """
        Encode the passed parameter m conditioned on auxiliary information x and y
        :param m: input mask
        :return: variational mean and variance
        """
        # convolution
        m = m.reshape((m.shape[0], 1, int(np.sqrt(self.num_dim)), int(np.sqrt(self.num_dim))))  # reshape mask for conv layer
        zm = F.relu(self.conv1m(m))
        zm = self.dropout2d_1_m(self.pool1(zm))
        zm = F.relu(self.conv2m(zm))
        zm = self.dropout2d_2_m(self.pool2(zm))

        # MLP
        zm = zm.view(-1, 32 * 9 * 9)
        h1zm = self.dropout1_y(F.relu(self.fc1m(zm)))
        h2m = self.dropout2_y(F.relu(self.fc21m(h1zm)))
        return self.fc211m(h2m), self.fc221m(h2m)

    def decode_m(self, zy, zm):
        """
        Decode a latent sample zm
        :param zy:  latent sample zy
        :param zm: latent sample zm
        :return: reconstructed mask
        """

        # MLP

        m1 = self.dropout3_m(F.relu(self.fc3m1(zm)))
        m2 = self.dropout3_m(F.relu(self.fc3m2(zy)))
        m = self.dropout4_m(F.relu(self.fc31m(m1 + m2)))

        m = F.relu(self.fc4m(m))

        # transposed convolution
        m = self.dropout2d_3_m(m.view(-1, 32, 9, 9))
        m = self.dropout2d_4_m(F.relu(self.deconv1m(m)))
        p_m = torch.sigmoid(self.deconv2m(m)).view(-1, self.num_dim)

        return p_m

    def decode_y(self, zy, zm):
        """
        Decode a latent sample zy

        :param zy: latent sample zy
        :param zm: latent sample zm
        :return: reconstructed data
        """

        # MLP
        y2 = self.dropout3_y(F.relu(self.fc3y2(zy)))
        if self.model_name == "LVAE-MNAR-full":
            y1 = self.dropout3_y(F.relu(self.fc3y1(zm)))
            y = self.dropout4_y(F.relu(self.fc31y(y1 + y2)))
        else:
            y = self.dropout4_y(F.relu(self.fc31y(y2)))

        y = F.relu(self.fc4y(y))

        # transposed convolution
        y = self.dropout2d_3_y(y.view(-1, 32, 9, 9))
        y = self.dropout2d_4_y(F.relu(self.deconv1y(y)))
        return torch.sigmoid(self.deconv2y(y))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y, x, m):
        mu_y, log_var_y = self.encode_y(y)
        mu_m, log_var_m = self.encode_m(m)
        zy = self.sample_latent(mu_y, log_var_y)
        zm = self.sample_latent(mu_m, log_var_m)
        p_m = self.decode_m(zy, zm)
        return self.decode_y(zy, zm), mu_y, log_var_y, p_m, mu_m, log_var_m

    def loss_function_y(self, recon_y, y, mask):
        """
        Reconstruction loss for data

        :param recon_y: reconstruction of latent sample
        :param y:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """

        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_y.view(-1, self.num_dim), y.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(log_vy))
        nll += torch.mul(0.5 * (np.log(2 * math.pi) + log_vy), mask.view(-1, self.num_dim))
        return mse, torch.sum(nll, dim=1)

    def loss_function_m(self, recon_m, m):
        """
        Reconstruction loss for mask

        :param recon_m: reconstruction of latent sample
        :param m: true mask
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """

        jitter = 10**-10
        loss = nn.MSELoss(reduction='none')
        se = loss(recon_m.view(-1, self.num_dim), m.view(-1, self.num_dim))
        mse = torch.sum(se, dim=1) / self.num_dim

        m = m.view(-1, self.num_dim)

        nll = - torch.sum(m * torch.log(recon_m + jitter) + (1 - m) * torch.log(1 - recon_m + jitter), dim=1)
        return mse, nll


class SimpleVAE(nn.Module):
    """
    Encoder and decoder for variational autoencoder with simple multi-layered perceptrons.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim_zy, latent_dim_zm, num_dim, covariate_dim, model_name="LVAE-MNAR-full", vy_init=1, vy_fixed=False):

        super(SimpleVAE, self).__init__()

        self.latent_dim_zy = latent_dim_zy
        self.latent_dim_zm = latent_dim_zm
        self.num_dim = num_dim
        self.covariate_dim = covariate_dim
        self.model_name = model_name

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        self.fc1y = nn.Linear(num_dim, 300)
        self.fc1m = nn.Linear(num_dim, 300)
        self.fc21y = nn.Linear(300, 30)
        self.fc21m = nn.Linear(300, 30)
        self.fc211y = nn.Linear(30, latent_dim_zy)
        self.fc221y = nn.Linear(30, latent_dim_zy)
        self.fc211m = nn.Linear(30, latent_dim_zm)
        self.fc221m = nn.Linear(30, latent_dim_zm)

        # decoder network
        self.fc3m1 = nn.Linear(latent_dim_zm, 30)
        self.fc3m2 = nn.Linear(latent_dim_zy, 30)
        self.fc31m = nn.Linear(30, 300)
        self.fc4m = nn.Linear(300, num_dim)

        self.fc3y1 = nn.Linear(latent_dim_zm, 30)
        self.fc3y2 = nn.Linear(latent_dim_zy, 30)
        self.fc31y = nn.Linear(30, 300)
        self.fc4y = nn.Linear(300, num_dim)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode_y(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """
        h1 = F.relu(self.fc1y(x))
        h2 = F.relu(self.fc21y(h1))
        return self.fc211y(h2), self.fc221y(h2)

    def encode_m(self, m):
        """
        Encode the passed parameter

        :param m: input mask
        :return: variational mean and variance
        """
        h1 = F.relu(self.fc1m(m))
        h2 = F.relu(self.fc21m(h1))
        return self.fc211m(h2), self.fc221m(h2)

    def decode_m(self, zy, zm):
        """
        Decode a latent sample zm
        :param zy:  latent sample zy
        :param zm: latent sample zm
        :return: reconstructed mask
        """
        # MLP
        m1 = F.relu(self.fc3m1(zm))
        m2 = F.relu(self.fc3m2(zy))
        m = F.relu(self.fc31m(m1 + m2))

        m = self.fc4m(m)
        p_m = torch.sigmoid(m)
        return p_m

    def decode_y(self, zy, zm):
        """
        Decode a latent sample zy

        :param zy: latent sample zy
        :param zm: latent sample zm
        :return: reconstructed data
        """
        # MLP
        y2 = F.relu(self.fc3y2(zy))
        if self.model_name == "LVAE-MNAR-full":
            y1 = F.relu(self.fc3y1(zm))
            y = F.relu(self.fc31y(y1 + y2))
        else:
            y = F.relu(self.fc31y(y2))

        y = self.fc4y(y)
        return y

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, y, x, m):
        mu_y, log_var_y = self.encode_y(y)
        mu_m, log_var_m = self.encode_m(m)
        zy = self.sample_latent(mu_y, log_var_y)
        zm = self.sample_latent(mu_m, log_var_m)
        p_m = self.decode_m(zy, zm)
        return self.decode_y(zy, zm), mu_y, log_var_y, p_m, mu_m, log_var_m

    def loss_function_y(self, recon_y, y, mask):
        """
        Reconstruction loss for data

        :param recon_y: reconstruction of latent sample
        :param y:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """

        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        #print(recon_y.shape)
        #print(y.view(-1, self.num_dim).shape)

        se = torch.mul(loss(recon_y.view(-1, self.num_dim), y.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(log_vy))
        nll += torch.mul(0.5 * (np.log(2 * math.pi) + log_vy), mask.view(-1, self.num_dim))
        return mse, torch.sum(nll, dim=1)

    def loss_function_m(self, recon_m, m):
        """
        Reconstruction loss for mask

        :param recon_m: reconstruction of latent sample
        :param m: true mask
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """

        jitter = 10**-10
        loss = nn.MSELoss(reduction='none')
        se = loss(recon_m.view(-1, self.num_dim), m.view(-1, self.num_dim))
        mse = torch.sum(se, dim=1) / self.num_dim

        m = m.view(-1, self.num_dim)

        nll = - torch.sum(m * torch.log(recon_m + jitter) + (1 - m) * torch.log(1 - recon_m + jitter), dim=1)
        return mse, nll


def pretrain_for_LVAE(csv_file_data, csv_file_label, mask_file, loss_function, type_nnet, dataset_type, data_source_path,
                      num_dim, vy_init, vy_fixed,latent_dim_zy, latent_dim_zm, id_covariate, save_path, model_name):
    """
      This is used for pre-training.
      """

    assert loss_function == 'mse' or loss_function == 'nll', ("Unknown loss function " + loss_function)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    # set up dataset
    if type_nnet == 'conv':
        if dataset_type == 'HealthMNIST':
            dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                             mask_file=mask_file, root_dir=data_source_path,
                                             transform=transforms.ToTensor())

        elif dataset_type == 'Physionet':
            dataset = PhysionetDataset(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                       mask_file=mask_file, root_dir=data_source_path,
                                       transform=transforms.ToTensor())

    print('Length of dataset:  {}'.format(len(dataset)))
    Q = len(dataset[0]['label'])

    # set up Data Loader
    dataloader = DataLoader(dataset, min(len(dataset), 256), shuffle=True, num_workers=1)

    vy = torch.Tensor(np.ones(num_dim) * vy_init)

    # set up model and send to GPU if available
    if type_nnet == 'conv':
        print('Using convolutional neural network')
        nnet_model = ConvVAE(latent_dim_zy, latent_dim_zm, num_dim, Q, model_name, vy, vy_fixed).to(device)
    elif type_nnet == 'simple':
        print('Using standard MLP')
        nnet_model = SimpleVAE(latent_dim_zy, latent_dim_zm, num_dim, Q, model_name, vy, vy_fixed).to(device)

    optimiser = torch.optim.Adam(nnet_model.parameters(), lr=1e-3)

    net_train_loss = np.empty((0, 1))

    epochs = 50

    for epoch in range(1, epochs + 1):

        # start training VAE
        nnet_model.train()
        train_loss = 0
        recon_loss_sum = 0
        recon_loss_sum_y = 0
        recon_loss_sum_m = 0
        nll_loss = 0
        nll_loss_y = 0
        nll_loss_m = 0
        kld_loss = 0
        kld_loss_y = 0
        kld_loss_m = 0

        for batch_idx, sample_batched in enumerate(dataloader):
            data = sample_batched['data']
            data = data.float().to(device)  # send to GPU
            mask = sample_batched['mask']
            mask = mask.float().to(device)
            label = sample_batched['label'].to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate + 1:]), dim=1)

            optimiser.zero_grad()  # clear gradients

            recon_batch_y, mu_y, log_var_y, recon_batch_m, mu_m, log_var_m = nnet_model(data, label, mask)

            [recon_loss_y, nll_y] = nnet_model.loss_function_y(recon_batch_y, data, mask)  # reconstruction of data
            [recon_loss_m, nll_m] = nnet_model.loss_function_m(recon_batch_m, mask)  # reconstruction of mask
            KLD_y = -0.5 * torch.sum(1 + log_var_y - mu_y.pow(2) - log_var_y.exp(), dim=1)
            KLD_m = -0.5 * torch.sum(1 + log_var_m - mu_m.pow(2) - log_var_m.exp(), dim=1)
            nll = nll_y + nll_m
            recon_loss = recon_loss_y + recon_loss_m
            KLD = KLD_y + KLD_m
            if loss_function == 'nll':
                loss = torch.sum(nll + KLD)
            elif loss_function == 'mse':
                loss = torch.sum(recon_loss + KLD)

            loss.backward()  # compute gradients
            train_loss += loss.item()
            recon_loss_sum += recon_loss.sum().item()
            recon_loss_sum_y += recon_loss_y.sum().item()
            recon_loss_sum_m += recon_loss_m.sum().item()
            nll_loss += nll.sum().item()
            nll_loss_y += nll_y.sum().item()
            nll_loss_m += nll_m.sum().item()
            kld_loss += KLD.sum().item()
            kld_loss_y += KLD_y.sum().item()
            kld_loss_m += KLD_m.sum().item()

            optimiser.step()  # update parameters

        print(
            '====> Epoch: {} - Average loss: {:.4f}  - KLD loss: {:.3f}  - NLL loss: {:.3f}  - Recon loss: {:.3f} - KLD_mask loss: {:.3f}  - NLL_mask loss: {:.3f}  - Recon_mask loss: {:.3f}'
            ''.format(epoch, train_loss, kld_loss, nll_loss, recon_loss_sum, kld_loss_m, nll_loss_m, recon_loss_sum_m))

        net_train_loss = np.append(net_train_loss, train_loss)

    print(nnet_model.vy)
    torch.save(nnet_model.state_dict(), os.path.join(save_path, 'model_params_vae.pth'))
