import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn
import gpytorch
import matplotlib.pyplot as plt

from dataset_def import HealthMNISTDatasetConv, PhysionetDataset
from utils import batch_predict, batch_predict_varying_T
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader

def MSE_test_GPapprox(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet, nnet_model,
                      covar_module0_y, covar_module1_y, covar_module0_m, covar_module1_m, likelihoods_y, likelihoods_m,
                      results_path, latent_dim_zy, latent_dim_zm, prediction_x,
                      prediction_mu_y, prediction_mu_m, zt_list_y, zt_list_m, P, T, id_covariate, varying_T=False,
                      save_file='result_error.csv', dataset_type='HealthMNIST'):
    """
    Function to compute Mean Squared Error of test set with GP approximationö

    """
    print("Running tests with a test set")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if type_nnet == 'conv':
        if dataset_type == 'HealthMNIST':
            test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                                  csv_file_label=csv_file_test_label,
                                                  mask_file=test_mask_file, root_dir=data_source_path,
                                                  transform=transforms.ToTensor())

    elif type_nnet == 'simple':
        if dataset_type == 'Physionet':
            test_dataset = PhysionetDataset(csv_file_data=csv_file_test_data,
                                                  csv_file_label=csv_file_test_label,
                                                  mask_file=test_mask_file, root_dir=data_source_path,
                                                  transform=transforms.ToTensor())

    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=4000, shuffle=False, num_workers=1)

    sum_loss_y = 0
    sum_loss_m = 0

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['data']
            data = data.double().to(device)
            mask = sample_batched['mask'].double()
            mask = mask.to(device)
            full_mask = torch.ones(mask.shape).to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate + 1:]), dim=1).double().to(device)

            test_x = label.type(torch.DoubleTensor).to(device)

            Zy_pred, cov_zy = batch_predict_varying_T(latent_dim_zy, covar_module0_y, covar_module1_y, likelihoods_y,
                                                      prediction_x, test_x, prediction_mu_y, zt_list_y, id_covariate,
                                                      eps=1e-6)
            Zm_pred, cov_zm = batch_predict_varying_T(latent_dim_zm, covar_module0_m, covar_module1_m, likelihoods_m,
                                                      prediction_x, test_x, prediction_mu_m, zt_list_m, id_covariate,
                                                      eps=1e-6)


            recon_y = nnet_model.decode_y(Zy_pred, Zm_pred)
            recon_m = nnet_model.decode_m(Zy_pred, Zm_pred)

            if dataset_type == 'HealthMNIST':
                [recon_loss_GP_y, nll_y] = nnet_model.loss_function_y(recon_y, data, full_mask)  # reconstruction loss for y
            elif dataset_type == 'Physionet':
                [recon_loss_GP_y, nll_y] = nnet_model.loss_function_y(recon_y, data, mask)  # reconstruction loss for y

            [recon_loss_GP_m, nll_m] = nnet_model.loss_function_m(recon_m, mask)  # reconstruction loss for m


            sum_loss_y += torch.sum(recon_loss_GP_y)
            sum_loss_m += torch.sum(recon_loss_GP_m)

            # pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy()])
            # np.savetxt(os.path.join(results_path, save_file), pred_results)

    return (sum_loss_y / len(test_dataset)).item(), (sum_loss_m / len(test_dataset)).item()


def impute_data(nnet_model, dataset, id_covariate, subjects_per_batch, num_dim):
    """
        Function to compute Mean Squared Error of imputed valies
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset,
                            batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(dataset, id_covariate),
                                                                    subjects_per_batch), num_workers=1)
    N = len(dataset)
    mse_sum = 0
    loss = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader):
            label_id = sample_batched['idx']
            full_y = sample_batched['data'].double().to(device)
            mask_curr = sample_batched['mask'].double().to(device).reshape(full_y.shape)
            y = torch.clone(full_y)
            y[~mask_curr.bool()] = 0
            mu_y, log_var_y = nnet_model.encode_y(y)
            mu_m, log_var_m = nnet_model.encode_m(mask_curr)
            zy = nnet_model.sample_latent(mu_y, log_var_y)
            zm = nnet_model.sample_latent(mu_m, log_var_m)
            recon_y = nnet_model.decode_y(zy, zm)
            mask_curr = ~mask_curr.bool()
            se = torch.mul(loss(recon_y.view(-1, num_dim), full_y.view(-1, num_dim)), mask_curr.view(-1, num_dim))
            mask_sum = torch.sum(mask_curr.view(-1, num_dim), dim=1)
            mask_sum[mask_sum == 0] = 1
            mse = torch.sum(se, dim=1) / mask_sum
            mse_sum += torch.sum(mse).item()

    return mse_sum / N