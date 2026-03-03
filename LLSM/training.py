from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

import numpy as np
import torch
import os

from elbo_functions import deviance_upper_bound, elbo, KL_closed, minibatch_KLD_upper_bound, minibatch_KLD_upper_bound_iter
from model_test import MSE_test_GPapprox
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader
from validation import validate

def hensman_training(nnet_model, type_nnet, epochs, dataset, optimiser, scheduler, type_KL, num_samples, latent_dim_zy,latent_dim_zm, covar_module0_y,
                     covar_module1_y, covar_module0_m,covar_module1_m, likelihoods_y, likelihoods_m, m_y, H_y, m_m, H_m, zt_list_y, zt_list_m, P, T, varying_T, Q, weight, id_covariate, loss_function,
                     natural_gradient=False, natural_gradient_lr=0.01, subjects_per_batch=20, memory_dbg=False,
                     eps=1e-6, results_path=None, validation_dataset=None, generation_dataset=None,
                     prediction_dataset=None, gp_model_y=None, gp_model_m=None, csv_file_validation_data=None, csv_file_validation_label=None,
                     validation_mask_file=None, data_source_path=None):

    """
    Perform training with minibatching and Stochastic Variational Inference [Hensman et. al, 2013]. See L-VAE supplementary
    materials

    :param nnet_model: encoder/decoder neural network model 
    :param type_nnet: type of encoder/decoder
    :param epochs: numner of epochs
    :param dataset: dataset to use in training
    :param optimiser: optimiser to be used
    :param type_KL: type of KL divergenve computation to use
    :param num_samples: number of samples to use
    :param latent_dim_zy: number of latent dimensions for zy
    :param latent_dim_zm: number of latent dimensions for zm
    :param covar_module0_y: additive kernel (sum of cross-covariances) without id covariate for zy
    :param covar_module1_y: additive kernel (sum of cross-covariances) with id covariate for zy
    :param covar_module0_m: additive kernel (sum of cross-covariances) without id covariate for zm
    :param covar_module1_m: additive kernel (sum of cross-covariances) with id covariate for zm
    :param likelihoods_y: GPyTorch likelihood model for zy
    :param likelihoods_m: GPyTorch likelihood model for zm
    :param m_y: variational mean for zy
    :param H_y: variational variance for zy
    :param m_m: variational mean for zm
    :param H_m: variational variance for zm
    :param zt_list_y: list of inducing points for zy
    :param zt_list_m: list of inducing points for zm
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param Q: number of covariates
    :param weight: value for the weight
    :param id_covariate: covariate number of the id
    :param loss_function: selected loss function
    :param natural_gradient: use of natural gradients
    :param natural_gradient_lr: natural gradients learning rate
    :param subject_per_batch; number of subjects per batch (vectorisation)
    :param memory_dbg: enable debugging
    :param eps: jitter
    :param results_path: path to results
    :param validation_dataset: dataset for vaildation set
    :param generation_dataset: dataset to help with sample image generation
    :param prediction_dataset; dataset with subjects for prediction
    :param gp_model_y: GPyTorch gp model for zy
    :param gp_model_m: GPyTorch gp model for zm
    :param csv_file_test_data: path to test data
    :param csv_file_test_label: path to test label
    :param test_mask_file: path to test mask
    :param data_source_path: path to data source

    :return trained models and resulting losses

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(dataset)
    assert type_KL == 'GPapprox_closed'

    if varying_T:
        n_batches = (P + subjects_per_batch - 1)//subjects_per_batch
        dataloader = HensmanDataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(dataset, id_covariate), subjects_per_batch), num_workers=1)
    else:
        batch_size = subjects_per_batch*T
        n_batches = (P*T + batch_size - 1)//(batch_size)
        dataloader = HensmanDataLoader(dataset, batch_sampler=BatchSampler(SubjectSampler(dataset, P, T), batch_size, drop_last=False), num_workers=1)

    net_train_loss_arr = np.empty((0, 1))
    recon_loss_arr = np.empty((0, 1))
    nll_loss_arr = np.empty((0, 1))
    kld_loss_arr = np.empty((0, 1))
    penalty_term_arr = np.empty((0, 1))
    best_val_pred_mse = np.inf
    best_val_recon_y = np.inf
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0
        recon_loss_sum_y = 0
        recon_loss_sum_m = 0
        nll_loss_sum_y = 0
        nll_loss_sum_m = 0
        kld_loss_sum_y = 0
        kld_loss_sum_m = 0
        iid_kld_sum = 0
        for batch_idx, sample_batched in enumerate(dataloader):
            optimiser.zero_grad()
            nnet_model.train()
            covar_module0_y.train()
            covar_module1_y.train()
            covar_module0_m.train()
            covar_module1_m.train()
            likelihoods_y.train()
            likelihoods_m.train()
            indices = sample_batched['idx']
            data = sample_batched['data'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]

            covariates = torch.cat((train_x[:, :id_covariate], train_x[:, id_covariate+1:]), dim=1)

            recon_batch_y, mu_y, log_var_y, recon_batch_m, mu_m, log_var_m = nnet_model(data,train_x, mask)
            [recon_loss_y, nll_y] = nnet_model.loss_function_y(recon_batch_y, data, mask)
            [recon_loss_m, nll_m] = nnet_model.loss_function_m(recon_batch_m, mask)
            recon_loss = torch.sum(recon_loss_y + recon_loss_m)
            nll_loss = torch.sum(nll_y + nll_m)

            PSD_H_y = H_y if natural_gradient else torch.matmul(H_y, H_y.transpose(-1, -2))
            PSD_H_m = H_m if natural_gradient else torch.matmul(H_m, H_m.transpose(-1, -2))

            if varying_T:
                P_in_current_batch = torch.unique(train_x[:, id_covariate]).shape[0]
                kld_loss_y, grad_m_y, grad_H_y = minibatch_KLD_upper_bound_iter(covar_module0_y, covar_module1_y, likelihoods_y, latent_dim_zy, m_y, PSD_H_y, train_x, mu_y, log_var_y, zt_list_y, P, P_in_current_batch, N, natural_gradient, id_covariate, eps)
                kld_loss_m, grad_m_m, grad_H_m = minibatch_KLD_upper_bound_iter(covar_module0_m, covar_module1_m, likelihoods_m, latent_dim_zm, m_m, PSD_H_m, train_x, mu_m, log_var_m, zt_list_m, P, P_in_current_batch, N, natural_gradient, id_covariate, eps)
            else:
                P_in_current_batch = N_batch // T
                kld_loss_y, grad_m_y, grad_H_y = minibatch_KLD_upper_bound(covar_module0_y, covar_module1_y, likelihoods_y, latent_dim_zy, m_y, PSD_H_y, train_x, mu_y, log_var_y, zt_list_y, P, P_in_current_batch, T, natural_gradient, eps)
                kld_loss_m, grad_m_m, grad_H_m = minibatch_KLD_upper_bound(covar_module0_m, covar_module1_m, likelihoods_m, latent_dim_zm, m_m, PSD_H_m,train_x, mu_m, log_var_m, zt_list_m, P,P_in_current_batch, T, natural_gradient, eps)

            recon_loss = recon_loss * P/P_in_current_batch
            nll_loss = nll_loss * P/P_in_current_batch
            recon_loss_y = recon_loss_y * P/P_in_current_batch
            recon_loss_m = recon_loss_m * P/P_in_current_batch
            nll_y = nll_y * P/P_in_current_batch
            nll_m = nll_m * P/P_in_current_batch

            recon_loss_sum_y += torch.sum(recon_loss_y) / n_batches
            recon_loss_sum_m += torch.sum(recon_loss_m) / n_batches
            nll_loss_sum_y += torch.sum(nll_y) / n_batches
            nll_loss_sum_m += torch.sum(nll_m) / n_batches

            if loss_function == 'nll':
                net_loss = nll_loss + kld_loss_y + kld_loss_m
            elif loss_function == 'mse':
                kld_loss_y = kld_loss_y / latent_dim_zy
                kld_loss_m = kld_loss_m / latent_dim_zm
                net_loss = recon_loss + (weight) * (kld_loss_y + kld_loss_m)

            net_loss.backward()
            optimiser.step()
            scheduler.step()

            if natural_gradient:
                LH_y = torch.cholesky(H_y)
                iH_y = torch.cholesky_solve(torch.eye(H_y.shape[-1], dtype=torch.double).to(device), LH_y)
                iH_new_y = iH_y + natural_gradient_lr*(grad_H_y + grad_H_y.transpose(-1,-2))
                LiH_new_y = torch.cholesky(iH_new_y)
                H_y = torch.cholesky_solve(torch.eye(H_y.shape[-1], dtype=torch.double).to(device), LiH_new_y).detach()
                m_y = torch.matmul(H_y, torch.matmul(iH_y, m_y) - natural_gradient_lr*(grad_m_y - 2*torch.matmul(grad_H_y, m_y))).detach()

                LH_m = torch.cholesky(H_m)
                iH_m = torch.cholesky_solve(torch.eye(H_m.shape[-1], dtype=torch.double).to(device), LH_m)
                iH_new_m = iH_m + natural_gradient_lr * (grad_H_m + grad_H_m.transpose(-1, -2))
                LiH_new_m = torch.cholesky(iH_new_m)
                H_m = torch.cholesky_solve(torch.eye(H_m.shape[-1], dtype=torch.double).to(device), LiH_new_m).detach()
                m_m = torch.matmul(H_m, torch.matmul(iH_m, m_m) - natural_gradient_lr * (grad_m_m - 2 * torch.matmul(grad_H_m, m_m))).detach()

            net_loss_sum += net_loss.item() / n_batches 
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += (kld_loss_y.item() + kld_loss_m.item()) / n_batches
            kld_loss_sum_y += kld_loss_y.item() / n_batches
            kld_loss_sum_m += kld_loss_m.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f - GP_mask loss: %.3f  - NLL_mask Loss: %.3f  - Recon Loss_mask: %.3f' % (
            epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum, kld_loss_sum_m, nll_loss_sum_m, recon_loss_sum_m), flush=True)
        penalty_term_arr = np.append(penalty_term_arr, 0.0)
        net_train_loss_arr = np.append(net_train_loss_arr,  net_loss_sum)
        recon_loss_arr = np.append(recon_loss_arr, recon_loss_sum)
        nll_loss_arr = np.append(nll_loss_arr, nll_loss_sum)
        kld_loss_arr = np.append(kld_loss_arr, kld_loss_sum)

        if (not epoch % 25) and epoch != epochs:
            with torch.no_grad():
                nnet_model.eval()
                covar_module0_y.eval()
                covar_module1_y.eval()
                covar_module0_m.eval()
                covar_module1_m.eval()
                if validation_dataset is not None:

                    val_pred_mse = validate(nnet_model, validation_dataset, type_KL, latent_dim_zy, latent_dim_zm, covar_module0_y, covar_module1_y, covar_module0_m, covar_module1_m, likelihoods_y, likelihoods_m,
                                 zt_list_y, zt_list_m, weight, id_covariate, loss_function, m_y, m_m, H_y, H_m, P, natural_gradient=True, eps=1e-6)

                    if val_pred_mse < best_val_pred_mse:
                        best_val_pred_mse = val_pred_mse
                        best_epoch = epoch

                        print('Saving better model')
                        try:
                            torch.save(nnet_model.state_dict(), os.path.join(results_path, 'nnet_model_best.pth'))
                            torch.save(gp_model_y.state_dict(), os.path.join(results_path, 'gp_model_y_best.pth'))
                            torch.save(gp_model_m.state_dict(), os.path.join(results_path, 'gp_model_m_best.pth'))
                            torch.save(zt_list_y, os.path.join(results_path, 'zt_list_y_best.pth'))
                            torch.save(zt_list_m, os.path.join(results_path, 'zt_list_m_best.pth'))
                            torch.save(m_y, os.path.join(results_path, 'm_y_best.pth'))
                            torch.save(H_y, os.path.join(results_path, 'H_y_best.pth'))
                            torch.save(m_m, os.path.join(results_path, 'm_m_best.pth'))
                            torch.save(H_m, os.path.join(results_path, 'H_m_best.pth'))

                        except e:
                            print(e)
                            print('Saving intermediate model failed!')
                            pass

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    return penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, kld_loss_arr, m_y, H_y, m_m, H_m, best_epoch
