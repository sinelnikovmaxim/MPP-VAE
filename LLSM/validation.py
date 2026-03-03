from torch.utils.data import DataLoader
from utils import HensmanDataLoader, VaryingLengthBatchSampler, VaryingLengthSubjectSampler

import torch

from elbo_functions import deviance_upper_bound, elbo, minibatch_KLD_upper_bound_iter
from utils import batch_predict_varying_T


def validation_dubo(latent_dim, covar_module0, covar_module1, likelihood, train_xt, m, log_v, z, P, T, eps):
    """
    Efficient KL divergence using the variational mean and variance instead of a sample from the latent space (DUBO).
    See L-VAE supplementary material.

    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihood: GPyTorch likelihood model
    :param train_xt: auxiliary covariate information
    :param m: variational mean
    :param log_v: (log) variational variance
    :param z: inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param eps: jitter
    :return: KL divergence between variational distribution and additive GP prior (DUBO)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.exp(log_v)
    torch_dtype = torch.double
    x_st = torch.reshape(train_xt, [P, T, train_xt.shape[1]]).to(device)
    stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=1)
    K0xz = covar_module0(train_xt, z).evaluate().to(device)
    K0zz = (covar_module0(z, z).evaluate() + eps * torch.eye(z.shape[1], dtype=torch_dtype).to(device)).to(device)
    LK0zz = torch.cholesky(K0zz).to(device)
    iK0zz = torch.cholesky_solve(torch.eye(z.shape[1], dtype=torch_dtype).to(device), LK0zz).to(device)
    K0_st = covar_module0(stacked_x_st, stacked_x_st).evaluate().transpose(0, 1)
    B_st = (covar_module1(stacked_x_st, stacked_x_st).evaluate() + torch.eye(T, dtype=torch.double).to(
        device) * likelihood.noise_covar.noise.unsqueeze(dim=2)).transpose(0, 1)
    LB_st = torch.cholesky(B_st).to(device)
    iB_st = torch.cholesky_solve(torch.eye(T, dtype=torch_dtype).to(device), LB_st)

    dubo_sum = torch.tensor([0.0]).double().to(device)
    for i in range(latent_dim):
        m_st = torch.reshape(m[:, i], [P, T, 1]).to(device)
        v_st = torch.reshape(v[:, i], [P, T]).to(device)
        K0xz_st = torch.reshape(K0xz[i], [P, T, K0xz.shape[2]]).to(device)
        iB_K0xz = torch.matmul(iB_st[i], K0xz_st).to(device)
        K0zx_iB_K0xz = torch.matmul(torch.transpose(K0xz[i], 0, 1), torch.reshape(iB_K0xz, [P * T, K0xz.shape[2]])).to(
            device)
        W = K0zz[i] + K0zx_iB_K0xz
        W = (W + W.T) / 2
        LW = torch.cholesky(W).to(device)
        logDetK0zz = 2 * torch.sum(torch.log(torch.diagonal(LK0zz[i]))).to(device)
        logDetB = 2 * torch.sum(torch.log(torch.diagonal(LB_st[i], dim1=-2, dim2=-1))).to(device)
        logDetW = 2 * torch.sum(torch.log(torch.diagonal(LW))).to(device)
        logDetSigma = -logDetK0zz + logDetB + logDetW
        iB_m_st = torch.linalg.solve(B_st[i], m_st).to(device)
        qF1 = torch.sum(m_st * iB_m_st).to(device)
        p = torch.matmul(K0xz[i].T, torch.reshape(iB_m_st, [P * T])).to(device)
        qF2 = torch.sum(torch.linalg.solve_triangular(LW, p[:, None], upper=False)[0] ** 2).to(device)
        qF = qF1 - qF2
        tr = torch.sum(iB_st[i] * K0_st[i]) - torch.sum(K0zx_iB_K0xz * iK0zz[i])
        logDetD = torch.sum(torch.log(v[:, i])).to(device)
        tr_iB_D = torch.sum(torch.diagonal(iB_st[i], dim1=-2, dim2=-1) * v_st).to(device)
        D05_iB_K0xz = torch.reshape(iB_K0xz * torch.sqrt(v_st)[:, :, None], [P * T, K0xz.shape[2]])
        K0zx_iB_D_iB_K0zx = torch.matmul(torch.transpose(D05_iB_K0xz, 0, 1), D05_iB_K0xz).to(device)
        tr_iB_K0xz_iW_K0zx_iB_D = torch.sum(torch.diagonal(torch.cholesky_solve(K0zx_iB_D_iB_K0zx, LW))).to(device)
        tr_iSigma_D = tr_iB_D - tr_iB_K0xz_iW_K0zx_iB_D
        dubo = 0.5 * (tr_iSigma_D + qF - P * T + logDetSigma - logDetD + tr)
        dubo_sum = dubo_sum + dubo
    return dubo_sum


def validate(nnet_model, dataset, type_KL, latent_dim_zy, latent_dim_zm, covar_module0_y,
             covar_module1_y, covar_module0_m, covar_module1_m, likelihoods_y, likelihoods_m,
             zt_list_y, zt_list_m,  weight, id_covariate, loss_function, m_y, m_m, H_y, H_m, P, natural_gradient=True, eps=1e-6):
    """
    Obtain loss of validation set.

    :param nnet_model: neural network model
    :param dataset: dataset to use
    :param type_KL: type of KL divergence computation
    :param latent_dim_zy: number of latent dimensions of zy
    :param latent_dim_zm: number of latent dimensions of zm
    :param covar_module0_y: additive kernel of y (sum of cross-covariances) without id covariate
    :param covar_module1_y: additive kernel of y (sum of cross-covariances) with id covariate
    :param covar_module0_m: additive kernel of y (sum of cross-covariances) without id covariate
    :param covar_module1_m: additive kernel of y (sum of cross-covariances) with id covariate
    :param likelihoods_y: GPyTorch likelihood model of y
    :param likelihoods_m: GPyTorch likelihood model of m
    :param zt_list_y: list of inducing points of y
    :param zt_list_m: list of inducing points of m
    :param weight: value for the weight
    :param id_covariate: covariate number of the id
    :param loss_function: selected loss function
    :param eps: jitter
    :param m_y: variational mean of zy
    :param m_m: variational mean of zm
    :param H_y: variational variance for zy
    :param H_m: variational variance for zm
    :param P: number of validation instances
    :param natural_gradient: if computation of variational parameters by natural gradient
    :return: KL divergence between variational distribution
    """

    print("Testing the model with a validation set")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert (type_KL == 'GPapprox_closed' or type_KL == 'GPapprox')

    # set up Data Loader for training

    dataloader = HensmanDataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(dataset, id_covariate), 1), num_workers=1)

    N = len(dataset)

    recon_loss_sum = 0
    nll_loss_sum = 0
    nll_loss_tpp_sum = 0
    gp_loss_sum_y = 0
    gp_loss_sum_m = 0

    for batch_idx, sample_batched in enumerate(dataloader):
        data = sample_batched['data'].double().to(device)
        mask = sample_batched['mask'].double().to(device)
        val_x = sample_batched['label'].double().to(device)

        recon_batch_y, mu_y, log_var_y, recon_batch_m, mu_m, log_var_m = nnet_model(data, val_x, mask)

        [recon_loss_y, nll_y] = nnet_model.loss_function_y(recon_batch_y, data, mask)
        [recon_loss_m, nll_m] = nnet_model.loss_function_m(recon_batch_m, mask)
        recon_loss = torch.sum(recon_loss_y + recon_loss_m)
        nll = torch.sum(nll_y + nll_m)

        PSD_H_y = H_y if natural_gradient else torch.matmul(H_y, H_y.transpose(-1, -2))
        PSD_H_m = H_m if natural_gradient else torch.matmul(H_m, H_m.transpose(-1, -2))

        P_in_current_batch = torch.unique(val_x[:, id_covariate]).shape[0]

        kld_loss_y, grad_m_y, grad_H_y = minibatch_KLD_upper_bound_iter(covar_module0_y, covar_module1_y, likelihoods_y, latent_dim_zy, m_y, PSD_H_y, val_x, mu_y, log_var_y,zt_list_y, P, P_in_current_batch, N, natural_gradient, id_covariate, eps)

        kld_loss_m, grad_m_m, grad_H_m = minibatch_KLD_upper_bound_iter(covar_module0_m, covar_module1_m, likelihoods_m, latent_dim_zm, m_m, PSD_H_m, val_x, mu_m, log_var_m, zt_list_m, P,  P_in_current_batch, N, natural_gradient, id_covariate, eps)

        gp_loss_sum_y += kld_loss_y.item() / P
        gp_loss_sum_m += kld_loss_m.item() / P

        recon_loss_sum = recon_loss_sum + recon_loss.item()
        nll_loss_sum = nll_loss_sum + nll.item()


    if loss_function == 'mse':
        gp_loss_sum_y /= latent_dim_zy
        gp_loss_sum_m /= latent_dim_zm
        gp_loss_sum = gp_loss_sum_y + gp_loss_sum_m
        net_loss_sum = weight * (gp_loss_sum) + recon_loss_sum
    elif loss_function == 'nll':
        gp_loss_sum = gp_loss_sum_y + gp_loss_sum_m
        net_loss_sum = gp_loss_sum + nll_loss_sum

    # Do logging
    print('Validation set - Loss: %.3f  - GP loss: %.3f  - NLL loss: %.3f  - Recon Loss: %.3f' % (
        net_loss_sum, gp_loss_sum, nll_loss_sum, recon_loss_sum))

    return net_loss_sum


def compute_KL_loss(nnet_model, type_KL, num_samples, latent_dim, covar_module0, covar_module1, likelihoods,
                    zt_list, T, P, full_mu, full_log_var, full_labels, eps=1e-6):
    """
        Obtain KL divergence of validation set.

        :param nnet_model: neural network model
        :param type_nnet: type of encoder/decoder
        :param dataset: dataset to use
        :param type_KL: type of KL divergence computation
        :param num_samples: number of samples
        :param latent_dim: number of latent dimensions
        :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
        :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
        :param likelihoods: GPyTorch likelihood model
        :param zt_list: list of inducing points
        :param T: number of timepoints
        :param P: number of unique instances
        :param full_mu: posterior mean
        :param full_log_var: posterior log variance
        :param full_labels: auxiliary covariates
        :param eps: jitter
        :return: KL divergence between variational distribution
        """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gp_loss_sum = 0

    if isinstance(covar_module0, list):
        if type_KL == 'GPapprox':
            for sample in range(0, num_samples):
                Z = nnet_model.sample_latent(full_mu, full_log_var)
                for i in range(0, latent_dim):
                    Z_dim = Z[:, i]
                    gp_loss = -elbo(covar_module0[i], covar_module1[i], likelihoods[i], full_labels, Z_dim,
                                    zt_list[i].to(device), P, T, eps)
                    gp_loss_sum = gp_loss.item() + gp_loss_sum
            gp_loss_sum /= num_samples

        elif type_KL == 'GPapprox_closed':
            for i in range(0, latent_dim):
                mu_sliced = full_mu[:, i]
                log_var_sliced = full_log_var[:, i]
                gp_loss = deviance_upper_bound(covar_module0[i], covar_module1[i],
                                               likelihoods[i], full_labels,
                                               mu_sliced, log_var_sliced,
                                               zt_list[i].to(device), P,
                                               T, eps)
                gp_loss_sum = gp_loss.item() + gp_loss_sum
    else:
        if type_KL == 'GPapprox_closed':
            gp_loss = validation_dubo(latent_dim, covar_module0, covar_module1,
                                      likelihoods, full_labels,
                                      full_mu, full_log_var,
                                      zt_list, P, T, eps)
            gp_loss_sum = gp_loss.item()

    return gp_loss_sum