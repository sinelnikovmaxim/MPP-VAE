import math
import sys
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import gpytorch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from timeit import default_timer as timer

from GP_def import ExactGPModel
from VAE import ConvVAE, SimpleVAE
from dataset_def import HealthMNISTDatasetConv, PhysionetDataset
from elbo_functions import elbo, KL_closed, deviance_upper_bound
from kernel_gen import generate_kernel, generate_kernel_approx, generate_kernel_batched
from model_test import MSE_test_GPapprox, impute_data
from parse_model_args import ModelArgs
from training import hensman_training
from validation import validate
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader
from VAE import pretrain_for_LVAE
from torch.utils.data import Dataset
from Gtilde_data import __G_lookup_table
import os
import logging

class Gtilde_lookup(torch.autograd.Function):
    "Class for computing G function from Lloyd 2015 with corresponding gradient"

    @staticmethod
    def forward(ctx,z):
        Gs, dGs = _Gtilde_lookup(z)
        ctx.save_for_backward(dGs)
        return Gs

    @staticmethod
    def backward(ctx, grad_output):
        dGs, = ctx.saved_tensors
        return dGs * grad_output


def _Gtilde_lookup(z):
    LOG = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = z.detach().clone()  # copy and do not overwrite input array

    is_scalar = z.ndim == 0
    z = torch.atleast_1d(z).to(device)  # otherwise array indexing does not work

    # Transform z -> -z, bit easier to think about!
    z = -z

    if torch.any(z >= 10 ** (len(__G_lookup_table) - 1)):
        n_overflow = torch.sum(z >= 10 ** (len(__G_lookup_table) - 1))
        n_lesszero = torch.sum(z < 0)
        LOG.warning(
            "Gtilde: z out of range: %d greater-than-zeros, %d out-of-ranges (of %d)",
            n_lesszero,
            n_overflow,
            len(z),
        )

    if torch.any(z < 0):  # remember, original zs were assumed to be negative
        raise ValueError("Gtilde: invalid z: we require z <= 0")

    REAL_MIN = torch.finfo(z.dtype).tiny  # smallest positive number
    z[z == 0] = REAL_MIN

    Gs = torch.zeros_like(z).to(device)
    dGs = torch.zeros_like(z).to(device)

    out_of_range = z >= 10 ** (len(__G_lookup_table) - 1)
    Gs[out_of_range] = dGs[out_of_range] = torch.nan

    lower = 0
    upper = 1
    binWidth = 0.001
    # For each region
    k = 0
    for Gi in __G_lookup_table:
        # Work out which z lie in this region
        zR = (lower <= z) & (z < upper)

        # fmt: off
        # Work out which are the upper (zj) and lower (zi) intervals for each z
        # and the fraction across the bin the point is (zr)
        zi = torch.floor(z[zR] / binWidth).type(torch.int)  # lower
        zj = torch.ceil(z[zR] / binWidth).type(torch.int)  # upper
        zr = torch.remainder(z[zR] / binWidth, 1)  # remainder
        # If the remainder is zero, increment the second point
        zj[zr == 0.0] += 1
        # Compute the gradient for each point
        dGs[zR] = torch.from_numpy(np.array(Gi[zj.cpu().detach()])).to(device) - torch.from_numpy(
            np.array(Gi[zi.cpu().detach()])).to(device)

        # Interpolate using the gradient to find the function value
        Gs[zR] = torch.from_numpy(np.array(Gi[zi.cpu().detach()])).to(device) + dGs[zR] * zr
        # Correct dG for bin width
        dGs[zR] = dGs[zR] / binWidth
        # fmt: on

        # Adjust binWidth, upper and lower boundaries for next region
        binWidth = binWidth * 10
        lower = upper
        upper = upper * 10

        k += 1
    # Correct dG for z -> -z transformation
    dGs = -dGs

    if is_scalar:  # undo atleast_1d
        Gs = Gs[0]
        dGs = dGs[0]
    return Gs, dGs

class TPP():
    """
    Temporal point process class
    """
    def __init__(self, D):
        super(TPP, self).__init__()
        self.D = D

    def compute_variational_params(self, m, L, tril_indices, zt_list, covar_module0, prev_timestamps):
        """
        Computes mean and marginal variance of a variational distribution of z_lambda
        :param m: variational mean of inducing points
        :param L: cholesky factor of variational covarice matrix of inducing points given as a vector
        :param tril_indices: indices of cholesky elements
        :param zt_list: inducing points locations
        :param covar_module0: covariance module
        :param prev_timestamps: differences with D previous timestamps which are used as input for covariance
        """
        eps = 1e-6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        M = m.shape[0]
        L_matrix = torch.zeros((M, M), dtype=torch.double).to(device)
        L_matrix[tril_indices[0], tril_indices[1]] = torch.squeeze(L, 1)
        Ktz = covar_module0(prev_timestamps, zt_list).evaluate()[0]
        Kzt = torch.transpose(Ktz, 0, 1)
        Kzz = covar_module0(zt_list, zt_list).evaluate()[0] + eps * torch.eye(zt_list.shape[0], dtype=torch.double).to(
            device)
        mean = torch.matmul(Ktz, torch.linalg.solve(Kzz, m))
        Ktt = (covar_module0(prev_timestamps, prev_timestamps).evaluate()[0] + eps * torch.eye(prev_timestamps.shape[0],
                                                                                     dtype=torch.double).to(device))
        Kzz_Kzt_inv = torch.linalg.solve(Kzz, Kzt)
        Kzz_S_inv = torch.linalg.solve(Kzz, torch.matmul(L_matrix, torch.transpose(L_matrix, 0, 1)))
        Sigma_t = Ktt - torch.matmul(Ktz, Kzz_Kzt_inv) + torch.matmul(torch.matmul(Ktz, Kzz_S_inv), Kzz_Kzt_inv)
        return torch.squeeze(mean, 1), Sigma_t + eps * torch.eye(Sigma_t.shape[0], dtype=torch.double).to(device)


    def integrate_log_fn_sqr(self, mean, var):
        """
        Computes expectation of log squared
        :param mean: variational mean of z_lambda
        :param var: variational marginal variance of z_lambda
        """
        z = -0.5 * torch.square(mean) / var
        C = 0.57721566  # Euler-Mascheroni constant
        G = Gtilde_lookup.apply(z)
        return -G + torch.log(0.5 * var) - C


    def integrate_lambda(self, covar_module0, m, L, tril_indices, beta, zt_list, prev_timestamps, mask_covar, mask_zt, t_n, t_n_1, t_d_n,
                         zt_list_reduced,subject_ids, lengthscales, gammas):
        """
        Computes expectation of the integral over T
        :param covar_module0: covariance module
        :param m: variational mean of q(u)
        :param L: variation cholesky factor of q(u)
        :param tril_indices: indices of elements for which cholesky factor is non-zero
        :param beta: trainable parameter that controls prior mean of GP
        :param zt_list: inducing positions of z_lambda
        :param prev_timestamps: differences with D previous timestamps which are used as input for covariance
        :param mask_covar: mask that represents which previous timestamps are NAN
        :param mask_zt: mask that represents which inducing positions are NAN
        :param t_n: timestamps
        :param t_n_1: timestamps one step back
        :param t_d_n: previous D timestamps
        :param zt_list_reduced: inducing points locations without mask part
        :param subject_ids: subject ids
        :param lengthscales: tensor of lengthscales
        :param gammas: tensor of gammas
        """

        eps = 1e-6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Kzz = covar_module0(zt_list, zt_list).evaluate()[0] + eps * torch.eye(zt_list.shape[0], dtype=torch.double).to(device)
        Phi = self.compute_phi(mask_covar, mask_zt, t_n, t_n_1, t_d_n, zt_list_reduced, lengthscales, gammas)
        Psi = self.compute_psi(mask_covar, mask_zt, t_n, t_n_1, t_d_n, zt_list_reduced, lengthscales, gammas)
        Kzz_inv_m = torch.linalg.solve(Kzz, m)
        expectation_z = torch.squeeze(torch.matmul(Phi, Kzz_inv_m),1)
        Psi_Kzz_inv_m = torch.matmul(Psi, Kzz_inv_m)
        expectation_z_squared = torch.matmul(torch.transpose(m,0,1), torch.linalg.solve(Kzz, Psi_Kzz_inv_m))

        total_time, beta_indices = self.compute_T(prev_timestamps, t_n, subject_ids)
        total_indicator_intervals = self.compute_indicator_integrals(prev_timestamps, t_n, subject_ids)
        variance_z = torch.sum(gammas * total_indicator_intervals)
        M = zt_list.shape[0]
        L_matrix = torch.zeros((M, M), dtype=torch.double).to(device)
        L_matrix[tril_indices[0], tril_indices[1]] = torch.squeeze(L, 1)
        S = torch.matmul(L_matrix, torch.transpose(L_matrix, 0, 1))

        Kzz_inv_Psi = torch.linalg.solve(Kzz, Psi)
        Kzz_inv_S = torch.linalg.solve(Kzz, S)
        trace_1 = torch.sum(torch.diagonal(Kzz_inv_Psi))
        trace_2 = torch.sum(torch.diagonal(torch.matmul(Kzz_inv_S, Kzz_inv_Psi)))
        variance_z += -trace_1 + trace_2

        L_t = expectation_z_squared + variance_z + torch.sum(2 * beta * expectation_z) + torch.sum(beta[beta_indices] ** 2 * total_time)
        return L_t

    def compute_T(self, prev_timestamps, t_n, subject_ids):
        """
        Computes integral of identity function over T
        :param prev_timestamps: differences with D previous timestamps which are used as input for covariance
        :param t_n: timestamps
        :param subject_ids: subject ids
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unique_ids = torch.unique(subject_ids)
        time_intervals = torch.zeros(unique_ids.shape[0], dtype=torch.double).to(device)
        beta_indices = []
        k = 0
        for id in unique_ids:
            indices = subject_ids == id
            beta_indices.append(indices.nonzero()[0])
            prev_event = t_n[indices][0] - prev_timestamps[indices][0,0]
            curr_id_times = t_n[indices]
            time_intervals[k] = curr_id_times[curr_id_times.shape[0] - 1] - prev_event
            k += 1

        return time_intervals, torch.tensor(beta_indices).type(dtype=torch.int64).to(device)


    def compute_indicator_integrals(self, prev_timestamps, t_n, subject_ids):
        """
        Computes integral of identity funtion over T taking into account missing timestamps
        :param prev_timestamps: differences with D previous timestamps which are used as input for covariance
        :param t_n: timestamps
        :param subject_ids: subject ids
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_indicator_intervals = torch.zeros(self.D, dtype=torch.double).to(device)
        unique_ids = torch.unique(subject_ids)
        for id in unique_ids:
            indices = subject_ids == id
            prev_event = t_n[indices][0] - prev_timestamps[indices][0, 0]
            curr_id_times = t_n[indices]
            T = curr_id_times[curr_id_times.shape[0] - 1]
            min_between_D_and_sequence_len = min(curr_id_times.shape[0], self.D)
            events_start = torch.cat((torch.tensor([prev_event], dtype=torch.double).to(device),
                                      curr_id_times[:min_between_D_and_sequence_len - 1]))

            total_indicator_intervals[:min_between_D_and_sequence_len] += T - events_start

        return total_indicator_intervals

    def collect_kernel_parameters(self, covar_module0):
        """
        Collects lengthscales and gammas into separate tensors
        :param covar_module0: covariance module
        """
        lenghtscales = torch.cat((covar_module0.kernels[0].base_kernel.kernels[0].lengthscale,))
        gammas = torch.cat((covar_module0.kernels[0].outputscale,))
        for d in range(1, self.D):
            lenghtscales = torch.cat((lenghtscales, covar_module0.kernels[d].base_kernel.kernels[0].lengthscale))
            gammas = torch.cat((gammas, covar_module0.kernels[d].outputscale))

        lenghtscales = torch.reshape(lenghtscales, (1, 1, lenghtscales.shape[0]))
        gammas = torch.reshape(gammas, (1, 1, gammas.shape[0]))
        return lenghtscales, gammas

    def compute_inputs_for_phi_psi(self, zt_list, prev_timestamps, t_n):
        """
        Computes inputs necessary for phi and psi functions
        :param zt_list: inducing points locations
        :param prev_timestamps: differences with D previous timestamps which are used as input for covariance
        :param t_n: timestamps
        """
        mask_covar = prev_timestamps[:, self.D:]
        mask_zt = zt_list[:, self.D:]
        zt_list_reduced = zt_list[:, :self.D]
        t_d_n = torch.unsqueeze(t_n,1) - prev_timestamps[:, :self.D]
        t_n_1 = t_d_n[:,0]
        return mask_covar, mask_zt, t_n_1, t_d_n, zt_list_reduced

    def compute_phi(self,mask_covar, mask_zt, t_n, t_n_1, t_d_n, zt_list_reduced, lengthscales, gammas):
        """
        Computes phi function
        :param mask_covar: mask that represents which previous timestamps are NAN
        :param mask_zt: mask that represents which inducing positions are NAN
        :param t_n: timestamps
        :param t_n_1: timestamps one step back
        :param t_d_n: previous D timestamps
        :param zt_list_reduced: inducing points locations without mask part
        :param lengthscales: tensor of lengthscales
        :param gamma: tensor of gammas
        """

        t_n = torch.unsqueeze(t_n, 1)
        t_n_1 = torch.unsqueeze(t_n_1, 1)
        mask_covar = mask_covar.unsqueeze(1)
        mask_zt = mask_zt.unsqueeze(0)
        diff1 = (t_n - t_d_n).unsqueeze(1)
        diff2 = (t_n_1 - t_d_n).unsqueeze(1)
        zt_list_reduced = zt_list_reduced.unsqueeze(0)

        erf1 = torch.erf((diff1 - zt_list_reduced) / torch.sqrt(2 * lengthscales ** 2))
        erf2 = torch.erf((diff2 - zt_list_reduced) / torch.sqrt(2 * lengthscales ** 2))
        phi = torch.sum(mask_covar * mask_zt * gammas * torch.sqrt(torch.pi * lengthscales ** 2 / 2) * (erf1 - erf2), dim=2)
        return phi

    def compute_psi(self, mask_covar, mask_zt, t_n, t_n_1, t_d_n, zt_list_reduced, lengthscales, gammas):
        """
        Computes psi function
        :param mask_covar: mask that represents which previous timestamps are NAN
        :param mask_zt: mask that represents which inducing positions are NAN
        :param t_n: timestamps
        :param t_n_1: timestamps one step back
        :param t_d_n: previous D timestamps
        :param zt_list_reduced: inducing points locations without mask part
        :param covar_module0: covariance module
        :param lengthscales: tensor of lengthscales
        :param gammas: tensor of gammas
        """

        t_n = torch.unsqueeze(t_n, 1)
        t_n_1 = torch.unsqueeze(t_n_1, 1)

        zt_list_stroke = zt_list_reduced
        zt_list_reduced = zt_list_reduced.unsqueeze(1)
        zt_list_stroke = zt_list_stroke.unsqueeze(0)

        mask_zt_stroke = mask_zt.unsqueeze(0)
        mask_zt_stroke = mask_zt_stroke.unsqueeze(2)
        mask_zt = mask_zt.unsqueeze(1)
        mask_zt = mask_zt.unsqueeze(3)

        diff1 = (t_n - t_d_n).unsqueeze(1)
        diff2 = (t_n_1 - t_d_n).unsqueeze(1)

        l_1 = torch.unsqueeze(lengthscales, 3)
        l_2 = torch.unsqueeze(lengthscales, 2)

        gamma_1 = torch.unsqueeze(gammas, 3)
        gamma_2 = torch.unsqueeze(gammas, 2)

        diff_z = zt_list_reduced.unsqueeze(3) - zt_list_stroke.unsqueeze(2)
        diff_t = t_d_n.unsqueeze(2) - t_d_n.unsqueeze(1)

        diff_t = diff_t.unsqueeze(1).unsqueeze(2)
        diff_z = diff_z.unsqueeze(0)

        exp_part = torch.exp(-(diff_t + diff_z) ** 2 / (2 * (l_1 ** 2 + l_2 ** 2)))

        zt_list_reduced = zt_list_reduced.squeeze(1).unsqueeze(0)

        erf1_numerator = (l_2 ** 2 * (diff1 - zt_list_reduced).unsqueeze(3)).unsqueeze(2) + (
                    l_1 ** 2 * (diff1 - zt_list_stroke).unsqueeze(2)).unsqueeze(1)
        erf2_numerator = (l_2 ** 2 * (diff2 - zt_list_reduced).unsqueeze(3)).unsqueeze(2) + (
                    l_1 ** 2 * (diff2 - zt_list_stroke).unsqueeze(2)).unsqueeze(1)

        denominator = torch.sqrt(2 * l_1 ** 2 * l_2 ** 2 * (l_1 ** 2 + l_2 ** 2))
        erf_part = torch.erf(erf1_numerator / denominator) - torch.erf(erf2_numerator / denominator)

        indicators_zt = mask_zt * mask_zt_stroke
        indicators_zt = indicators_zt.unsqueeze(0)
        indicators_covar = mask_covar.unsqueeze(2) * mask_covar.unsqueeze(1)
        indicators_covar = indicators_covar.unsqueeze(1).unsqueeze(2)

        indicators = indicators_zt * indicators_covar

        psi = indicators * gamma_1 * gamma_2 * torch.sqrt(
            torch.pi * l_1 ** 2 * l_2 ** 2 / (2 * (l_1 ** 2 + l_2 ** 2))) * exp_part * erf_part

        psi = torch.sum(torch.sum(torch.sum(psi, dim=4), dim=3), dim=0)

        return psi

    def compute_KL_tpp(self, covar_module0, zt_list, m, L, tril_indices):
        """
        Computes KL divergences
        :param covar_module0: covariance module
        :param zt_list: inducing points locations
        :param m: variational mean of q(u)
        :param L: variational cholesky factor of q(u)
        :param tril_indices: indices corresponding to non-zero elements of cholesky factor
        """
        eps = 1e-6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        M = m.shape[0]
        L_matrix = torch.zeros((M, M), dtype=torch.double).to(device)
        L_matrix[tril_indices[0], tril_indices[1]] = torch.squeeze(L, 1)
        Kzz = covar_module0(zt_list, zt_list).evaluate()[0] + eps * torch.eye(zt_list.shape[0], dtype=torch.double).to(device)
        LKzz = torch.linalg.cholesky(Kzz)
        chol_prod = torch.linalg.solve(LKzz, L_matrix)
        first_term = torch.sum(chol_prod ** 2)
        second_term = torch.linalg.norm(torch.linalg.solve(LKzz, -m)) ** 2
        k = M
        third_term = 2 * (torch.sum(
            torch.log(torch.abs((torch.linalg.diagonal(LKzz)))) - (
                torch.log(torch.abs(torch.linalg.diagonal(L_matrix))))))

        return (1 / 2) * (first_term + second_term - k + third_term)

    def sample_intensity(self, m, L, tril_indices, beta, zt_list, covar_module0, prev_timestamps):
        """
        Samples z from the variational distribution
        :param m: variational mean of q(u)
        :param L: variational cholesky factor of q(u)
        :param tril_indices: indices corresponding to non-zero elements of cholesky factor
        :param beta: trainable parameter that controls prior mean of GP
        :param zt_list: inducing points locations
        :param covar_module0: covariance module
        :param prev_timestamps: differences with D previous timestamps which are used as input for covariance
        """
        mean, Sigma = self.compute_variational_params(m, L, tril_indices, zt_list,
                                                     covar_module0, prev_timestamps)
        eps_norm = torch.randn_like(mean)
        cholesky_Sigma = torch.linalg.cholesky(Sigma)
        sample_intensity = torch.unsqueeze((torch.matmul(cholesky_Sigma, eps_norm) + mean + beta) ** 2, 1)
        return sample_intensity


    def loss_tpp(self, prev_timestamps, t_n, subject_ids, zt_list, covar_module0, m,L, tril_indices, beta):
        """
        Computes loss of temporal point process
        :param prev_timestamps: differences with D previous timestamps which are used as input for covariance
        :param t_n: timestamps
        :param subject_ids: subject ids
        :param zt_list: inducing points locations
        :param covar_module0: covariance module
        :param m: variational mean of q(u)
        :param L: variational cholesky factor of q(u)
        :param tril_indices: indices corresponding to non-zero elements of cholesky factor
        :param beta: trainable parameter that controls prior mean of GP
        """
        mean, Sigma = self.compute_variational_params(m, L, tril_indices, zt_list, covar_module0, prev_timestamps)
        var = torch.diagonal(Sigma)
        mask_covar, mask_zt, t_n_1, t_d_n, zt_list_reduced = self.compute_inputs_for_phi_psi(zt_list, prev_timestamps, t_n)
        L_n = torch.sum(self.integrate_log_fn_sqr(mean + beta, var))
        lengthscales, gammas = self.collect_kernel_parameters(covar_module0)
        L_t = self.integrate_lambda(covar_module0, m, L, tril_indices, beta, zt_list, prev_timestamps, mask_covar, mask_zt, t_n,  t_n_1, t_d_n, zt_list_reduced, subject_ids, lengthscales, gammas)
        nll_loss = -(L_n - L_t)
        kld_loss = self.compute_KL_tpp(covar_module0, zt_list, m, L, tril_indices)
        return nll_loss, kld_loss


    def validation_TPP(self, dataloader_val, covar_module0, zt_list, m, L, tril_indices, beta, id_covariate, disease_covariate):
        """
        Computes validation error
        :param dataloader_val: Mini-batching for validation
        :param covar_module0: covariance module
        :param zt_list: inducing points locations
        :param m: variational mean of q(u)
        :param L: variational cholesky factor of q(u)
        :param tril_indices: indices corresponding to non-zero elements of cholesky factor
        :param beta: trainable parameter that controls prior mean of GP
        :param id_covariate: covariate number of the id
        :param disease_covariate: covariate of the presence of disease
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nll_loss_sum = 0
        covar_module0.train()
        for batch_idx, sample_batched in enumerate(dataloader_val):
            val_x = sample_batched['label'].double().to(device)
            subject_ids = val_x[:, id_covariate]
            t_n = val_x[:, 0]
            disease_present = val_x[:, disease_covariate].type(dtype=torch.int64)
            one_hot_representation = F.one_hot(disease_present, num_classes=2).type(dtype=torch.double).to(device)
            beta_disease = torch.matmul(one_hot_representation, beta)
            prev_timestamps = sample_batched['prev_timestamps'].double().to(device)
            nll_loss, kld_loss = self.loss_tpp(prev_timestamps, t_n, subject_ids, zt_list, covar_module0, m, L,
                                               tril_indices, beta_disease)
            nll_loss_sum += nll_loss


        loss = nll_loss_sum + kld_loss
        print('Val Loss: %.3f - Val nll: %.3f -  Val KL Loss: %.3f' % (loss, nll_loss_sum, kld_loss), flush=True)
        return loss.item()

    def pretrain_TPP(self, dataloader_train, dataloader_val, covar_module0, gp_model, zt_list, m, L, tril_indices, beta, id_covariate, disease_covariate, epochs, optimiser, P_train, n_batches_train, save_path):
        """
        Pretrains temporal point process
        :param dataloader_train: Mini-batching for train
        :param dataloader_val: Mini-batching for validation
        :param covar_module0: covariance module
        :param gp_model: GP model
        :param zt_list: inducing points locations
        :param m: variational mean of q(u)
        :param L: variational cholesky factor of q(u)
        :param tril_indices: indices corresponding to non-zero elements of cholesky factor
        :param beta: trainable parameter that controls prior mean of GP
        :param id_covariate: covariate number of the id
        :param disease_covariate: covariate of the presence of disease
        :param epochs: number of epochs
        :param optimiser: optimiser type
        :param P_train: number of instances for training
        :param n_batches_train: number of batches for training
        :param save_path: path to save results
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_val_loss = torch.inf
        epochs = 11
        for epoch in range(1, epochs + 1):
            nll_loss_sum = 0
            kll_loss_sum = 0
            loss_sum = 0
            for batch_idx, sample_batched in enumerate(dataloader_train):
                optimiser.zero_grad()
                covar_module0.train()
                train_x = sample_batched['label'].double().to(device)
                subject_ids = train_x[:, id_covariate]
                P_in_current_batch = torch.unique(subject_ids).shape[0]
                prev_timestamps = sample_batched['prev_timestamps'].double().to(device)
                t_n = train_x[:, 0]
                disease_present = train_x[:, disease_covariate].type(dtype=torch.int64)
                one_hot_representation = F.one_hot(disease_present, num_classes=2).type(dtype=torch.double).to(device)
                beta_disease = torch.matmul(one_hot_representation, beta)
                nll_loss, kld_loss = self.loss_tpp(prev_timestamps, t_n, subject_ids, zt_list, covar_module0, m, L, tril_indices, beta_disease)
                loss = (P_train / P_in_current_batch) * nll_loss + kld_loss
                loss_sum += loss / n_batches_train
                kll_loss_sum += kld_loss / n_batches_train
                nll_loss_sum += nll_loss / n_batches_train
                loss.backward()
                optimiser.step()

            print('Iter %d/%d - Loss: %.3f - KL Loss: %.3f' % (epoch, epochs, loss_sum, kll_loss_sum), flush=True)

            if (not epoch % 5) and epoch != epochs:
                with torch.no_grad():
                    val_loss = self.validation_TPP(dataloader_val, covar_module0, zt_list, m, L, tril_indices, beta, id_covariate, disease_covariate)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(gp_model.state_dict(), os.path.join(save_path, 'gp_model_lambda_best.pth'))
                        torch.save(zt_list, os.path.join(save_path, 'zt_list_lambda_best.pth'))
                        torch.save(m, os.path.join(save_path, 'm_lambda_best.pth'))
                        torch.save(L, os.path.join(save_path, 'L_lambda_best.pth'))
                        torch.save(beta, os.path.join(save_path, 'beta_best.pth'))
                        print("Saving better model")


def kernel_indices(D):
    sqexp_kernel = range(D)
    covariate_missing_val = []
    for d in range(D):
        covariate_missing_val.append({"covariate": d, "mask": D + d})

    return sqexp_kernel, covariate_missing_val