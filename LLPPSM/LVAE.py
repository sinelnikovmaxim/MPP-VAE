import os
import sys

from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gpytorch
from torch import nn
import matplotlib.pyplot as plt

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
from TPP import TPP

eps = 1e-6

def train_val_split_health_mnist(csv_file_data_full, csv_file_data, csv_file_label, mask_file, csv_file_train_data, csv_file_train_label,
                train_mask_file, csv_file_validation_data, csv_file_validation_label,validation_mask_file,
                csv_file_prediction_data, csv_file_prediction_label, prediction_mask_file,
                csv_file_train_prediction_data, csv_file_train_prediction_label, train_prediction_mask_file,
                csv_file_imputation_data, root_dir, num_val_instances, num_predict_instances):
 
    full_data = pd.read_csv(os.path.join(root_dir, csv_file_data_full), header=None)
    data = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
    labels = pd.read_csv(os.path.join(root_dir, csv_file_label))
    mask = pd.read_csv(os.path.join(root_dir, mask_file), header=None)

    instances = pd.unique(labels["subject"])
    num_instances = instances.shape[0]

    assert num_val_instances < num_instances

    assert num_predict_instances < num_instances - num_val_instances

    num_train_instances = num_instances - num_val_instances
    train_instances = np.random.choice(instances, num_train_instances, replace=False)
    mask_instances = np.ones(num_instances, dtype=bool)
    mask_instances[train_instances] = 0
    val_instances = instances[mask_instances]
    predict_instances = np.random.choice(train_instances, num_predict_instances, replace=False)

    idx = np.arange(0, data.shape[0])
    train_idx = idx[labels["subject"].isin(train_instances) & ~labels["subject"].isin(predict_instances)]
    train_prediction_idx = np.copy(train_idx)

    predict_idx = np.zeros(num_predict_instances * 15, dtype=int)
    k = 0
    for el in predict_instances:
        curr_idx = labels[labels["subject"] == el].first_valid_index()
        train_prediction_idx = np.append(train_prediction_idx, np.arange(curr_idx, curr_idx + 5))
        predict_idx[k:k + 15] = np.arange(curr_idx + 5, curr_idx + 20, dtype=int)
        k += 15

    val_idx = idx[labels["subject"].isin(val_instances)]

    train_data = data.iloc[train_idx, :]
    train_labels = labels.iloc[train_idx, :]
    train_mask = mask.iloc[train_idx, :]

    train_predict_data = data.iloc[train_prediction_idx, :]
    train_predict_labels = labels.iloc[train_prediction_idx, :]
    train_predict_mask = mask.iloc[train_prediction_idx, :]

    val_data = data.iloc[val_idx,:]
    val_labels = labels.iloc[val_idx, :]
    val_mask = mask.iloc[val_idx, :]

    predict_data = full_data.iloc[predict_idx, :]
    predict_labels = labels.iloc[predict_idx, :]
    predict_mask = mask.iloc[predict_idx, :]

    imputation_data = full_data.iloc[predict_idx, :]

    train_data.to_csv(os.path.join(root_dir, csv_file_train_data), index=False, header=False)
    train_labels.to_csv(os.path.join(root_dir, csv_file_train_label), index=False)
    train_mask.to_csv(os.path.join(root_dir, train_mask_file), index=False, header=False)

    train_predict_data.to_csv(os.path.join(root_dir, csv_file_train_prediction_data), index=False, header=False)
    train_predict_labels.to_csv(os.path.join(root_dir, csv_file_train_prediction_label), index=False)
    train_predict_mask.to_csv(os.path.join(root_dir, train_prediction_mask_file), index=False, header=False)

    val_data.to_csv(os.path.join(root_dir, csv_file_validation_data), index=False, header=False)
    val_labels.to_csv(os.path.join(root_dir, csv_file_validation_label), index=False)
    val_mask.to_csv(os.path.join(root_dir, validation_mask_file), index=False, header=False)

    predict_data.to_csv(os.path.join(root_dir, csv_file_prediction_data), index=False, header=False)
    predict_labels.to_csv(os.path.join(root_dir, csv_file_prediction_label), index=False)
    predict_mask.to_csv(os.path.join(root_dir, prediction_mask_file), index=False, header=False)

    imputation_data.to_csv(os.path.join(root_dir, csv_file_imputation_data), index=False, header=False)

def tpp_kernel_def(D):
    cat_kernel_tpp = []
    bin_kernel_tpp = []
    cat_int_kernel_tpp = []
    bin_int_kernel_tpp = []
    sqexp_kernel_tpp = range(D)
    covariate_missing_val_tpp = []
    for d in range(D):
        covariate_missing_val_tpp.append({"covariate": d, "mask":D + d})

    return sqexp_kernel_tpp, covariate_missing_val_tpp, cat_kernel_tpp, bin_kernel_tpp, cat_int_kernel_tpp, bin_int_kernel_tpp


if __name__ == "__main__":
    """
    Root file for running L-VAE.

    Run command: python LVAE.py --f=path_to_config-file.txt 
    """

    # create parser and set variables

    opt = ModelArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    model_name = "LLPPSM"

    assert loss_function == 'mse' or loss_function == 'nll', ("Unknown loss function " + loss_function)

    print(torch.version.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    pred_y_arr = []
    pred_m_arr = []
    imputation_arr = []
    accuracy_arr = []

    number_of_repetitions = 1

    if dataset_type == "HealthMNIST" and split_data:
        train_val_split_health_mnist(csv_file_data_full, csv_file_data, csv_file_label, mask_file, csv_file_train_data,
                                     csv_file_train_label, train_mask_file, csv_file_validation_data,
                                     csv_file_validation_label,
                                     validation_mask_file, csv_file_prediction_data, csv_file_prediction_label,
                                     prediction_mask_file, csv_file_train_prediction_data,
                                     csv_file_train_prediction_label, train_prediction_mask_file,
                                     csv_file_imputation_data, data_source_path, num_val_instances,
                                     num_predict_instances)

    for l in range(number_of_repetitions):

        pretrain_for_LVAE(csv_file_train_data, csv_file_train_label, train_mask_file, "nll", type_nnet, dataset_type,
                          data_source_path,
                          num_dim, vy_init, vy_fixed, latent_dim_zy, latent_dim_zm, id_covariate, save_path,
                          model_name)  # pretrain model by classical VAE

        # set up dataset
        if type_nnet == 'conv':
            if dataset_type == 'HealthMNIST':
                dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_train_data, csv_file_label=csv_file_train_label,
                                                 mask_file=train_mask_file, root_dir=data_source_path,
                                                 transform=transforms.ToTensor())

        elif type_nnet == 'simple':
            if dataset_type == 'Physionet':
                dataset = PhysionetDataset(csv_file_data=csv_file_train_data, csv_file_label=csv_file_train_label,
                                           mask_file=train_mask_file, root_dir=data_source_path,
                                           transform=transforms.ToTensor())

        # Set up prediction dataset
        if run_tests:
            if dataset_type == 'HealthMNIST' and type_nnet == 'conv':
                prediction_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_prediction_data,
                                                            csv_file_label=csv_file_prediction_label,
                                                            mask_file=prediction_mask_file, root_dir=data_source_path,
                                                            transform=transforms.ToTensor())
                print('Length of prediction dataset:  {}'.format(len(prediction_dataset)))

            elif dataset_type == 'Physionet':
                prediction_dataset = PhysionetDataset(csv_file_data=csv_file_prediction_data,
                                                      csv_file_label=csv_file_prediction_label,
                                                      mask_file=prediction_mask_file, root_dir=data_source_path,
                                                      transform=transforms.ToTensor())

                print('Length of prediction dataset:  {}'.format(len(prediction_dataset)))

            if type_nnet == 'conv':
                if dataset_type == 'HealthMNIST':
                    train_prediction_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_train_prediction_data,
                                                     csv_file_label=csv_file_train_prediction_label,
                                                     mask_file=train_prediction_mask_file, root_dir=data_source_path,
                                                     transform=transforms.ToTensor())

            elif type_nnet == 'simple':
                if dataset_type == 'Physionet':
                    train_prediction_dataset = PhysionetDataset(csv_file_data=csv_file_train_prediction_data,
                                                     csv_file_label=csv_file_train_prediction_label,
                                                     mask_file=train_prediction_mask_file, root_dir=data_source_path,
                                                     transform=transforms.ToTensor())
        else:
            prediction_dataset = None
            train_prediction_dataset = None

        # Set up validation dataset
        if run_validation:
            if dataset_type == 'HealthMNIST':
                validation_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_validation_data,
                                                            csv_file_label=csv_file_validation_label,
                                                            mask_file=validation_mask_file, root_dir=data_source_path,
                                                            transform=transforms.ToTensor())
                print('Length of validation dataset:  {}'.format(len(validation_dataset)))

            elif dataset_type == 'Physionet':
                validation_dataset = PhysionetDataset(csv_file_data=csv_file_validation_data,
                                                      csv_file_label=csv_file_validation_label,
                                                      mask_file=validation_mask_file, root_dir=data_source_path,
                                                      transform=transforms.ToTensor())
                print('Length of validation dataset:  {}'.format(len(validation_dataset)))

        else:
            validation_dataset = None

        print('Length of dataset:  {}'.format(len(dataset)))
        N = len(dataset)

        if not N:
            print("ERROR: Dataset is empty")
            exit(1)

        Q = len(dataset[0]['label'])

        # set up model and send to GPU if available
        if type_nnet == 'conv':
            print('Using convolutional neural network')
            nnet_model = ConvVAE(latent_dim_zy, latent_dim_zm, num_dim, Q, model_name, vy_init, vy_fixed,
                                 p_input=dropout_input, p=dropout).to(device)

        elif type_nnet == 'simple':
            print('Using standard MLP')
            nnet_model = SimpleVAE(latent_dim_zy, latent_dim_zm, num_dim, Q, model_name, vy_init, vy_fixed).to(device)

        # Load pre-trained encoder/decoder parameters if present
        try:
            nnet_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')))
            print('Loaded pre-trained values.')
        except:
            print('Did not load pre-trained values.')

        nnet_model = nnet_model.double().to(device)

        tpp = TPP(D)
        latent_dim_z_lambda = 1

        # set up Data Loader for GP initialisation
        setup_dataloader = DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=1)

        # Get values for GP initialisation:
        Z_y = torch.zeros(N, latent_dim_zy, dtype=torch.double).to(device)
        Z_m = torch.zeros(N, latent_dim_zm, dtype=torch.double).to(device)
        Z_lambda = torch.randn(N, latent_dim_z_lambda, dtype=torch.double).to(device)
        train_x = torch.zeros(N, Q, dtype=torch.double).to(device)
        prev_timestamps = torch.zeros(N, D * 2, dtype=torch.double).to(device)
        nnet_model.eval()
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(setup_dataloader):
                # no mini-batching. Instead get a batch of dataset size
                mask = sample_batched['mask'].double().to(device)
                label_id = sample_batched['idx']
                train_x[label_id] = sample_batched['label'].double().to(device)
                data = sample_batched['data'].double().to(device)
                covariates = torch.cat((train_x[label_id, :id_covariate], train_x[label_id, id_covariate + 1:]), dim=1)
                prev_timestamps[label_id] = sample_batched['prev_timestamps'].double().to(device)

                mu_y, log_var_y = nnet_model.encode_y(data)
                mu_m, log_var_m = nnet_model.encode_m(mask)
                Z_y[label_id] = nnet_model.sample_latent(mu_y, log_var_y)
                Z_m[label_id] = nnet_model.sample_latent(mu_m, log_var_m)

        sqexp_kernel_tpp, covariate_missing_val_tpp, cat_kernel_tpp, bin_kernel_tpp, cat_int_kernel_tpp, bin_int_kernel_tpp = tpp_kernel_def(D)

        covar_module_y = []
        covar_module0_y = []
        covar_module1_y = []
        covar_module_m = []
        covar_module0_m = []
        covar_module1_m = []
        covar_module0_lambda = []
        zt_list_y = []
        gp_models_y = []
        zt_list_m = []
        gp_models_m = []
        zt_list_lambda = []
        gp_models_lambda = []
        adam_param_list = []

        likelihoods_y = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim_zy]),
                                                                noise_constraint=gpytorch.constraints.GreaterThan(
                                                                    1.000E-08)).to(device)

        likelihoods_m = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim_zm]),
                                                                noise_constraint=gpytorch.constraints.GreaterThan(
                                                                    1.000E-08)).to(device)

        likelihoods_lambda = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim_z_lambda]),
                                                                     noise_constraint=gpytorch.constraints.GreaterThan(
                                                                         1.000E-08)).to(device)

        if constrain_scales:
            likelihoods_y.noise = 1
            likelihoods_y.raw_noise.requires_grad = False
            likelihoods_m.noise = 1
            likelihoods_m.raw_noise.requires_grad = False
            likelihoods_lambda.noise = 1
            likelihoods_lambda.raw_noise.requires_grad = False

        covar_module0_y, covar_module1_y = generate_kernel_batched(latent_dim_zy,
                                                                   cat_kernel, bin_kernel, sqexp_kernel,
                                                                   cat_int_kernel, bin_int_kernel,
                                                                   covariate_missing_val, id_covariate, lengthscale=0.1)

        covar_module0_m, covar_module1_m = generate_kernel_batched(latent_dim_zm,
                                                                   cat_kernel, bin_kernel, sqexp_kernel,
                                                                   cat_int_kernel, bin_int_kernel,
                                                                   covariate_missing_val, id_covariate, lengthscale=0.1)

        covar_module0_lambda, _ = generate_kernel_batched(latent_dim_z_lambda,
                                                          cat_kernel_tpp, bin_kernel_tpp, sqexp_kernel_tpp,
                                                          cat_int_kernel_tpp, bin_int_kernel_tpp,
                                                          covariate_missing_val_tpp, id_covariate, D, lengthscale=2.5)

        gp_model_y = ExactGPModel(torch.cat((train_x, torch.zeros((train_x.shape[0], 1)).to(device)), dim=1),
                                  Z_y.type(torch.DoubleTensor), likelihoods_y,
                                  covar_module0_y + covar_module1_y).to(device)

        gp_model_m = ExactGPModel(torch.cat((train_x, torch.zeros((train_x.shape[0], 1)).to(device)), dim=1),
                                  Z_m.type(torch.DoubleTensor), likelihoods_m,
                                  covar_module0_m + covar_module1_m).to(device)

        gp_model_lambda = ExactGPModel(prev_timestamps, Z_lambda.type(torch.DoubleTensor), likelihoods_lambda,
                                       covar_module0_lambda).to(device)

        # initialise inducing points for y and m
        zt_list_y = torch.zeros(latent_dim_zy, M_y, Q, dtype=torch.double).to(device)
        zt_list_m = torch.zeros(latent_dim_zm, M_m, Q, dtype=torch.double).to(device)
        zt_list_lambda = torch.zeros(M_lambda, D * 2, dtype=torch.double).to(device)
        n = train_x.shape[0]

        inducing_points_indices = np.random.choice(N, M_y, replace=False)

        # inducing points for y
        for i in range(latent_dim_zy):
            zt_list_y[i] = train_x[inducing_points_indices].clone().detach()

        # inducing points for m
        for i in range(latent_dim_zm):
            zt_list_m[i] = train_x[inducing_points_indices].clone().detach()

        zt_list_lambda = prev_timestamps[inducing_points_indices].clone().detach()

        adam_param_list.append({'params': covar_module0_lambda.parameters()})
        covar_module0_lambda.train().double()
        likelihoods_lambda.train().double()

        setup_dataloader = HensmanDataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(
            VaryingLengthSubjectSampler(dataset, id_covariate), 1), num_workers=1)

        total_time_across_individuals = {"sick": 0, "healthy": 0}
        total_number_of_events = {"sick": 0, "healthy": 0}
        # Get values for GP initialisation:
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(setup_dataloader):
                curr_x = sample_batched['label'].double().to(device)
                curr_prev = sample_batched['prev_timestamps'].double().to(device)
                curr_len = curr_x.shape[0]
                curr_time = curr_x[curr_len - 1, 0] - (curr_x[0, 0] - curr_prev[0, 0])
                if curr_x[0, disease_covariate]:
                    total_time_across_individuals["sick"] += curr_time
                    total_number_of_events["sick"] += curr_len
                else:
                    total_time_across_individuals["healthy"] += curr_time
                    total_number_of_events["healthy"] += curr_len

        beta = torch.sqrt(
            torch.tensor([total_number_of_events["healthy"] / (total_time_across_individuals["healthy"] + 1),
                          total_number_of_events["sick"] / (total_time_across_individuals["sick"] + 1)])).double().to(
            device).detach()

        m_lambda = torch.randn(M_lambda, 1).double().to(device).detach()
        # L_lambda = torch.randn(int(M_lambda * (M_lambda + 1) / 2), 1).double().to(device).detach() / 10

        S = covar_module0_lambda(zt_list_lambda).evaluate().clone().detach() + eps * torch.eye(M_lambda,dtype=torch.double).to(device)

        tril_indices = torch.tril_indices(row=M_lambda, col=M_lambda, offset=0)

        L_lambda = torch.linalg.cholesky(S)[0]

        L_lambda = torch.unsqueeze(L_lambda[tril_indices[0, :], tril_indices[1, :]], 1)

        adam_param_list.append({'params': beta})
        adam_param_list.append({'params': m_lambda})
        adam_param_list.append({'params': L_lambda})
        m_lambda.requires_grad_(True)
        L_lambda.requires_grad_(True)
        beta.requires_grad_(True)

        if varying_T:
            n_batches_train = (P + subjects_per_batch_tpp - 1) // subjects_per_batch_tpp
            dataloader_train = HensmanDataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(
                VaryingLengthSubjectSampler(dataset, id_covariate), subjects_per_batch_tpp), num_workers=1)
        else:
            batch_size = subjects_per_batch_tpp * T
            n_batches_train = (P * T + batch_size - 1) // (batch_size)
            dataloader_train = HensmanDataLoader(dataset,
                                                 batch_sampler=BatchSampler(SubjectSampler(dataset, P, T), batch_size,
                                                                            drop_last=False), num_workers=1)
        # Mini-batching for validation
        if varying_T:
            n_batches_val = (num_val_instances + subjects_per_batch_tpp - 1) // subjects_per_batch_tpp
            dataloader_val = HensmanDataLoader(validation_dataset, batch_sampler=VaryingLengthBatchSampler(
                VaryingLengthSubjectSampler(validation_dataset, id_covariate), subjects_per_batch_tpp), num_workers=1)
        else:
            batch_size = subjects_per_batch_tpp * T
            n_batches_val = (num_val_instances * T + batch_size - 1) // (batch_size)
            dataloader_val = HensmanDataLoader(validation_dataset,
                                               batch_sampler=BatchSampler(
                                                   SubjectSampler(validation_dataset, num_val_instances, T), batch_size,
                                                   drop_last=False), num_workers=1)

        optimiser = torch.optim.Adam(adam_param_list, lr=1e-3)

        print("Training TPP")

        tpp.pretrain_TPP(dataloader_train, dataloader_val, covar_module0_lambda, gp_model_lambda, zt_list_lambda,
                         m_lambda, L_lambda, tril_indices, beta,
                         id_covariate, disease_covariate, epochs, optimiser, P, n_batches_train, save_path)

        gp_model_lambda.load_state_dict(torch.load(os.path.join(save_path, 'gp_model_lambda_best.pth'), map_location=torch.device(device)))
        zt_list_lambda = torch.load(os.path.join(save_path, 'zt_list_lambda_best.pth'),
                                    map_location=torch.device(device))
        m_lambda = torch.load(os.path.join(save_path, 'm_lambda_best.pth'), map_location=torch.device(device)).detach()
        L_lambda = torch.load(os.path.join(save_path, 'L_lambda_best.pth'), map_location=torch.device(device)).detach()
        beta = torch.load(os.path.join(save_path, 'beta_best.pth'), map_location=torch.device(device)).detach()

        adam_param_list = []

        adam_param_list.append({'params': covar_module0_y.parameters()})
        adam_param_list.append({'params': covar_module1_y.parameters()})
        adam_param_list.append({'params': covar_module0_m.parameters()})
        adam_param_list.append({'params': covar_module1_m.parameters()})

        covar_module0_y.train().double()
        covar_module1_y.train().double()
        likelihoods_y.train().double()
        covar_module0_m.train().double()
        covar_module1_m.train().double()
        likelihoods_m.train().double()

        m_y = torch.randn(latent_dim_zy, M_y, 1).double().to(device).detach()
        H_y = (torch.randn(latent_dim_zy, M_y, M_y) / 10).double().to(device).detach()
        m_m = torch.randn(latent_dim_zm, M_m, 1).double().to(device).detach()
        H_m = (torch.randn(latent_dim_zm, M_m, M_m) / 10).double().to(device).detach()

        if natural_gradient:
            H_y = torch.matmul(H_y, H_y.transpose(-1, -2)).detach().requires_grad_(False)
            H_m = torch.matmul(H_m, H_m.transpose(-1, -2)).detach().requires_grad_(False)

        if not natural_gradient:
            adam_param_list.append({'params': m_y})
            adam_param_list.append({'params': H_y})
            adam_param_list.append({'params': m_m})
            adam_param_list.append({'params': H_m})
            m_y.requires_grad_(True)
            H_y.requires_grad_(True)
            m_m.requires_grad_(True)
            H_m.requires_grad_(True)

        nnet_model.train()
        adam_param_list.append({'params': nnet_model.parameters()})
        optimiser = torch.optim.Adam(adam_param_list, lr=1e-3)

        if memory_dbg:
            print("Max memory allocated during initialisation: {:.2f} MBs".format(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
            torch.cuda.reset_max_memory_allocated(device)

        if type_KL == 'closed':
            covar_modules = [covar_module_y, covar_module_m]
        elif type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
            covar_modules = [covar_module0_y, covar_module1_m]

        start = timer()

        _ = hensman_training(nnet_model, tpp, type_nnet, epochs, dataset,
                             optimiser, type_KL, num_samples, latent_dim_zy, latent_dim_zm,
                             covar_module0_y, covar_module1_y, covar_module0_m, covar_module1_m, covar_module0_lambda,
                             likelihoods_y, likelihoods_m, m_y, H_y, m_m, H_m, m_lambda, L_lambda, tril_indices, beta,
                             zt_list_y, zt_list_m, zt_list_lambda, P, T, varying_T, Q,
                             weight, id_covariate, disease_covariate, loss_function, natural_gradient,
                             natural_gradient_lr,
                             subjects_per_batch, memory_dbg, eps,
                             results_path, validation_dataset, validation_dataset, prediction_dataset, gp_model_y,
                             gp_model_m, gp_model_lambda,
                             csv_file_validation_data=csv_file_validation_data,
                             csv_file_validation_label=csv_file_validation_label,
                             validation_mask_file=validation_mask_file, data_source_path=data_source_path)

        m_y, H_y, m_m, H_m = _[5], _[6], _[7], _[8]

        print("Duration of training: {:.2f} seconds".format(timer() - start))

        if memory_dbg:
            print("Max memory allocated during training: {:.2f} MBs".format(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
            torch.cuda.reset_max_memory_allocated(device)

        penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr = _[0], _[1], _[2], _[3], _[4]
        print('Best results in epoch: ' + str(_[9]))
        # saving
        print('Saving')
        pd.to_pickle([penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr],
                     os.path.join(save_path, 'diagnostics.pkl'))

        pd.to_pickle([train_x, mu_y, log_var_y, Z_y, mu_m, log_var_m, Z_m, label_id],
                     os.path.join(save_path, 'plot_values.pkl'))
        torch.save(nnet_model.state_dict(), os.path.join(save_path, 'final-vae_model.pth'))

        torch.save(gp_model_y.state_dict(), os.path.join(save_path, 'gp_model_y.pth'))
        torch.save(gp_model_m.state_dict(), os.path.join(save_path, 'gp_model_m.pth'))
        torch.save(zt_list_y, os.path.join(save_path, 'zt_list_y.pth'))
        torch.save(zt_list_m, os.path.join(save_path, 'zt_list_m.pth'))
        torch.save(m_y, os.path.join(save_path, 'm_y.pth'))
        torch.save(H_y, os.path.join(save_path, 'H_y.pth'))
        torch.save(m_m, os.path.join(save_path, 'm_m.pth'))
        torch.save(H_m, os.path.join(save_path, 'H_m.pth'))

        torch.save(gp_model_lambda.state_dict(), os.path.join(save_path, 'gp_model_lambda.pth'))
        torch.save(zt_list_lambda, os.path.join(save_path, 'zt_list_lambda.pth'))
        torch.save(m_lambda, os.path.join(save_path, 'm_lambda.pth'))
        torch.save(L_lambda, os.path.join(save_path, 'L_lambda.pth'))
        torch.save(beta, os.path.join(save_path, 'beta.pth'))

        if memory_dbg:
            print("Max memory allocated during saving and post-processing: {:.2f} MBs".format(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
            torch.cuda.reset_max_memory_allocated(device)

        nnet_model.eval()

        if run_validation:
            covar_module0_y.eval()
            covar_module1_y.eval()
            covar_module0_m.eval()
            covar_module1_m.eval()
            covar_module0_lambda.eval()
            likelihoods_y.eval()
            likelihoods_m.eval()
            with torch.no_grad():
                validate(nnet_model, tpp, type_nnet, validation_dataset, type_KL, num_samples, latent_dim_zy,
                         latent_dim_zm, covar_module0_y, covar_module1_y, covar_module0_m, covar_module1_m,
                         covar_module0_lambda, likelihoods_y, likelihoods_m,
                         zt_list_y, zt_list_m, zt_list_lambda, m_lambda, L_lambda, tril_indices, beta, T, weight,
                         id_covariate, disease_covariate, loss_function, m_y, m_m, H_y, H_m, P, natural_gradient, eps=1e-6)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        gp_model_y.load_state_dict(
            torch.load(os.path.join(save_path, 'gp_model_y_best.pth'), map_location=torch.device(device)))
        gp_model_m.load_state_dict(
            torch.load(os.path.join(save_path, 'gp_model_m_best.pth'), map_location=torch.device(device)))
        zt_list_y = torch.load(os.path.join(save_path, 'zt_list_y_best.pth'), map_location=torch.device(device))
        zt_list_m = torch.load(os.path.join(save_path, 'zt_list_m_best.pth'), map_location=torch.device(device))
        m_y = torch.load(os.path.join(save_path, 'm_y_best.pth'), map_location=torch.device(device)).detach()
        H_y = torch.load(os.path.join(save_path, 'H_y_best.pth'), map_location=torch.device(device)).detach()
        m_m = torch.load(os.path.join(save_path, 'm_m_best.pth'), map_location=torch.device(device)).detach()
        H_m = torch.load(os.path.join(save_path, 'H_m_best.pth'), map_location=torch.device(device)).detach()

        gp_model_lambda.load_state_dict(
            torch.load(os.path.join(save_path, 'gp_model_lambda_best.pth'), map_location=torch.device(device)))
        zt_list_lambda = torch.load(os.path.join(save_path, 'zt_list_lambda_best.pth'),
                                    map_location=torch.device(device))
        m_lambda = torch.load(os.path.join(save_path, 'm_lambda_best.pth'), map_location=torch.device(device)).detach()
        L_lambda = torch.load(os.path.join(save_path, 'L_lambda_best.pth'), map_location=torch.device(device)).detach()
        beta = torch.load(os.path.join(save_path, 'beta_best.pth'), map_location=torch.device(device)).detach()

        nnet_model.load_state_dict(
            torch.load(os.path.join(save_path, "nnet_model_best.pth"), map_location=torch.device(device)))

        nnet_model.eval()

        covar_module0_y.eval()
        covar_module1_y.eval()
        covar_module0_m.eval()
        covar_module1_m.eval()
        covar_module0_lambda.eval()
        likelihoods_y.eval()
        likelihoods_m.eval()

        if run_tests:
            train_prediction_dataloader = DataLoader(train_prediction_dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(train_prediction_dataset, id_covariate), subjects_per_batch), num_workers=1)
            full_mu_y = torch.zeros(len(train_prediction_dataset), latent_dim_zy, dtype=torch.double).to(device)
            full_mu_m = torch.zeros(len(train_prediction_dataset), latent_dim_zm, dtype=torch.double).to(device)
            prediction_x = torch.zeros(len(train_prediction_dataset), Q + 1, dtype=torch.double).to(device)

            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(train_prediction_dataloader):
                    label_id = sample_batched['idx']
                    curr_x = sample_batched['label'].double().to(device)
                    prev_timestamps = sample_batched['prev_timestamps'].double().to(device)
                    disease_present = curr_x[:, disease_covariate].type(dtype=torch.int64)
                    one_hot_representation = F.one_hot(disease_present, num_classes=2).type(dtype=torch.double).to(device)
                    beta_disease = torch.matmul(one_hot_representation, beta)
                    mean_z_lambda, Sigma_z_lambda = tpp.compute_variational_params(m_lambda, L_lambda, tril_indices,
                                                                                   zt_list_lambda, covar_module0_lambda,
                                                                                   prev_timestamps)
                    marginal_var = torch.diagonal(Sigma_z_lambda)
                    prediction_x[label_id] = torch.cat((curr_x, torch.unsqueeze((mean_z_lambda + beta_disease) ** 2 + marginal_var, 1)), dim=1)
                    data = sample_batched['data'].double().to(device)
                    mask = sample_batched['mask'].double().to(device)
                    covariates = torch.cat((prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate + 1:]), dim=1)
                    mu_y, log_var_y = nnet_model.encode_y(data)
                    mu_m, log_var_m = nnet_model.encode_m(mask)
                    full_mu_y[label_id] = mu_y
                    full_mu_m[label_id] = mu_m

        # MSE test
        if run_tests:
            with torch.no_grad():
                recon_loss_GP_y, recon_loss_GP_m = MSE_test_GPapprox(csv_file_prediction_data,
                                                                     csv_file_prediction_label,
                                                                     prediction_mask_file, data_source_path,
                                                                     type_nnet,
                                                                     nnet_model, tpp, covar_module0_y,
                                                                     covar_module1_y, covar_module0_m,
                                                                     covar_module1_m, covar_module0_lambda,
                                                                     likelihoods_y,
                                                                     likelihoods_m, results_path, latent_dim_zy,
                                                                     latent_dim_zm,
                                                                     prediction_x, full_mu_y, full_mu_m, zt_list_y,
                                                                     zt_list_m, zt_list_lambda, m_lambda, L_lambda,
                                                                     tril_indices, beta, P, T, id_covariate,
                                                                     disease_covariate, varying_T, dataset_type=dataset_type)

                pred_y_arr.append(recon_loss_GP_y)
                pred_m_arr.append(recon_loss_GP_m)


        if run_imputation and dataset_type == 'HealthMNIST':
            imputation_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_imputation_data,
                                                        csv_file_label=csv_file_prediction_label,
                                                        mask_file=prediction_mask_file, root_dir=data_source_path,
                                                        transform=transforms.ToTensor())

            with torch.no_grad():
                imputation_mse = impute_data(nnet_model, imputation_dataset, id_covariate, subjects_per_batch, num_dim)
                imputation_arr.append(imputation_mse)

        if memory_dbg:
            print("Max memory allocated during tests: {:.2f} MBs".format(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
            torch.cuda.reset_max_memory_allocated(device)

    pred_y_arr = np.array(pred_y_arr)
    pred_m_arr = np.array(pred_m_arr)
    imputation_arr = np.array(imputation_arr)
    accuracy_arr = np.array(accuracy_arr)
    print("prediction data error: ", np.mean(pred_y_arr), " +- ", np.std(pred_y_arr))
    print("prediction mask error: ", np.mean(pred_m_arr), " +- ", np.std(pred_m_arr))
    print("imputation error: ", np.mean(imputation_arr), " +- ", np.std(imputation_arr))

