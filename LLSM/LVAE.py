import os
import sys

from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
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
from torch.optim.lr_scheduler import StepLR

eps = 1e-6

def train_val_split_health_mnist(csv_file_data_full, csv_file_data, csv_file_label, mask_file, csv_file_train_data, csv_file_train_label,
            train_mask_file, csv_file_validation_data, csv_file_validation_label, validation_mask_file,
            csv_file_prediction_data, csv_file_prediction_label, prediction_mask_file,  csv_file_train_prediction_data, csv_file_train_prediction_label, train_prediction_mask_file,
            csv_file_imputation_data, root_dir, data_model_path, num_val_instances, num_predict_instances):

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
    predict_idx = np.zeros(num_predict_instances * 15, dtype=int)
    train_prediction_idx = np.copy(train_idx)

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

    train_prediction_data = data.iloc[train_prediction_idx, :]
    train_prediction_labels = labels.iloc[train_prediction_idx, :]
    train_prediction_mask = mask.iloc[train_prediction_idx, :]

    val_data = data.iloc[val_idx, :]
    val_labels = labels.iloc[val_idx, :]
    val_mask = mask.iloc[val_idx, :]

    predict_data = full_data.iloc[predict_idx, :]
    predict_labels = labels.iloc[predict_idx, :]
    predict_mask = mask.iloc[predict_idx, :]

    imputation_data = full_data.iloc[predict_idx, :]

    train_data.to_csv(os.path.join(data_model_path, csv_file_train_data), index=False, header=False)
    train_labels.to_csv(os.path.join(data_model_path, csv_file_train_label), index=False)
    train_mask.to_csv(os.path.join(data_model_path, train_mask_file), index=False, header=False)

    val_data.to_csv(os.path.join(data_model_path, csv_file_validation_data), index=False, header=False)
    val_labels.to_csv(os.path.join(data_model_path, csv_file_validation_label), index=False)
    val_mask.to_csv(os.path.join(data_model_path, validation_mask_file), index=False, header=False)

    predict_data.to_csv(os.path.join(data_model_path, csv_file_prediction_data), index=False, header=False)
    predict_labels.to_csv(os.path.join(data_model_path, csv_file_prediction_label), index=False)
    predict_mask.to_csv(os.path.join(data_model_path, prediction_mask_file), index=False, header=False)

    train_prediction_data.to_csv(os.path.join(data_model_path, csv_file_train_prediction_data), index=False, header=False)
    train_prediction_labels.to_csv(os.path.join(data_model_path, csv_file_train_prediction_label), index=False)
    train_prediction_mask.to_csv(os.path.join(data_model_path, train_prediction_mask_file), index=False, header=False)

    imputation_data.to_csv(os.path.join(data_model_path, csv_file_imputation_data), index=False, header=False)



if __name__ == "__main__":
    """
    Root file for running L-VAE.

    Run command: python LVAE.py --f=path_to_config-file.txt 
    """

    # create parser and set variables

    model_name = "LVAE"
    opt = ModelArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    assert not (hensman and mini_batch)
    assert loss_function == 'mse' or loss_function == 'nll', ("Unknown loss function " + loss_function)
    assert not varying_T or hensman, "varying_T can't be used without hensman"

    print(torch.version.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    pred_y_arr = []
    pred_m_arr = []
    imputation_arr = []

    number_of_repetitions = 1

    original_data_path = data_source_path

    data_source_path = data_model_path

    for l in range(number_of_repetitions):

        if dataset_type == "HealthMNIST" and split_data:
            train_val_split_health_mnist(csv_file_data_full, csv_file_data, csv_file_label, mask_file,
                                         csv_file_train_data,
                                         csv_file_train_label, train_mask_file, csv_file_validation_data,
                                         csv_file_validation_label, validation_mask_file,
                                         csv_file_prediction_data, csv_file_prediction_label, prediction_mask_file,
                                         csv_file_train_prediction_data, csv_file_train_prediction_label,
                                         train_prediction_mask_file, csv_file_imputation_data, original_data_path, data_model_path,
                                         num_val_instances, num_predict_instances)

        pretrain_for_LVAE(csv_file_train_data, csv_file_train_label, train_mask_file, "nll", type_nnet, dataset_type,
                          data_source_path,
                          num_dim, vy_init, vy_fixed, latent_dim_zy, latent_dim_zm, id_covariate, save_path,
                          model_name)  # pretrain model by classical VAE


        # Set up dataset
        if type_nnet == 'conv':
            if dataset_type == 'HealthMNIST':
                dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_train_data, csv_file_label=csv_file_train_label,
                                                 mask_file=train_mask_file, root_dir=data_source_path,
                                                 transform=transforms.ToTensor())

        elif type_nnet == 'simple':
            dataset = PhysionetDataset(csv_file_data=csv_file_train_data, csv_file_label=csv_file_train_label,
                                           mask_file=train_mask_file, root_dir=data_source_path,
                                           transform=transforms.ToTensor())

        # Set up prediction dataset
        if run_tests:
            if dataset_type == 'HealthMNIST':
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


            if dataset_type == 'HealthMNIST':
                    train_prediction_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_train_prediction_data,
                                                     csv_file_label=csv_file_train_prediction_label,
                                                     mask_file=train_prediction_mask_file, root_dir=data_source_path,
                                                     transform=transforms.ToTensor())

            elif dataset_type == 'Physionet':
                    train_prediction_dataset = PhysionetDataset(csv_file_data=csv_file_train_prediction_data,
                                                                csv_file_label=csv_file_train_prediction_label,
                                                                mask_file=train_prediction_mask_file,
                                                                root_dir=data_source_path,
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

        # set up Data Loader for GP initialisation
        # Kalle: Hard-coded batch size 1000
        setup_dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=1)

        # Get values for GP initialisation:
        Z_y = torch.zeros(N, latent_dim_zy, dtype=torch.double).to(device)
        Z_m = torch.zeros(N, latent_dim_zm, dtype=torch.double).to(device)
        train_x = torch.zeros(N, Q, dtype=torch.double).to(device)
        nnet_model.eval()
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(setup_dataloader):
                # no mini-batching. Instead get a batch of dataset size
                mask = sample_batched['mask'].double().to(device)
                label_id = sample_batched['idx']
                train_x[label_id] = sample_batched['label'].double().to(device)
                data = sample_batched['data'].double().to(device)

                covariates = torch.cat((train_x[label_id, :id_covariate], train_x[label_id, id_covariate + 1:]), dim=1)

                mu_y, log_var_y = nnet_model.encode_y(data)
                mu_m, log_var_m = nnet_model.encode_m(mask)
                Z_y[label_id] = nnet_model.sample_latent(mu_y, log_var_y)
                Z_m[label_id] = nnet_model.sample_latent(mu_m, log_var_m)

        covar_module_y = []
        covar_module0_y = []
        covar_module1_y = []
        covar_module_m = []
        covar_module0_m = []
        covar_module1_m = []
        zt_list_y = []
        likelihoods_y = []
        gp_models_y = []
        zt_list_m = []
        likelihoods_m = []
        gp_models_m = []
        adam_param_list = []

        likelihoods_y = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim_zy]),
                                                                    noise_constraint=gpytorch.constraints.GreaterThan(
                                                                        1.000E-08)).to(device)

        likelihoods_m = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim_zm]),
                                                                noise_constraint=gpytorch.constraints.GreaterThan(
                                                                    1.000E-08)).to(device)

        if constrain_scales:
            likelihoods_y.noise = 1.000E-08
            likelihoods_y.raw_noise.requires_grad = False
            likelihoods_m.noise = 1.000E-08
            likelihoods_m.raw_noise.requires_grad = False

        covar_module0_y, covar_module1_y = generate_kernel_batched(latent_dim_zy,
                                                                   cat_kernel, bin_kernel, sqexp_kernel,
                                                                   cat_int_kernel, bin_int_kernel,
                                                                   covariate_missing_val, id_covariate)

        covar_module0_m, covar_module1_m = generate_kernel_batched(latent_dim_zm,
                                                                   cat_kernel, bin_kernel, sqexp_kernel,
                                                                   cat_int_kernel, bin_int_kernel,
                                                                   covariate_missing_val, id_covariate)

        gp_model_y = ExactGPModel(train_x, Z_y.type(torch.DoubleTensor), likelihoods_y,
                                  covar_module0_y + covar_module1_y).to(device)

        gp_model_m = ExactGPModel(train_x, Z_m.type(torch.DoubleTensor), likelihoods_m,
                                  covar_module0_m + covar_module1_m).to(device)

        # initialise inducing points for y and m
        zt_list_y = torch.zeros(latent_dim_zy, M_y, Q, dtype=torch.double).to(device)
        zt_list_m = torch.zeros(latent_dim_zm, M_m, Q, dtype=torch.double).to(device)
        n = train_x.shape[0]

        inducing_points_indices = np.random.choice(N, M_y, replace=False)

        # inducing points for y
        for i in range(latent_dim_zy):
            zt_list_y[i] = train_x[inducing_points_indices].clone().detach()

        # inducing points for m
        for i in range(latent_dim_zm):
            zt_list_m[i] = train_x[inducing_points_indices].clone().detach()

        adam_param_list.append({'params': covar_module0_y.parameters()})
        adam_param_list.append({'params': covar_module1_y.parameters()})
        adam_param_list.append({'params': covar_module0_m.parameters()})
        adam_param_list.append({'params': covar_module1_m.parameters()})
        adam_param_list.append({'params': zt_list_y})
        adam_param_list.append({'params': zt_list_m})

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
        optimiser = torch.optim.Adam(adam_param_list, lr=0.001)
        scheduler = StepLR(optimiser, step_size=50, gamma=1)

        if memory_dbg:
            print("Max memory allocated during initialisation: {:.2f} MBs".format(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
            torch.cuda.reset_max_memory_allocated(device)

        if type_KL == 'closed':
            covar_modules = [covar_module_y, covar_module_m]
        elif type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
            covar_modules = [covar_module0_y, covar_module1_m]

        start = timer()

        _ = hensman_training(nnet_model, type_nnet, epochs, dataset,
                             optimiser, scheduler, type_KL, num_samples, latent_dim_zy, latent_dim_zm,
                             covar_module0_y, covar_module1_y, covar_module0_m, covar_module1_m,
                             likelihoods_y, likelihoods_m, m_y, H_y, m_m, H_m, zt_list_y, zt_list_m, P, T, varying_T, Q,
                             weight,
                             id_covariate, loss_function, natural_gradient, natural_gradient_lr,
                             subjects_per_batch, memory_dbg, eps,
                             results_path, validation_dataset,
                             validation_dataset, prediction_dataset, gp_model_y, gp_model_m,
                             csv_file_validation_data=csv_file_validation_data,
                             csv_file_validation_label=csv_file_validation_label,
                             validation_mask_file=validation_mask_file,
                             data_source_path=data_source_path)

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

        try:
            torch.save(gp_model_y.state_dict(), os.path.join(save_path, 'gp_model_y.pth'))
            torch.save(gp_model_m.state_dict(), os.path.join(save_path, 'gp_model_m.pth'))
            torch.save(zt_list_y, os.path.join(save_path, 'zt_list_y.pth'))
            torch.save(zt_list_m, os.path.join(save_path, 'zt_list_m.pth'))
            torch.save(m_y, os.path.join(save_path, 'm_y.pth'))
            torch.save(H_y, os.path.join(save_path, 'H_y.pth'))
            torch.save(m_m, os.path.join(save_path, 'm_m.pth'))
            torch.save(H_m, os.path.join(save_path, 'H_m.pth'))
        except:
            pass

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
            likelihoods_y.eval()
            likelihoods_m.eval()
            with torch.no_grad():
                val_pred_mse = validate(nnet_model, validation_dataset, type_KL, latent_dim_zy, latent_dim_zm, covar_module0_y, covar_module1_y, covar_module0_m, covar_module1_m, likelihoods_y, likelihoods_m,
                                    zt_list_y, zt_list_m, weight, id_covariate, loss_function, m_y, m_m, H_y, H_m, P, natural_gradient=True, eps=1e-6)

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
        nnet_model.load_state_dict(
            torch.load(os.path.join(save_path, "nnet_model_best.pth"), map_location=torch.device(device)))

        nnet_model.eval()

        covar_module0_y.eval()
        covar_module1_y.eval()
        covar_module0_m.eval()
        covar_module1_m.eval()
        likelihoods_y.eval()
        likelihoods_m.eval()

        if run_tests:
            #train_prediction_dataloader = DataLoader(train_prediction_dataset, batch_size=80, shuffle=False, num_workers=1)
            train_prediction_dataloader = DataLoader(train_prediction_dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(train_prediction_dataset, id_covariate), subjects_per_batch), num_workers=4)
            full_mu_y = torch.zeros(len(train_prediction_dataset), latent_dim_zy, dtype=torch.double).to(device)
            full_mu_m = torch.zeros(len(train_prediction_dataset), latent_dim_zm, dtype=torch.double).to(device)
            prediction_x = torch.zeros(len(train_prediction_dataset), Q, dtype=torch.double).to(device)
            # full_mu_y = torch.zeros(80, latent_dim_zy, dtype=torch.double).to(device)
            # full_mu_m = torch.zeros(80, latent_dim_zm, dtype=torch.double).to(device)
            # prediction_x = torch.zeros(80, Q, dtype=torch.double).to(device)
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(train_prediction_dataloader):
                    label_id = sample_batched['idx']
                    prediction_x[label_id] = sample_batched['label'].double().to(device)
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
                                                                     csv_file_prediction_label, prediction_mask_file,
                                                                     data_source_path, type_nnet,
                                                                     nnet_model, covar_module0_y, covar_module1_y,
                                                                     covar_module0_m, covar_module1_m, likelihoods_y,
                                                                     likelihoods_m,
                                                                     results_path, latent_dim_zy, latent_dim_zm,
                                                                     prediction_x,
                                                                     full_mu_y, full_mu_m, zt_list_y, zt_list_m, P, T,
                                                                     id_covariate, varying_T, dataset_type=dataset_type)

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
    print("prediction data error: ", np.mean(pred_y_arr), " +- ", np.std(pred_y_arr))
    print("prediction mask error: ", np.mean(pred_m_arr), " +- ", np.std(pred_m_arr))
    print("imputation error: ", np.mean(imputation_arr), " +- ", np.std(imputation_arr))


