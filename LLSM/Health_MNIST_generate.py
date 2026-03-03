# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import argparse
# import math
# from scipy.special import expit as sigmoid
# from scipy import ndimage
#
#
# """
# Code to generate the Health MNIST data.
#
# This code manipulates the original MNIST images as described in the L-VAE paper.
# """
#
#
# def parse_arguments():
#     """
#     Parse the command line arguments
#     :return: parsed arguments object (2 arguments)
#     """
#
#     parser = argparse.ArgumentParser(description='Enter configuration for generating data')
#     parser.add_argument('--source', type=str, default='./trainingSet', help='Path to MNIST image root directory')
#     parser.add_argument('--destination', type=str, default='./data', help='Path to save the generated dataset')
#     parser.add_argument('--num_3', type=int, default=50, help='Number of unique instances for digit 3')
#     parser.add_argument('--num_6', type=int, default=50, help='Number of unique instances for digit 6')
#     parser.add_argument('--missing', type=float, default=25, choices=range(-1, 101),
#                         help='Percentage of missing in range [0, 100]')
#     parser.add_argument('--data_file_name', type=str, default='health_MNIST_data.csv',
#                         help='File name of generated data')
#     parser.add_argument('--data_masked_file_name', type=str, default='health_MNIST_data_masked.csv',
#                         help='File name of generated masked data')
#     parser.add_argument('--labels_file_name', type=str, default='health_MNIST_label.csv',
#                         help='File name of generated labels')
#     parser.add_argument('--mask_file_name', type=str, default='mask.csv',
#                         help='File name of generated mask')
#
#     return vars(parser.parse_args())
#
#
# def create_data_file(path, open_str):
#     if os.path.exists(path):
#         os.remove(path)
#     return open(path, open_str)
#
#
# def write_label_file_header(label_file):
#     df = pd.DataFrame.from_dict({}, orient='index',
#                                 columns=['subject', 'digit', 'angle', 'disease', 'gender',
#                                          'time_age', 'location'])
#     df.to_csv(label_file, index=False)
#
#
# def save_data(data_file, mask_file, data_masked_file, label_file, rotated_MNIST, label_dict, mask):
#     # save rotated MNIST
#     np.savetxt(data_file, rotated_MNIST, fmt='%d', delimiter=',')
#
#     # generate mask
#     # mask = np.random.choice([0, 1], size=rotated_MNIST.shape, p=[missing_frac, observed_frac])
#
#     # 0 implies missing, 1 implies observed
#     masked_data = np.multiply(rotated_MNIST, mask)
#
#     np.savetxt(data_masked_file, masked_data, fmt='%d', delimiter=',')
#     np.savetxt(mask_file, mask, fmt='%d', delimiter=',')
#
#     df = pd.DataFrame.from_dict(label_dict, orient='index',
#                                 columns=['subject', 'digit', 'angle', 'disease', 'gender',
#                                          'time_age', 'location'])
#
#     # save labels
#     df.to_csv(label_file, index=False, header=False)
#
#
# def generate_mask_mnar(image, sick_var, time_point, shape):
#     image = image / 255
#     if sick_var:
#         p = (image / 2) + (time_point / 50) * int(time_point > 0)
#         mask = np.random.binomial(1, 1 - p, shape)
#
#     else:
#         p = image / 2
#         mask = np.random.binomial(1, 1 - p, shape)
#
#     mask = mask.reshape(1, 1296)
#     return mask
#
#
# def generate_mask_mnar_4(image, sick_var, time_point, shape):
#     image = image / 255
#     mask = np.ones(image.shape)
#     if (not sick_var) or int(time_point <= 0):
#         center = int(image.shape[0] / 2)
#         mask[center - 4:center + 4, center - 4:center + 4] = 0
#     else:
#         size_mask = time_point * 2
#         half_size = int(size_mask / 2)
#         center_y = image.shape[0] - size_mask
#         center_x = size_mask
#         mask[center_y - half_size: center_y + half_size, center_x - half_size:center_x + half_size] = 0
#
#     mask = mask.reshape(shape)
#     return mask
#
#
# def generate_mask_mnar_5(image, sick_var, rotation,
#                          shape):  # for sick instances box increases in time, moving along the diagonal
#     image = image / 255
#     p = (image / 2) * 1.5
#     mask = np.random.binomial(1, 1 - p, image.shape)
#     if sick_var:
#         size_mask = int(18 * rotation)
#         half_size = int(size_mask / 2)
#         center_y = image.shape[0] - size_mask
#         center_x = size_mask
#         mask[center_y - half_size: center_y + half_size, center_x - half_size:center_x + half_size] = 0
#
#     mask = mask.reshape(shape)
#     return mask
#
#
# def generate_mask_mnar_6(gender, sick_var, time_point,
#                          shape):  # for sick instances, the width of rectangular box increases in time
#     mid = 18
#     mask = np.ones((36, 36))
#     if sick_var and time_point > 0:
#         if gender:
#             mask[0:mid, 0:time_point * 3] = 0
#         else:
#             mask[mid:36, 0:time_point * 3] = 0
#
#     return mask.reshape(shape)
#
#
# def generate_mask_mnar_7(image, time_point, shape):  # boxes deacrease in time
#     image = image / 255
#     p = (image / 2) * 1.5
#     mask = np.random.binomial(1, 1 - p, image.shape)
#     if int(time_point) < 0:
#         size_mask = abs(time_point) * 2
#         half_size = int(size_mask / 2)
#         center_y = image.shape[0] - size_mask
#         center_x = size_mask
#         mask[center_y - half_size: center_y + half_size, center_x - half_size:center_x + half_size] = 0
#
#     mask = mask.reshape(shape)
#     return mask
#
#
# def generate_mask_mnar_8(image, time_point, shape):  # boxes move along the diagonal all the time
#     image = image / 255
#     p = (image / 2) * 1.5
#     mask = np.random.binomial(1, 1 - p, image.shape)
#     idx = time_point + 9
#     size = 9
#     curr_x = (idx * size) % 36
#     curr_y = (idx * size) % 36
#     mask[curr_y:curr_y + size, curr_x: curr_x + size] = 0
#     mask = mask.reshape(shape)
#     return mask
#
#
# def generate_locations_probs(missingness_block_size, age, mask_time_point, dim, p_l_time, p_r_time, p_age):
#     locations_prob = np.zeros(dim - missingness_block_size)
#     center = dim // 2 - missingness_block_size // 2
#     center = \
#     np.random.choice([min(dim - missingness_block_size - 1, center + mask_time_point), center - mask_time_point,
#                       min(dim - missingness_block_size - 1, center + age // 2)], 1, p=[p_r_time, p_l_time, p_age])[0]
#     k = int(center)
#     while k >= 0:
#         locations_prob[k] = dim - (center - k)
#         k -= 1
#
#     k = int(center) + 1
#     while k < dim - missingness_block_size:
#         locations_prob[k] = dim - (k - center)
#         k += 1
#
#     return locations_prob / np.sum(locations_prob)
#
#
# def generate_edge_points(missingness_block_size, age, mask_time_point, dim, p_l_time, p_r_time, p_age):
#     locations_prob = generate_locations_probs(missingness_block_size, age, mask_time_point, dim, p_l_time, p_r_time,
#                                               p_age)
#     center = missingness_block_size // 2 + np.random.choice(dim - missingness_block_size, 1, p=locations_prob)[0]
#     left = max(center - missingness_block_size // 2, 0)
#     right = min(center + missingness_block_size // 2, dim)
#     return left, right
#
#
# def homogenous_poisson_process(mu, t_n_1):
#     u = np.random.rand()
#     t = -math.log(1 - u) / mu + t_n_1
#     return t
#
#
# def inhomogenous_poisson_process(g, t_n_1):
#     g_max = 1
#     t = t_n_1
#     while True:
#         t = homogenous_poisson_process(g_max, t)
#         u = np.random.rand()
#         if u <= g(t) / g_max:
#             return t
#
#
# def trigerring_kernel(t, h):
#     kernel = 0
#     for i in range(len(h)):
#         if h[i] > t:
#             break
#         kernel += 0.5 * math.exp(-((t - h[i])) ** 2 / 0.5)
#
#     return kernel
#
#
# def hawkes_process(t_n_1, h, lambda_base):
#     lambda_max = lambda_base + trigerring_kernel(t_n_1, h)
#     t = t_n_1
#     while True:
#         t = homogenous_poisson_process(lambda_max, t)
#         u = np.random.rand()
#         if u <= (lambda_base + trigerring_kernel(t, h)) / lambda_max:
#             return t
#
#
# def calculate_intensity(t, h, lambda_base):
#     intensity = lambda_base + trigerring_kernel(t, h)
#     return intensity
#
#
# def plot_intensity_hawkes(h):
#     t = np.linspace(0, 40, 1000)
#     lambda_base = 1
#     lambd = []
#     for el in t:
#         lambd.append(lambda_base + trigerring_kernel(el, h))
#
#     lambd_h = []
#     for el in h:
#         lambd_h.append(lambda_base + trigerring_kernel(el, h))
#
#     plt.plot(t, lambd)
#     plt.scatter(h[8:40], lambd_h[8:40], color="r")
#     plt.savefig("./figures/original_intensity.png")
#     plt.show()
#
#
# def sigmoid(t, a):
#     return 1 / (3 * (1 + np.exp(-(t - a))))
#
#
# def tanh(x):
#     return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
#
#
# def normalized_rotation(intensity):
#     return (tanh(intensity) - tanh(1)) * 4.5
#
#
# # rotation based on intensity
# if __name__ == '__main__':
#     opt = parse_arguments()
#     for key in opt.keys():
#         print('{:s}: {:s}'.format(key, str(opt[key])))
#     locals().update(opt)
#
#     digit_mod = {'3': num_3, '6': num_6}
#     sick_prob = 0.5  # probability of instance being sick
#     sample_index = 0
#     subject_index = 0
#     label_dict = {}
#     gender = 0
#     beta = 10
#
#     # 20 time points
#     time_points_rotations = np.arange(0, 20)
#
#     # accumulate digits
#     rotated_MNIST = np.empty((0, 1296))
#     mask = np.empty((0, 1296))
#
#     data_file = create_data_file(os.path.join(destination, data_file_name), "ab")
#     mask_file = create_data_file(os.path.join(destination, mask_file_name), "ab")
#     data_masked_file = create_data_file(os.path.join(destination, data_masked_file_name), "ab")
#     label_file = create_data_file(os.path.join(destination, labels_file_name), "a")
#     write_label_file_header(label_file)
#
#     missing_frac = missing / 100
#     observed_frac = 1 - missing_frac
#
#     count = 0
#     lambda_base = 0.5
#
#     for digit in digit_mod.keys():
#         print("Creating instances of digit {}".format(digit))
#
#         # read in the files
#         data_path = os.path.join(source, digit)
#         files = glob.glob('{}/*.jpg'.format(data_path))
#
#         # Assume requested files less than total available!
#         for i in range(digit_mod[digit]):
#
#             count += 1
#
#             original_image = plt.imread(files[i])
#             original_image_pad = np.pad(original_image, ((4, 4), (4, 4)), 'constant')
#
#             # decide on sickness
#             sick_var = np.random.binomial(1, sick_prob)
#
#             # irrelevant location
#             loc_var = np.random.binomial(1, 0.5)
#
#             # introduce some noise
#             rotations = np.random.normal(0, 2, len(time_points_rotations))
#
#
#             if digit == '3':
#                 gender = 0
#             else:
#                 gender = 1
#
#             curr_time = 0
#
#             timestamps_hawkes = []
#
#             time_age = []
#
#             normalized_for_rotation = []
#
#             for j in range(rotations.shape[0]):
#
#                 if sick_var:
#                     curr_time = hawkes_process(curr_time, timestamps_hawkes, lambda_base)
#                     timestamps_hawkes.append(curr_time)
#                     intensity = calculate_intensity(curr_time, timestamps_hawkes, lambda_base)
#                     normalized_for_rotation.append(normalized_rotation(intensity))
#
#                 else:
#                     curr_time += np.random.exponential(beta)
#                     normalized_for_rotation.append(0)
#
#                 time_age.append(curr_time)
#
#             normalized_for_rotation = np.array(normalized_for_rotation)
#
#             if sick_var:
#                 # simulate disease effect
#                 rotations += 45 * normalized_for_rotation
#             else:
#                 # baseline rotation for non-sick
#                 rotations += 5
#
#             time_with_zero = np.array([0] + time_age)
#
#             for idx, rotation in enumerate(rotations):
#
#                 # rotate an instance
#                 img = ndimage.rotate(original_image_pad, angle=rotation, reshape=False)
#
#                 # diagonal shift the image
#
#                 shift_time = min(200, time_age[idx])
#
#                 img = ndimage.shift(img, shift=shift_time * 2 / 200)
#
#                 if sick_var == 1:
#                     label_dict[sample_index] = \
#                         [subject_index, digit, rotation, sick_var, gender, time_age[idx],
#                          loc_var]
#                 elif sick_var == 0:
#                     label_dict[sample_index] = [subject_index, digit, rotation, sick_var, gender,
#                                                 time_age[idx], loc_var]
#
#                 rotated_MNIST = np.append(rotated_MNIST, np.reshape(img, (1, 1296)), axis=0)
#
#
#                 curr_mask = generate_mask_mnar_5(img, sick_var, normalized_for_rotation[idx], (1, 1296))
#
#                 mask = np.append(mask, curr_mask, axis=0)
#
#                 sample_index += 1
#
#             subject_index += 1
#
#             if i % 200 == 199:
#                 print("Instance no {} for digit {}".format(i + 1, digit))
#
#                 save_data(data_file, mask_file, data_masked_file, label_file,
#                           rotated_MNIST, label_dict, mask)
#                 rotated_MNIST = np.empty((0, 1296))
#                 mask = np.empty((0, 1296))
#                 label_dict = {}
#
#         save_data(data_file, mask_file, data_masked_file, label_file,
#                   rotated_MNIST, label_dict, mask)
#         rotated_MNIST = np.empty((0, 1296))
#         mask = np.empty((0, 1296))
#         label_dict = {}
#
#     print('Saved! Number of samples: {}'.format(sample_index))


import os
import glob
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy import ndimage
import matplotlib.pyplot as plt
import argparse
import math

"""
Code to generate the Health MNIST data.

This code manipulates the original MNIST images as described in the L-VAE paper.
"""


def parse_arguments():
    """
    Parse the command line arguments
    :return: parsed arguments object (2 arguments)
    """

    parser = argparse.ArgumentParser(description='Enter configuration for generating data')
    parser.add_argument('--source', type=str, default='./trainingSet', help='Path to MNIST image root directory')
    parser.add_argument('--destination', type=str, default='./data', help='Path to save the generated dataset')
    parser.add_argument('--num_3', type=int, default=50, help='Number of unique instances for digit 3')
    parser.add_argument('--num_6', type=int, default=50, help='Number of unique instances for digit 6')
    parser.add_argument('--missing', type=float, default=25, choices=range(-1, 101),
                        help='Percentage of missing in range [0, 100]')
    parser.add_argument('--data_file_name', type=str, default='health_MNIST_data.csv',
                        help='File name of generated data')
    parser.add_argument('--data_masked_file_name', type=str, default='health_MNIST_data_masked.csv',
                        help='File name of generated masked data')
    parser.add_argument('--labels_file_name', type=str, default='health_MNIST_label.csv',
                        help='File name of generated labels')
    parser.add_argument('--mask_file_name', type=str, default='mask.csv',
                        help='File name of generated mask')

    parser.add_argument('--D', type=int, default=15,
                        help='Number of previous time stamps')

    return vars(parser.parse_args())


def create_data_file(path, open_str):
    if os.path.exists(path):
        os.remove(path)
    return open(path, open_str)


def write_label_file_header_irregular(label_file, columns_of_prev_times):
    df = pd.DataFrame.from_dict({}, orient='index',
                                columns=['subject', 'digit', 'angle', 'disease', 'gender',
                                         'time_age', 'location'] + columns_of_prev_times)
    df.to_csv(label_file, index=False)


def save_data_irregular(data_file, mask_file, data_masked_file, label_file, rotated_MNIST, label_dict, mask):
    # save rotated MNIST
    np.savetxt(data_file, rotated_MNIST, fmt='%d', delimiter=',')

    # generate mask
    # mask = np.random.choice([0, 1], size=rotated_MNIST.shape, p=[missing_frac, observed_frac])

    # 0 implies missing, 1 implies observed
    masked_data = np.multiply(rotated_MNIST, mask)

    np.savetxt(data_masked_file, masked_data, fmt='%d', delimiter=',')
    np.savetxt(mask_file, mask, fmt='%d', delimiter=',')

    df = pd.DataFrame.from_dict(label_dict, orient='index',
                                columns=['subject', 'digit', 'angle', 'disease', 'gender',
                                         'time_age', 'location'])

    # save labels
    df.to_csv(label_file, index=False, header=False)


def write_label_file_header_regular(label_file):
    df = pd.DataFrame.from_dict({}, orient='index',
                                columns=['subject', 'digit', 'angle', 'disease',
                                         'disease_time', 'gender',
                                         'time_age', 'location'])
    df.to_csv(label_file, index=False)


def save_data_regular(data_file, mask_file, data_masked_file, label_file, rotated_MNIST, label_dict, mask):
    # save rotated MNIST
    np.savetxt(data_file, rotated_MNIST, fmt='%d', delimiter=',')

    # generate mask
    # mask = np.random.choice([0, 1], size=rotated_MNIST.shape, p=[missing_frac, observed_frac])

    # 0 implies missing, 1 implies observed
    masked_data = np.multiply(rotated_MNIST, mask)

    np.savetxt(data_masked_file, masked_data, fmt='%d', delimiter=',')
    np.savetxt(mask_file, mask, fmt='%d', delimiter=',')

    df = pd.DataFrame.from_dict(label_dict, orient='index',
                                columns=['subject', 'digit', 'angle', 'disease',
                                         'disease_time', 'gender',
                                         'time_age', 'location'])

    # save labels
    df.to_csv(label_file, index=False, header=False)


def generate_mask_mnar_irregular(image, sick_var, rotation,
                                 shape):  # for sick instances box increases in time, moving along the diagonal
    image = image / 255
    p = (image / 2) * 1.5
    mask = np.random.binomial(1, 1 - p, image.shape)
    if sick_var:
        size_mask = int(18 * rotation)
        half_size = int(size_mask / 2)
        center_y = image.shape[0] - size_mask
        center_x = size_mask
        mask[center_y - half_size: center_y + half_size, center_x - half_size:center_x + half_size] = 0

    mask = mask.reshape(shape)
    return mask


def generate_mask_mnar_regular(image, sick_var, time_point, shape):
    image = image / 255
    p = (image / 2) * 1.5
    mask = np.random.binomial(1, 1 - p, image.shape)
    if sick_var and int(time_point) > 0:
        size_mask = time_point * 2
        half_size = int(size_mask / 2)
        center_y = image.shape[0] - size_mask
        center_x = size_mask
        mask[center_y - half_size: center_y + half_size, center_x - half_size:center_x + half_size] = 0

    mask = mask.reshape(shape)
    return mask


def homogenous_poisson_process(mu, t_n_1):
    u = np.random.rand()
    t = -math.log(1 - u) / mu + t_n_1
    return t


def inhomogenous_poisson_process(g, t_n_1):
    g_max = 1
    t = t_n_1
    while True:
        t = homogenous_poisson_process(g_max, t)
        u = np.random.rand()
        if u <= g(t) / g_max:
            return t


def trigerring_kernel(t, h):
    kernel = 0
    for i in range(len(h)):
        if h[i] > t:
            break
        kernel += 0.5 * math.exp(-((t - h[i])) ** 2 / 0.5)

    return kernel


def hawkes_process(t_n_1, h, lambda_base):
    lambda_max = lambda_base + trigerring_kernel(t_n_1, h)
    t = t_n_1
    while True:
        t = homogenous_poisson_process(lambda_max, t)
        u = np.random.rand()
        if u <= (lambda_base + trigerring_kernel(t, h)) / lambda_max:
            return t


def calculate_intensity(t, h, lambda_base):
    intensity = lambda_base + trigerring_kernel(t, h)
    return intensity


def plot_intensity_hawkes(h):
    t = np.linspace(0, 40, 1000)
    lambda_base = 1
    lambd = []
    for el in t:
        lambd.append(lambda_base + trigerring_kernel(el, h))

    lambd_h = []
    for el in h:
        lambd_h.append(lambda_base + trigerring_kernel(el, h))

    plt.plot(t, lambd)
    plt.scatter(h[8:40], lambd_h[8:40], color="r")
    plt.savefig("./figures/original_intensity.png")
    plt.show()


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def normalized_rotation(intensity):
    return (tanh(intensity) - tanh(1)) * 4.5


# rotation based on intensity

def regular_timestamps():
    digit_mod = {'3': num_3, '6': num_6}
    sick_prob = 0.5  # probability of instance being sick
    sample_index = 0
    subject_index = 0
    label_dict = {}

    # 20 time points
    time_age = np.arange(0, 20)
    time_points = np.arange(-9, 11)

    # accumulate digits
    rotated_MNIST = np.empty((0, 1296))
    mask = np.empty((0, 1296))

    data_file = create_data_file(os.path.join(destination, data_file_name), "ab")
    mask_file = create_data_file(os.path.join(destination, mask_file_name), "ab")
    data_masked_file = create_data_file(os.path.join(destination, data_masked_file_name), "ab")
    label_file = create_data_file(os.path.join(destination, labels_file_name), "a")
    write_label_file_header_regular(label_file)

    missing_frac = missing / 100
    observed_frac = 1 - missing_frac

    count = 0

    for digit in digit_mod.keys():
        print("Creating instances of digit {}".format(digit))

        # read in the files
        data_path = os.path.join(source, digit)
        files = glob.glob('{}/*.jpg'.format(data_path))

        # Assume requested files less than total available!
        for i in range(digit_mod[digit]):

            count += 1

            original_image = plt.imread(files[i])
            original_image_pad = np.pad(original_image, ((4, 4), (4, 4)), 'constant')

            # decide on sickness
            sick_var = np.random.binomial(1, sick_prob)

            # irrelevant location
            loc_var = np.random.binomial(1, 0.5)

            # introduce some noise
            rotations = np.random.normal(0, 2, len(time_points))

            # define rotation for each instance
            if sick_var:
                # simulate disease effect
                rotations += 45 * sigmoid(time_points)
            else:
                # baseline rotation for non-sick
                rotations += 5

            if digit == '3':
                gender = 0
            else:
                gender = 1

            for idx, rotation in enumerate(rotations):

                # rotate an instance
                img = ndimage.rotate(original_image_pad, angle=rotation, reshape=False)

                # diagonal shift the image

                img = ndimage.shift(img, shift=idx / 10)

                if sick_var == 1:
                    label_dict[sample_index] = \
                        [subject_index, digit, rotation, sick_var, time_points[idx], gender, time_age[idx], loc_var]
                elif sick_var == 0:
                    label_dict[sample_index] = [subject_index, digit, rotation, sick_var, 'nan', gender,
                                                time_age[idx], loc_var]

                rotated_MNIST = np.append(rotated_MNIST, np.reshape(img, (1, 1296)), axis=0)

                curr_mask = generate_mask_mnar_regular(img, sick_var, time_points[idx], (1, 1296))

                mask = np.append(mask, curr_mask, axis=0)

                sample_index += 1

            subject_index += 1

            if i % 200 == 199:
                print("Instance no {} for digit {}".format(i + 1, digit))

                save_data_regular(data_file, mask_file, data_masked_file, label_file, rotated_MNIST, label_dict, mask)
                rotated_MNIST = np.empty((0, 1296))
                mask = np.empty((0, 1296))
                label_dict = {}

        save_data_regular(data_file, mask_file, data_masked_file, label_file, rotated_MNIST, label_dict, mask)
        rotated_MNIST = np.empty((0, 1296))
        mask = np.empty((0, 1296))
        label_dict = {}

    print('Saved! Number of samples: {}'.format(sample_index))


def irregular_timestamps():
    opt = parse_arguments()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    digit_mod = {'3': num_3, '6': num_6}
    sick_prob = 0.5  # probability of instance being sick
    sample_index = 0
    subject_index = 0
    label_dict = {}
    gender = 0
    beta = 10

    # 20 time points
    time_points_rotations = np.arange(0, 20)

    # accumulate digits
    rotated_MNIST = np.empty((0, 1296))
    mask = np.empty((0, 1296))

    columns_of_prev_times = []

    for d in range(D):
        columns_of_prev_times.append("t" + str(d + 1))

    for d in range(D):
        columns_of_prev_times.append("m" + str(d + 1))

    data_file = create_data_file(os.path.join(destination, data_file_name), "ab")
    mask_file = create_data_file(os.path.join(destination, mask_file_name), "ab")
    data_masked_file = create_data_file(os.path.join(destination, data_masked_file_name), "ab")
    label_file = create_data_file(os.path.join(destination, labels_file_name), "a")
    write_label_file_header_irregular(label_file, columns_of_prev_times)

    missing_frac = missing / 100
    observed_frac = 1 - missing_frac

    count = 0
    lambda_base = 0.5

    for digit in digit_mod.keys():
        print("Creating instances of digit {}".format(digit))

        # read in the files
        data_path = os.path.join(source, digit)
        files = glob.glob('{}/*.jpg'.format(data_path))

        # Assume requested files less than total available!
        for i in range(digit_mod[digit]):

            print(count)

            count += 1

            original_image = plt.imread(files[i])
            original_image_pad = np.pad(original_image, ((4, 4), (4, 4)), 'constant')

            # decide on sickness
            sick_var = np.random.binomial(1, sick_prob)

            # irrelevant location
            loc_var = np.random.binomial(1, 0.5)

            # introduce some noise
            rotations = np.random.normal(0, 2, len(time_points_rotations))

            if digit == '3':
                gender = 0
            else:
                gender = 1

            curr_time = 0

            timestamps_hawkes = []

            time_age = []

            normalized_for_rotation = []

            for j in range(rotations.shape[0]):

                if sick_var:
                    curr_time = hawkes_process(curr_time, timestamps_hawkes, lambda_base)
                    timestamps_hawkes.append(curr_time)
                    intensity = calculate_intensity(curr_time, timestamps_hawkes, lambda_base)
                    normalized_for_rotation.append(normalized_rotation(intensity))

                else:
                    curr_time += np.random.exponential(beta)
                    normalized_for_rotation.append(0)

                time_age.append(curr_time)

            normalized_for_rotation = np.array(normalized_for_rotation)

            if sick_var:
                # simulate disease effect
                rotations += 45 * normalized_for_rotation
            else:
                # baseline rotation for non-sick
                rotations += 5

            time_with_zero = np.array([0] + time_age)

            for idx, rotation in enumerate(rotations):
                # rotate an instance
                img = ndimage.rotate(original_image_pad, angle=rotation, reshape=False)

                # diagonal shift the image

                shift_time = min(200, time_age[idx])

                img = ndimage.shift(img, shift=shift_time * 2 / 200)

                if sick_var == 1:
                    label_dict[sample_index] = \
                        [subject_index, digit, rotation, sick_var, gender, time_age[idx],
                         idx]
                elif sick_var == 0:
                    label_dict[sample_index] = [subject_index, digit, rotation, sick_var, gender,
                                                time_age[idx], idx]

                rotated_MNIST = np.append(rotated_MNIST, np.reshape(img, (1, 1296)), axis=0)

                curr_mask = generate_mask_mnar_irregular(img, sick_var, normalized_for_rotation[idx], (1, 1296))

                mask = np.append(mask, curr_mask, axis=0)

                sample_index += 1

            subject_index += 1

            # if not i:
            #     plot_intensity_hawkes(timestamps)

            if i % 200 == 199:
                print("Instance no {} for digit {}".format(i + 1, digit))

                save_data_irregular(data_file, mask_file, data_masked_file, label_file,
                                    rotated_MNIST, label_dict, mask)
                rotated_MNIST = np.empty((0, 1296))
                mask = np.empty((0, 1296))
                label_dict = {}

        save_data_irregular(data_file, mask_file, data_masked_file, label_file,
                            rotated_MNIST, label_dict, mask)
        rotated_MNIST = np.empty((0, 1296))
        mask = np.empty((0, 1296))
        label_dict = {}

    print('Saved! Number of samples: {}'.format(sample_index))


if __name__ == '__main__':
    opt = parse_arguments()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)
    regular_timestamps()