import numpy as np
import h5py


def get_one_col_of_phsp(col_name,start=0,stop=-1):
    """
    :param col_name: the name of the coloumn to retrieve eg: px or energy
    :param start: the offset index of the coloumn. default = 0
    :param stop: the stop index of the coloumn. default = -1
    :return: a numpy array of complete coloumn
    """


    h1 = h5py.File("../phsp/chunk_aa.h5" , "r")
    h2 = h5py.File("../phsp/chunk_ab.h5" , "r")
    h3 = h5py.File("../phsp/chunk_ac.h5" , "r")
    h4 = h5py.File("../phsp/chunk_ad.h5" , "r")
    h5 = h5py.File("../phsp/chunk_ae.h5" , "r")
    h6 = h5py.File("../phsp/chunk_af.h5" , "r")
    h7 = h5py.File("../phsp/chunk_ag.h5" , "r")
    h8 = h5py.File("../phsp/chunk_ah.h5" , "r")
    h9 = h5py.File("../phsp/chunk_ai.h5" , "r")
    h10= h5py.File("../phsp/chunk_aj.h5" , "r")

    print(len(h1[col_name][:]))
    print(h1[col_name][:])

    e = np.concatenate((h1[col_name][start:stop],h2[col_name][start:stop],h3[col_name][start:stop],h4[col_name][start:stop],h5[col_name][start:stop],h6[col_name][start:stop],h7[col_name][start:stop]
                        ,h8[col_name][start:stop],h9[col_name][start:stop],h10[col_name][start:stop]), axis=0)

    return e


def calc_gamma_dose(sim_dose, sim_dist, meas_dose, meas_dist):
    diff_dist = np.subtract(sim_dist, meas_dist)
    diff_dose = np.subtract(sim_dose, meas_dose)

    diff_dist = [ix ** 2 / (1.5 ** 2) for ix in diff_dist]
    diff_dose = [ix ** 2 / (0.05 ** 2) for ix in diff_dose]

    gamma = np.add(diff_dist, diff_dose)
    gamma = [ix ** (0.5) for ix in gamma]
    # print(len(gamma))
    return gamma

def get_sim_data(file_path, coord={'x': 0, 'y': 0, 'z': 0}, clipped = False):
    with open(file_path) as f:
        content = f.readlines()

    x_start_index = coord['x']
    y_start_index = coord['y']
    z_start_index = coord['z']
    dict_content = {'np_x': 0, 'np_y': 0, 'np_z': 0}
    content = [x.strip() for x in content]
    content = [ix.split(",") for ix in content]
    inx = 0

    print(coord)
    for key, val in coord.iteritems():
        if val != -1:
            print(key)
            print(val)
            dict_content['np_' + key] = np.array([ix[inx] for ix in content[val:]], dtype=float)
            inx += 1
            dose_start_index = val

            dose_start_index = z_start_index
            np_dose = np.array([ix[inx] for ix in content[dose_start_index:]], dtype=float)
    if clipped:
        np_sim_dist = [float(ix) * 10 for ix in dict_content['np_z']]
        np_sim_dist_clipped = [ix for ix in np_sim_dist if abs(ix) < 82]
        np_sim_dose_clipped = np_dose[np_sim_dist.index(np_sim_dist_clipped[0]):np_sim_dist.index(np_sim_dist_clipped[-1]) + 1]
        print('banan')
        return np_sim_dose_clipped, np_sim_dist_clipped
    return np_dose, dict_content['np_z']

def get_measured_data(file_path, start_index=0):
    with open(file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [ix.split('\t\t') for ix in content]
    np_z = np.array([ix[0] for ix in content], dtype=float)

    np_dose = np.array([ix[1] for ix in content], dtype=float)
    return np_dose,np_z

def normalize_dose(np_dose, high=1.0, low=0.0):
    # np_dose /= np.max(np.abs(np_dose), axis=0)
    # # print('loooo')
    # # print(np.max(np.abs(np_dose), axis=0))
    # # print(np_dose)
    # return np_dose
    mins = np.min(np_dose, axis=0)
    maxs = np.max(np_dose, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - np_dose)) / rng)
        # print(len(e))
# print(np.asarray(e))
# import matplotlib.pyplot as plt
# n_bins = 100000
#
# plt.hist(e, n_bins, histtype='step', stacked=False, fill=False)
# plt.show()
# print(e)
# get_one_col_of_phsp("energy")