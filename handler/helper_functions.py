import numpy as np
import h5py


def get_one_col_of_phsp(file_path,col_name,start=0,stop=-1):
    """
    :param col_name: the name of the coloumn to retrieve eg: px or energy
    :param start: the offset index of the coloumn. default = 0
    :param stop: the stop index of the coloumn. default = -1
    :return: a numpy array of complete coloumn
    """


    h1 = h5py.File(file_path+"/phsp/chunk_aa.h5" , "r")
    h2 = h5py.File(file_path+"/phsp/chunk_ab.h5" , "r")
    h3 = h5py.File(file_path+"/phsp/chunk_ac.h5" , "r")
    h4 = h5py.File(file_path+"/phsp/chunk_ad.h5" , "r")
    h5 = h5py.File(file_path+"/phsp/chunk_ae.h5" , "r")
    h6 = h5py.File(file_path+"/phsp/chunk_af.h5" , "r")
    h7 = h5py.File(file_path+"/phsp/chunk_ag.h5" , "r")
    h8 = h5py.File(file_path+"/phsp/chunk_ah.h5" , "r")
    h9 = h5py.File(file_path+"/phsp/chunk_ai.h5" , "r")
    h10= h5py.File(file_path+"/phsp/chunk_aj.h5" , "r")

    print(len(h1[col_name][:]))
    print(h1[col_name][:])

    e = np.concatenate((h1[col_name][start:stop],h2[col_name][start:stop],h3[col_name][start:stop],h4[col_name][start:stop],h5[col_name][start:stop],h6[col_name][start:stop],h7[col_name][start:stop]
                        ,h8[col_name][start:stop],h9[col_name][start:stop],h10[col_name][start:stop]), axis=0)

    return e


def calc_gamma_dose(sim_dose, sim_dist, meas_dose, meas_dist):
    """

    :param sim_dose: simulated dose
    :param sim_dist: simulated distances
    :param meas_dose: measured dose
    :param meas_dist: measured distances
    :return: the gamma at the distances. requires same distances
    """
    diff_dist = np.subtract(sim_dist, meas_dist)
    diff_dose = np.subtract(sim_dose, meas_dose)

    diff_dist = [ix ** 2 / (1.5 ** 2) for ix in diff_dist]
    diff_dose = [ix ** 2 / (0.05 ** 2) for ix in diff_dose]

    gamma = np.add(diff_dist, diff_dose)
    gamma = [ix ** (0.5) for ix in gamma]
    # print(len(gamma))
    return gamma

def get_coloumns_from_txt(file_path, cols=None):

    """
    used for PDD
    :param file_path: path to simulated file
    :param coord: 1 is 1st col, 2 is 2nd..
    :param clipped:clipped acc to dist
    :return:all coloumns
    """

    dict_content = {}

    with open(file_path) as f:
        content = f.readlines()

    content = [x.strip().split(",") for x in content]
    for key, val in cols.iteritems():
        if val != -1:
            dict_content['np_' + key] = np.array([ix[int(key)-1] for ix in content[val:]], dtype=float)

    return dict_content

def get_measured_data(file_path, start_index=0):
    with open(file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content[start_index:]]
    content = [ix.split('\t') for ix in content]
    np_z = np.array([ix[0] for ix in content[:-1]], dtype=float)
    # raise()
    np_dose = np.array([ix[1] for ix in content[:-1]], dtype=float)
    return np_dose,np_z

def normalize_dose(np_dose, high=None, low=0.0):
    """

    :param np_dose: the dose
    :param high: highest dose after normalisation
    :param low: lowest dose after normalisation
    :return: normalised dose
    """
    print(high)

    if high is None:
        high = np.max(np_dose)

    return np_dose/high
    # mins = np.min(np_dose, axis=0)
    # maxs = np.max(np_dose, axis=0)
    # rng = maxs - mins
    # return high - (((high - low) * (maxs - np_dose)) / rng)

        # print(len(e))
# print(np.asarray(e))
# import matplotlib.pyplot as plt
# n_bins = 100000
#
# plt.hist(e, n_bins, histtype='step', stacked=False, fill=False)
# plt.show()
# print(e)
# get_one_col_of_phsp("energy")