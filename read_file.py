import numpy as np


def get_sim_data(file_path, coord={'x':0,'y':0,'z':0}):
    with open(file_path) as f:
        content = f.readlines()

    x_start_index = coord['x']
    y_start_index = coord['y']
    z_start_index = coord['z']
    dict_content = {'np_x':0,'np_y':0,'np_z':0}
    content = [x.strip() for x in content]
    content = [ix.split(",") for ix in content]
    inx = 0

    print(coord)
    for key,val in coord.iteritems():
        if val !=-1:
            print(key)
            print(val)
            dict_content['np_' + key]= np.array([ix[inx] for ix in content[val:]],dtype = float)
            inx +=1
            dose_start_index = val

            dose_start_index = z_start_index
            np_dose = np.array([ix[inx] for ix in content[dose_start_index:]],dtype = float)
    return np_dose,dict_content['np_z']


def get_measured_data(file_path,start_index=0):
    with open(file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [ix.split('\t\t') for ix in content]
    np_z = np.array([ix[0] for ix in content])

    np_dose = np.array([ix[1] for ix in content], dtype = float)
    return np_z, np_dose

np_sim_dose , np_sim_dist = get_sim_data('Lydia_txt', coord={'x':-1,'y':-1,'z':0})
np_meas_dist, np_meas_dose = get_measured_data('MEAS_inplane_10x10_SSD100_depth16mm.txt')

print(np_sim_dose)
print(np_sim_dist)
print(np_meas_dist)
print(np_meas_dose)



np_sim_dist = [float(ix)*10 for ix in np_sim_dist]
print(np_sim_dist)

# np_sim_dict = dict(zip(np_sim_dist,np_meas_dose))
# np_sim_dist =
# print(np_sim_dict)
np_sim_dist_clipped = [ix for ix in np_sim_dist if abs(ix) <82]
np_sim_dose_clipped = np_sim_dose[np_sim_dist.index(np_sim_dist_clipped[0]):np_sim_dist.index(np_sim_dist_clipped[-1])]

print(np_sim_dist_clipped)
print(np_sim_dose_clipped)
print('MANGO')
print(len(np_meas_dose))
print(len(np_sim_dose_clipped))

def calc_gamma(sim_dose,sim_dist, meas_dose,meas_dist):

    diff_dist = np.subtract(sim_dist,meas_dist)
    diff_dose = np.subtract(sim_dose,meas_dose)

    diff_dist = [ix**2/(1.5**2) for ix in diff_dist]
    diff_dose = [ix ** 2/(0.02**2) for ix in diff_dose]

    gamma = np.add(diff_dist,diff_dose)
    gamma = [ix**(0.5) for ix in gamma]

    print(gamma)


# calc_gamma(np_sim_dose_clipped,np_sim_dist_clipped,np_meas_dose,np_meas_dist)