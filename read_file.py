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
    np_z = np.array([ix[0] for ix in content],dtype=float)

    np_dose = np.array([ix[1] for ix in content], dtype = float)
    return np_z, np_dose

np_sim_dose , np_sim_dist = get_sim_data('Lydia_txt', coord={'x':-1,'y':-1,'z':0})
np_meas_dist, np_meas_dose = get_measured_data('MEAS_inplane_10x10_SSD100_depth16mm.txt')
#
# print(np_sim_dose)
# print(np_sim_dist)
# print(np_meas_dist)
# print(np_meas_dose)
#


np_sim_dist = [float(ix)*10 for ix in np_sim_dist]
# print(np_sim_dist)

# np_sim_dict = dict(zip(np_sim_dist,np_meas_dose))
# np_sim_dist =
# print(np_sim_dict)
np_sim_dist_clipped = [ix for ix in np_sim_dist if abs(ix) <82]
np_sim_dose_clipped = np_sim_dose[np_sim_dist.index(np_sim_dist_clipped[0]):np_sim_dist.index(np_sim_dist_clipped[-1])+1]

# print(np_sim_dist_clipped)
# print(np_sim_dose_clipped)
# print('MANGO')
# print(len(np_meas_dose))
# print(len(np_sim_dose_clipped))
# print(len(np_meas_dist))
# print(len(np_sim_dist_clipped))

def calc_gamma(sim_dose,sim_dist, meas_dose,meas_dist):

    diff_dist = np.subtract(sim_dist,meas_dist)
    diff_dose = np.subtract(sim_dose,meas_dose)

    diff_dist = [ix**2/(1.0**2) for ix in diff_dist]
    diff_dose = [ix ** 2/(0.05**2) for ix in diff_dose]

    gamma = np.add(diff_dist,diff_dose)
    gamma = [ix**(0.5) for ix in gamma]
    # print(len(gamma))
    return gamma

def calc_gamma_dose(sim_dose,sim_dist, meas_dose,meas_dist):

    diff_dist = np.subtract(sim_dist,meas_dist)
    diff_dose = np.subtract(sim_dose,meas_dose)

    diff_dist = [ix**2/(1.5**2) for ix in diff_dist]
    diff_dose = [ix ** 2/(0.05**2) for ix in diff_dose]

    gamma = np.add(diff_dist,diff_dose)
    gamma = [ix**(0.5) for ix in gamma]
    # print(len(gamma))
    return gamma

def linear_interpolate(sim_dist,sim_dose,meas_dist):
    print('ELEPHANT')
    # print(meas_dist)
    new_sim_dose = []
    new_sim_dist = []
    for inx in meas_dist:
        # print(sim_dist)
        # print(meas_dist[inx])
        temp = [abs(ix-inx) for ix in sim_dist]
        # print(sim_dist)
        # print(meas_dist)
        # print(temp)

        closest_inx = temp.index(min(temp))
        a = sim_dose[closest_inx]
        x = sim_dist[closest_inx]
        if closest_inx +1 < len(sim_dose):
            b = sim_dose[closest_inx+1]
            y = sim_dist[closest_inx+1]
        else:
            b = sim_dose[-1]
            y = sim_dist[-1]


        if y-x != 0:
            slope = (b-a)/(y-x)

            print(inx)
            print('closest inx' + str(closest_inx))
            print('slope' + str(slope))
            #
            # print('YAYYAA')
            # print(x,y,a,b)
            # print(slope)
            sim_interpolated_meas = inx*slope
            print('slope' + str(slope))
            new_sim_dose = new_sim_dose + [sim_interpolated_meas]
            new_sim_dist = new_sim_dist + [inx]

        else:
            new_sim_dose = new_sim_dose + [new_sim_dose[-1]]
            new_sim_dist = new_sim_dist + [inx]

    new_sim_dist = np.around(new_sim_dist,decimals=2)
    print('MANGOOOOOO')
    # print(new_sim_dose)
    print(len(new_sim_dose))
    # print(new_sim_dist)
    # print(len(new_sim_dist))
    return new_sim_dist,new_sim_dose , meas_dist


def normalize_dose(np_dose, high=1.0, low=0.0):
    # np_dose /= np.max(np.abs(np_dose), axis=0)
    # # print('loooo')
    # # print(np.max(np.abs(np_dose), axis=0))
    # # print(np_dose)
    # return np_dose
    mins = np.min(np_dose, axis=0)
    maxs = np.max(np_dose, axis=0)
    rng = maxs - mins
    print('rng:   ... ' + str(maxs))
    print('rngffdsf:   ... ' + str(mins))
    return high - (((high - low) * (maxs - np_dose)) / rng)


# new_sim_dist,new_sim_dose,meas_dist = linear_interpolate(np_sim_dist_clipped,np_sim_dose_clipped,np_meas_dist)
# print('HEY')
# print(new_sim_dose)
np_meas_dose =normalize_dose(np_meas_dose)
# print(np_meas_dose)
# print('BLING')
# print(new_sim_dose)
np_sim_dose_clipped = normalize_dose(np_sim_dose_clipped)
# print (np_sim_dose_clipped)
# print(np_sim_dose_clipped.dtype)
# print('BANANA')

# print(new_sim_dose)
# print(np_meas_dose)

# print(new_sim_dose)



import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

sim = interp1d(np_sim_dist_clipped,np_sim_dose_clipped,kind='cubic' )
meas = interp1d(np_meas_dist,np_meas_dose,kind='cubic')

print(np_meas_dose)
# print(np_meas_dose.dtype)

weird_val = [ix for ix in np_meas_dose if ix < 0.0]

print(weird_val)
print(np_meas_dist)

fresh_x = np_meas_dist[:len(np_meas_dist)/2]
fresh_y = np_meas_dose[:len(np_meas_dose)/2]
print('FRESHHH')
print(len(fresh_x))
print(len(fresh_y))
sim_gamma_dose = interp1d(np_sim_dose_clipped[:len(np_sim_dose_clipped)/2],np_sim_dist_clipped[:len(np_sim_dist_clipped)/2],kind='linear' )
meas_gamma_dose = interp1d(np_meas_dose[:len(np_meas_dose)/2],np_meas_dist[:len(np_meas_dist)/2],kind='linear')
print(np_meas_dose[:len(np_meas_dose)/2])
xnew = np.linspace(-80, 80, num=160, endpoint=True)
np_meas_dose = np.around(np_meas_dose, decimals= 5)
np_meas_dose = np.unique(np_meas_dose)
x_new_dose = np.linspace(np.min(np_meas_dose), np.max(np_meas_dose),100, endpoint=True)
print(x_new_dose)
# print(x_new_dose)
sim_inter = sim(xnew)
meas_inter = meas(xnew)
print(np_sim_dist_clipped)
sim_inter_dose = sim_gamma_dose(x_new_dose)
print(sim_inter_dose)
meas_inter_dist = meas_gamma_dose([ix for ix in x_new_dose if 0<ix<1])
fresh_dose = [ix for ix in x_new_dose if 0<ix<1]
print(len(fresh_dose))
print(len(meas_inter_dist))

sim_inter_dist = sim_gamma_dose([ix for ix in x_new_dose if 0<ix<1])
fresh_dose = [ix for ix in x_new_dose if 0.0<ix<1]
print(len(fresh_dose))
print(len(sim_inter_dist))
# gamma_dist = calc_gamma(sim_inter,xnew,meas_inter,xnew)
gamma_dose = calc_gamma(fresh_dose, meas_inter_dist, fresh_dose, sim_inter_dist)
print(len(gamma_dose))
# plt.plot(x_new_dose, meas_inter_dose, '-')
# plt.plot(fresh_dose, sim_inter_dist, '-', fresh_dose, meas_inter_dist, '--', fresh_dose , gamma_dose , '.')
#
# print(gamma_dose)
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# # plt.show()
# sim_inter_dist = normalize_dose(sim_inter_dist,1,0)
# meas_inter_dist = normalize_dose(meas_inter_dist,1,0)
plt.plot(fresh_dose, sim_inter_dist, '-', fresh_dose, meas_inter_dist, '--', fresh_dose , gamma_dose , '.')
plt.legend(['cubic'], loc='best')
plt.show()

# # gamma_dose = calc_gamma_dose()
#
# print('GAMMMMA' ,gamma, len(gamma))
#
# print('SIM INTER : ... ' , sim_inter, len(sim_inter))
# print('MEAS INTER: ... ' , meas_inter, len(meas_inter))
# print(xnew)
# plt.plot(xnew, sim_inter, '-', xnew, meas_inter, '--', xnew , gamma , 'o')
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# plt.show()
# fig,ax = plt.subplots()
# s = 121
# # print(new_sim_dist)
# # print(new_sim_dose)
# ax.scatter(np_sim_dist_clipped, np_sim_dose_clipped, color='r', s=s/10, alpha=.4)
# ax.scatter(np_meas_dist, np_meas_dose, color='b', s=s/10, alpha=.4)
# # ax.scatter(meas_dist,gamma, color='g', s=s/10, marker='s', alpha=.4)
# plt.show()



