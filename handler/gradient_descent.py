import numpy as np
import helper_functions
from scipy import stats
import plot_and_calc_cost
import stat

# np_energy = helper_functions.get_one_col_of_phsp(file_path="/project/med/MAPDOSI/Rangoli.Saxena/pdd_sim", col_name="energy")
# np_px = helper_functions.get_one_col_of_phsp("px")
# np_py = helper_functions.get_one_col_of_phsp("py")
# np_pz = helper_functions.get_one_col_of_phsp("pz")


#retreiving data from txt files
# sim_dose , sim_dist = helper_functions.get_sim_data("../Lydia_txt", coord={'x':-1,'y':-1,'z':0}, clipped=True)
# meas_dose, meas_dist = helper_functions.get_measured_data('../MEAS_inplane_10x10_SSD100_depth16mm.txt')

f_path_meas = "/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_meas/PDD_2x2_measurements.txt"

sim_dist, sim_dose, meas_dist, meas_dose = plot_and_calc_cost.pdd_get_sim_meas_and_plot(folder='/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/',f_path_meas=f_path_meas)
#normalising dose
sim_dose = helper_functions.normalize_dose(sim_dose)
meas_dose = helper_functions.normalize_dose(meas_dose)

#focussing on the linear part
sim_dose_lin = [ix for ix in sim_dose if 0.2<ix<0.8 and list(sim_dose).index(ix)<len(sim_dose)/2]
sim_dist_lin = [sim_dist[list(sim_dose).index(ix)] for ix in sim_dose_lin]
meas_dose_lin = [ix for ix in meas_dose if 0.2<ix<0.8 and list(meas_dose).index(ix)<len(meas_dose)/2]
meas_dist_lin = [meas_dist[list(meas_dose).index(ix)] for ix in meas_dose_lin]
#
sim_dose_lin = [0.20394088669950727, 0.43694581280788172, 0.5814285714285714, 0.79522167487684735]
sim_dist_lin = [-51.829999999999998, -50.82, -49.799999999999997, -49.380000000000001]
slope, intercept, r_value, p_value, std_err = stats.linregress(meas_dist_lin,meas_dose_lin)
abline_values = [slope * i + intercept for i in meas_dist_lin]
print(std_err)
print(r_value)
print(p_value)
import matplotlib.pyplot as plt

plt.scatter(meas_dist_lin,meas_dose_lin)
plt.plot(meas_dist_lin,abline_values)

slope, intercept, r_value, p_value, std_err = stats.linregress(sim_dist_lin,sim_dose_lin)
ablin_values = [slope * i + intercept for i in sim_dist_lin]
print(std_err)
print(r_value)
print(p_value)

plt.scatter(sim_dist_lin,sim_dose_lin)
plt.plot(sim_dist_lin,ablin_values)
plt.show()
print(sim_dose_lin,sim_dist_lin,meas_dose_lin,meas_dist_lin)

#
# nxt_possible_energy = np.subtract(np_energy,0.1)
# print(nxt_possible_energy)
# median = np.median(np_energy)

