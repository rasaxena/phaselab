import numpy as np

import helper_functions
# import matplotlib
# # matplotlib.use('Agg')

# <something>_cols is a dictionary which keys means the nth coloumn number and value denotes the index when the senseful data start

def pdd_get_sim_meas_and_plot(folder='None',IAEApart='3_', profile ='PDD_', version ='0.txt',fs=None,f_path_sim=None, f_path_meas=None,path=None):


    """

    :param fs_list: the list of field sizes
    :param plot: if true then it plots the values and interpolates it cubically as well
    :param path: paths can be customized according to the field size to save time
    :return: 4 np_arrays: sim_dist, sim_dose, meas_dist, meas_dose
    """

    sim_dist_all_fs = []
    sim_dose_all_fs = []
    meas_dose_all_fs = []
    meas_dist_all_fs = []
    #get simulation coloumns

    sim1_cols = helper_functions.get_col_from_PDD_text(folder, cols={'1':3, '2':3, '3':3, '4':3})

    # sim1_cols = helper_functions.get_col_from_PDD_text(folder + IAEApart + profile + fs + version, cols={'1':3, '2':3, '3':3, '4':3})
    # sim2_cols = helper_functions.get_col_from_PDD_text(folder + IAEApart + profile + fs + version, cols={'1':3, '2':3, '3':3, '4':3})
    # sim3_cols = helper_functions.get_col_from_PDD_text(folder + IAEApart + profile + fs + version, cols={'1':3, '2':3, '3':3, '4':3})
    # sim4_cols = helper_functions.get_col_from_PDD_text(folder + IAEApart + profile + fs + version, cols={'1':3, '2':3, '3':3, '4':3})

    #get measurement coloumns
    meas_dose,meas_dist = helper_functions.get_meas_PDD(f_path_meas + fs + 'measurements.txt')

    #get sim dist for all field sizes
    sim_dist = sim1_cols['np_2']
    sim_dist_all_fs = np.array(sim_dist_all_fs + [sim_dist])

    # get sim dose for all field sizes and flip it
    # sim_dose = np.flipud(np.array([sim1_cols['np_4'][ix]+sim2_cols['np_4'][ix]+
    #                                sim3_cols['np_4'][ix]+ sim4_cols['np_4'][ix] for ix in range(len(sim4_cols['np_4']))]))
    # sim_dose_all_fs = np.array(sim_dose_all_fs + [
    #     helper_functions.normalize_dose(sim_dose, high=np.mean(sim_dose[14:18]))])

    # print(sim1_cols)
    sim_dose = np.flipud(np.array(sim1_cols['np_4']))
    sim_dose_all_fs =  helper_functions.normalize_dose(sim_dose, high=np.mean(sim_dose[15:18]))

    # raise()

    #get meas dist for all field sizes
    meas_dist_all_fs = np.array(meas_dist_all_fs + [meas_dist])

    #get meas dose for all field sizes
    meas_dose = helper_functions.normalize_dose(meas_dose)
    meas_dose_all_fs = np.array(meas_dose_all_fs + [meas_dose])

    return sim_dist_all_fs, sim_dose_all_fs, meas_dist_all_fs, meas_dose_all_fs


def fit_exponential_curve(x, a, b,c):
    return a * np.exp(-b * x)+c

def fit_exponential_curve_2(x, a1, a2,b1,b2,c):
    # print("a1",a1,"a2",a2, "b1", b1, "b2",b2, "c",c)
    return (a1**3.0) * np.exp(-1.5*b1 * x) - a2**1.5* np.exp(-2.5*b2 * x)+c*x
    return (a1) * np.exp(-b1 * x) - a2* np.exp(-b2 * x)

from scipy import optimize


import matplotlib.pyplot as plt

def plot_curve(plt, fit_curve_dist,fit_curve_dose, c,*popt):
    plt.scatter(fit_curve_dist,fit_curve_dose, c=c)
    # raise()
    plt.plot(fit_curve_dist, fit_exponential_curve_2(fit_curve_dist, *popt), c=c)
    return plt

def get_popt_poptm_diff(sim_dist=None, sim_dose=None, meas_dist=None, meas_dose=None, folder=None, f_path_meas =None, fs=None, plot=False):

    sim_dist_all_fs, sim_dose_all_fs, meas_dist_all_fs, meas_dose_all_fs = pdd_get_sim_meas_and_plot(folder,f_path_meas =f_path_meas, fs=fs)

    index_of_max_sim_dose = np.where(sim_dose_all_fs[0] == np.max(sim_dose_all_fs[0]))
    index_of_max_meas_dose = np.where(meas_dose_all_fs[0] == np.max(meas_dose_all_fs[0]))

    index_of_max_dist = np.where(sim_dist_all_fs[0] == np.max(meas_dist_all_fs[0]))
    # print("MAX INDEX" , index_of_max_dist)

    # print(index_of_max_dist[0])
    # fit_curve_dose = np.array(sim_dose_all_fs[0][:index_of_max_dist[0][0]])
    # fit_curve_dist = np.array(sim_dist_all_fs[0][:index_of_max_dist[0][0]])

    print(index_of_max_dist[0])
    fit_curve_dose = np.array(sim_dose_all_fs[2:300])
    fit_curve_dist = np.array(sim_dist_all_fs[0][2:300])

    fit_curve_dose_meas = np.array(meas_dose_all_fs[0][0:len(meas_dose_all_fs[0])])
    fit_curve_dist_meas = np.array(meas_dist_all_fs[0][0:len(meas_dist_all_fs[0])])

    popt, pconv = optimize.curve_fit(fit_exponential_curve_2, fit_curve_dist, fit_curve_dose)
    poptm, pconvm = optimize.curve_fit(fit_exponential_curve_2, fit_curve_dist_meas, fit_curve_dose_meas)

    #plot the scatter and curve
    # raise()
    if plot:
        plot_curve(plt, fit_curve_dist_meas,fit_curve_dose_meas,'g',*poptm)
        plot_curve(plt, fit_curve_dist,fit_curve_dose,"r",*popt)
        # plt.fill_between()
        plt.xlabel('Depth in Y')
        plt.ylabel('Normalised Dose')
        plt.title(folder)
    plt.show()
    print("POPT",popt)
    print("POPTM",poptm)

    diff = [(popt[inx]-poptm[inx])/poptm[inx] for inx in range(len(popt))]
    # print(diff)


    return popt, poptm, diff


# save = open('save2x' ,'wb')
# with open("opt_param_results" , 'wb') as f:
# ob = []
# f.write('\n\nCurrent Differences in parameters. Date and time: ' +
#         datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '\n\n')


    #     f.write('FIELD SIZE :'+fs + '  \n')
    #     f.write('popt : '+str(popt) + '\n')
    #     f.write('poptm : '+str(poptm) + '\n')
    #     f.write('diff : '+str(diff) + '\n\n\n')
    #     ob = ob + [[popt]+[poptm]+[diff]]
    # print(ob)
    # pickle.dump(ob,save)
    # save.close()


def calc_G(poptm,popt,p):


    wiggling = np.linspace(popt[p],poptm[p], 100)
    del_p = wiggling[0]-wiggling[1]
    J =[]
    for wiggled_param in wiggling:
        popt[p] = wiggled_param
        J = J + [calc_J(poptm,popt)]
    G = [(J[ix+1] - J[ix])/del_p for ix in range(len(J[:-1]))]
    # plt.scatter(wiggling, J, c = 'g')
    plt.scatter(wiggling[:-1],G)
    # J = np.sum(J)/9999

    # plt.show()
    print(G)



def calc_J(poptm, popt):

    x = np.linspace(2,298,2980)

    opt = fit_exponential_curve_2(x,popt[0],popt[1],popt[2],popt[3],popt[4])
    optm = fit_exponential_curve_2(x,poptm[0],poptm[1],poptm[2],poptm[3],poptm[4])

    J = np.absolute(np.subtract(opt,optm))
    J = np.sum(J)/2980

    # raise()
    return J

folder='/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/'
f_path_meas = "/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_meas/PDD_"


fs_list = ["5x_"]
for ii, fs in enumerate(fs_list):
    popt, poptm, diff = get_popt_poptm_diff(folder='/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/Profiles_meas/Crossline/5x5_Measured_Cross.txt', f_path_meas=f_path_meas, fs=fs, plot=True)
    popt, poptm, diff = get_popt_poptm_diff(folder='/home/knossos/garmnt/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/Profiles_Sim/profiles_v1.txt', f_path_meas=f_path_meas, fs=fs, plot=True)
    #
    # calc_G(poptm,popt,0)
    # print(calc_J(poptm, popt))


# (1.030830120^3.5) * exp(-1.5*0.00440850385* x) - 0.783567450^0.5*exp(-2.5*0.103431740 * x)+0.0000492084402*x
# (1.03567040^3.5) * exp(-1.5* 0.00439332405* x) -  0.72665992^0.5*exp(-2.5*0.0925965056 * x)+0.0000638067288*x
# (1.03567040^3.5) * exp(-1.5* 0.00439332405* x) -  0.72665992^0.5*exp(-2.5*0.0925965056 * x)+0.0000638067288*x  - ((1.030830120^3.5) * exp(-1.5*0.00440850385* x) -
    # 0.783567450^0.5*exp(-2.5*0.103431740 * x)+0.0000492084402*x)

# "/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/pdd_4_6_exp3_3p0.txt
# /project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/pdd_6_8_exp3_3p0.txt
# /project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/pdd_10_15_mm_exp3_3p0.txt
# /project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/pdd_15_20_mm_exp3_3p0.txt
# /project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/pdd_4_6_mm_exp3_0p1.txt
# /project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/pdd_0_2_exp3_3p0.txt
# /project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/pdd_8_10_mm_exp3_0p1.txt
# /project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/pdd_6_8_mm_exp3_0p1.txt
# /project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/pdd_0_5_mm_exp3_3p0.txt
# /project/med/MAPDOSI/Rangoli.Saxena/PSFMan/pdd_sim/pdd_2_4_mm_exp3_0p1.txt"