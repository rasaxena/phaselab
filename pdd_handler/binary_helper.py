import numpy as np
# /project/med/MAPDOSI/Rangoli.Saxena/
import io
# bin_file =
# bin_file = "/home/r/Rangoli.Saxena/Downloads/fake.bin"
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from fast_histogram import histogram1d, histogram2d


def plot_histogram(bin_path, val_pos=1,dtype='float32'):

    # energy = np.zeros((int(126491813/10),))
    # x = np.zeros((int(126491813 / 10),))
    # y = np.zeros((int(126491813 / 10),))
    # energyp = np.zeros((int(126491813/10),))
    # energym = np.zeros((47026533,))
    with io.open('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_0.bin','rb') as f:

        noOfParticles = len(f.read()) / 60
        x = np.zeros((int(noOfParticles / 10),))
        y = np.zeros((int(noOfParticles / 10),))
        energy = np.zeros((int(noOfParticles / 10),))
        print(noOfParticles)

        print(len(energy))
        f.seek(0)
        for i in range(int(noOfParticles/10)):
            # val_pos = 28 for energy
            f.seek(28 + (i * 60))
            # data = f.read(8)
            # xi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            # data = f.read(8)
            # yi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            # data = f.read(8)
            # zi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            data = f.read(8)
            energyi = np.frombuffer(data, dtype="float64", offset=0, count=1)
            energy[i] = energyi
            # x[i] = xi
            # y[i] = yi
    print("energy",energy[:10])






    with io.open(bin_path, 'rb') as f:
        noOfParticles = len(f.read()) / 33
        x = np.zeros((int(noOfParticles / 10),))
        y = np.zeros((int(noOfParticles / 10),))
        energy10 = np.zeros((int(noOfParticles / 10),))
        print(noOfParticles)



        print(len(energy))
        f.seek(0)
        for i in range(int(noOfParticles/10)):
            # val_pos = 28 for energy
            f.seek(val_pos + (i * 33))
            data = f.read(4)
            # xi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            # data = f.read(8)
            # yi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            # data = f.read(8)
            # zi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            # data = f.read(8)
            energyi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            energy10[i] = abs(energyi)
            # x[i] = xi
            # y[i] = yi
    print("energy",energy[:10])
    # print("x",x[:10])
    # print("y",y[:10])

    # plt.hist(energy, 1000)
    # plt.show()


    # with io.open('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_IAEA_PSF_p_0x10_0.bin','rb') as f:
    #     noOfParticles = len(f.read()) / 33
    #     print(noOfParticles)
    #     # energyp = np.zeros((int(noOfParticles/2),))
    #     print(len(energyp))
    #     f.seek(0)
    #     for i in range(int(noOfParticles/10)):
    #         # val_pos = 28 for energy
    #         f.seek(val_pos + (i * 33))
    #         data = f.read(4)
    #         y = np.frombuffer(data, dtype='float32', offset=0, count=1)
    #         energyp[i] = y
    # print(energyp[:10])

    # with io.open('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_m_0x01_0.bin','rb') as f:
    #     noOfParticles = len(f.read()) / 60
    #     print(noOfParticles)
    #     # energym = np.zeros((int(noOfParticles/2),))
    #     print(len(energym))
    #     f.seek(0)
    #     for i in range(int(noOfParticles/2)):
    #         # val_pos = 28 for energy
    #         f.seek(val_pos + (i * 60))
    #         data = f.read(8)
    #         y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
    #         energym[i] = y
    # print(energym[:10])
    bins = np.linspace(0, 11, 1000)
    #
    plt.hist(energy, bins, alpha=0.3, label='5.75 Mev')
    plt.hist(energy10, bins, alpha=0.3, label='10 Mev')

    # plt.hist2d(y, energy, bins=400, norm=LogNorm())
    # plt.colorbar()
    # plt.show()
    # plt.hist(energym, bins, alpha = 0.3, label = 'minus 0.01')
    # plt.legend(loc='upper right')
    plt.show()

# plot_histogram('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_0.bin')
# plot_histogram('/home/r/Rangoli.Saxena/Downloads/ELEKTA_PRECISE_10mv_part1.IAEAphsp')

def manipulate_PSF(bin_path=None, val_pos=12,delta=0, dtype='float64',min=None, max=None, fresh=False, IAEA=False):

    """
    :param bin_path: path for the binary file
    :param val_pos: byte position of the value to be changed
    :param delta: change in value. (/always added. In case of subtraction, send negative)
    :param dtype: dtype of the value to be read.
    :return: nothing. makes the changes in the binary file and saves it in the same position
    """

    if bin_path == None:
        print("Need binary File path. Given None")
        raise ValueError

    if fresh and delta !=0:
        src = '/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_original/3_PSF_0_wt_origHist.bin'
        dst = bin_path
        copyfile(src,dst)

    with io.open(bin_path, "r+b") as f:
        noOfParticles = len(f.read()) / 68
        print(noOfParticles)
        print("---------------" + str(val_pos))
        f.seek(0)

        changed_particles =0

        for i in range(noOfParticles):
            f.seek(val_pos + (i * 68))
            data = f.read(8)
            x = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            data = f.read(8)
            y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            f.seek(36 + (i * 68))
            energy = f.read(8)
            energy = np.frombuffer(energy, dtype="float64", offset=0, count=1)
            ev_energy = int(energy * 100000)
            if ev_energy != 51100 and (min**2) <= ((x*10)**2 + (y*10)**2) <(max**2):
                new_energy = abs(energy) + delta
                f.seek(36 + (i * 68))
                f.write(np.getbuffer(new_energy))
                changed_particles += 1
                if i < 1000:
                    print("new energy", new_energy)

            if i < 1000:
                print("x...",x)
                print("y....",y)
                print("old energy" , energy)
        f.close()
        print("-----------# of changed particle, .....", changed_particles)

manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_0_2_0p10_0.bin', delta=0.1, min = 0 , max = 2, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_2_4_0p10_0.bin', delta=0.1, min = 2 , max = 4, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_4_6_0p10_0.bin', delta=0.1, min = 4 , max = 6, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_6_8_0p10_0.bin', delta=0.1, min = 6 , max = 8, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_8_10_0p10_0.bin', delta=0.1, min = 8 , max = 10, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_10_12_0p10_0.bin', delta=0.1, min = 10 , max = 12, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_12_14_0p10_0.bin', delta=0.1, min = 12 , max = 14, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_14_16_0p10_0.bin', delta=0.1, min = 14 , max = 16, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_16_18_0p10_0.bin', delta=0.1, min = 16 , max = 18, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_18_20_0p10_0.bin', delta=0.1, min = 18 , max = 20, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_20_22_0p10_0.bin', delta=0.1, min = 20 , max = 22, fresh=True, IAEA=False)
manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_wt_2_4_0p10_0.bin', delta=0.1, min = 0 , max = 2, fresh=True, IAEA=False)



def manipulate_PSF_energy(bin_path=None, val_pos=0, delta=0, dtype='float64', fresh=False, IAEA=False):

    """
    :param bin_path: path for the binary file
    :param val_pos: byte position of the value to be changed
    :param delta: change in value. (/always added. In case of subtraction, send negative)
    :param dtype: dtype of the value to be read.
    :return: nothing. makes the changes in the binary file and saves it in the same position
    """
    if bin_path == None:
        print("Need binary File path. Given None")
        raise ValueError

    if fresh and delta != 0:
        src = '/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_0.bin'
        dst = bin_path
        copyfile(src, dst)
        # bin_path = fresh

    # if IAEA:
    with io.open(bin_path, "rb") as f:
        noOfParticles = len(f.read()) / 68
        print(noOfParticles)
        print("---------------" + str(val_pos))
        f.seek(0)
        # energy = np.zeros((noOfParticles/10,))

        for i in [31580640,31580641,31580642,31580643,31580644,31580645,31580646,31580647,31580648,31580649,31580650]:

            print("STARTING PRINTING FOR PARTICLE: " ,i )
            f.seek(val_pos + (i * 68))
            data = f.read(4)
            y = np.frombuffer(data, dtype='int8', offset=0, count=1)
            print('type....', y)
            data = f.read(8)
            y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            print('wt....', y)
            data = f.read(8)
            y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            print('x....', y)
            data = f.read(8)
            y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            print('y....', y)
            data = f.read(8)
            y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            print('z....', y)
            data = f.read(8)
            y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            print('energy....', y)
            data = f.read(8)
            y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            print('px....', y)
            data = f.read(8)
            y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            print('py....', y)
            data = f.read(8)
            y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            print('pz....', y)

        print("----------- particle done")
            # print(x)
            # print(new_y)

    # else:
    #     with io.open(bin_path, "r+b") as f:
    #         noOfParticles = len(f.read()) / 60
    #         f.seek(0)
    #         print(noOfParticles)
    #         characterRad = {}
    #         for i in range(10):
    #             # val_pos = 28 for energy
    #             f.seek(val_pos + (i * 60))
    #             data = f.read(8)
    #             y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
    #
    #             x = int(y * 100000)
    #             if i % 2 == 0 and x != 51100 and (y + delta) > 0 and delta != 0:
    #                 y = y + delta
    #                 f.seek(val_pos + (i * 60))
    #                 f.write(np.getbuffer(y))
    #                 # print (y)
    #         f.close()
    #         print("-----------")


        # print(ix)
# manipulate_PSF_energy('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_original/3_PSF_0_wt_origHist.bin', delta=0.0, fresh=False, IAEA=False)
# manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_m_0x05_0.bin', delta=0.0, fresh=False)
# manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_p_0x05_0.bin', delta=0.05, fresh=True)
# manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_p_0x15_0.bin', delta=0.15, fresh=True)
# manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_m_0x05_0.bin', delta=-0.05, fresh=True)


# manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_p_0x01_0.bin', delta=0.01, fresh=True)
# manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_p_0x02_0.bin', delta=0.02, fresh=False)
# manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_p_0x03_0.bin', delta=0.03, fresh=False)
# manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_m_0x01_0.bin', delta=-0.01, fresh=False)
# manipulate_PSF('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_m_0x02_0.bin', delta=-0.02, fresh=False)



# manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_0.bin",delta=0.01, fresh=True, IAEA=True, val_pos=1)

# manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_p_0x05_0.bin",delta=0.05, fresh=True, IAEA=True, val_pos=1)
# manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_p_0x15_0.bin",delta=0.15, fresh=True, IAEA=True, val_pos=1)
# manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_m_0x05_0.bin",delta=-0.05, fresh=True, IAEA=True, val_pos=1)
# manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_m_0x10_0.bin",delta=-0.10, fresh=True, IAEA=True, val_pos=1)
# manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_p_0x01_0.bin",delta=0.01, fresh=True, IAEA=True, val_pos=1)
# manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_p_0x02_0.bin",delta=0.02, fresh=True, IAEA=True, val_pos=1)
# manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_p_0x03_0.bin",delta=0.03, fresh=True, IAEA=True, val_pos=1)
# manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_m_0x01_0.bin",delta=-0.01, fresh=True, IAEA=True, val_pos=1)
# manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/1_IAEA_PSF_m_0x02_0.bin",delta=-0.02, fresh=True, IAEA=True, val_pos=1)



def copying_and_creating_new_version_phsp(version):
    src = "/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_"+ str(version)+".bin"
    version = version +1
    dst = "/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_"+ str(version)+".bin"
    copyfile(src, dst)

# copying_and_creating_new_version_phsp(9)
#
# src = '/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_0.bin'
# #
# # dst = '/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_p_0x02_0.bin'
# # # copyfile(src, dst)
# dst = '/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_p_0x10_0.bin'
# copyfile(src, dst)
# dst = '/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_m_0x10_0.bin'
# copyfile(src, dst)
# dst = '/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_m_0x02_0.bin'
# # copyfi1le(src, dst)