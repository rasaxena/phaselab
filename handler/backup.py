import numpy as np
# /project/med/MAPDOSI/Rangoli.Saxena/
import io
# bin_file =
# bin_file = "/home/r/Rangoli.Saxena/Downloads/fake.bin"
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from fast_histogram import histogram1d, histogram2d

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

    with io.open(bin_path, "rb") as f:
        noOfParticles = len(f.read()) / 68
        print(noOfParticles)
        print("---------------" + str(val_pos))
        f.seek(0)

        changed_particles =0

        for i in range(10):
            # f.seek(val_pos + (i * 68))
            # data = f.read(8)
            # x = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            # data = f.read(8)
            # y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            f.seek(36 + (i * 68))
            energy = f.read(8)
            energy = np.frombuffer(energy, dtype="float64", offset=0, count=1)
            ev_energy = int(energy * 100000)
            # if ev_energy != 51100 and (min**2) <= ((x*10)**2 + (y*10)**2) <(max**2):
            #     new_energy = abs(energy) + delta
            #     f.seek(36 + (i * 68))
            #     f.write(np.getbuffer(new_energy))
            #     changed_particles += 1
            #     if i < 1000:
            #         print("new energy", new_energy)

            if i < 1000:
                # print("x...",x)
                # print("y....",y)
                print("old energy" , energy)
        f.close()
        print("-----------# of changed particle, .....", changed_particles)

manipulate_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_0_50_0p20_01.bin", fresh=False)