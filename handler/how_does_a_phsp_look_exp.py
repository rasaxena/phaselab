


import numpy as np
import io
import sys
import matplotlib.pyplot as plt

def manipulate_PSF(bin_path=None, val_pos=7,delta=0, dtype='float8', fresh=False):

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

    # if fresh and delta !=0:
    #     src = '/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_0.bin'
    #     dst = bin_path
    #     copyfile(src,dst)
        # bin_path = fresh

    for ix in range(1):


        with io.open(bin_path, "rb") as f:
            # for x in range(120):
            noOfParticles = len(f.read())/33

            print(noOfParticles)
            dt = np.dtype('float32')
            # dt = dt.newbyteorder('>')
            # for val_pos in range(4):
            for val_pos in [0]:
                # val_pos = 28 for energy
                print("---------------" + str(val_pos))
                f.seek(0)
                energyf = np.zeros((noOfParticles/100,))
                xf = np.zeros((noOfParticles/100,))
                yf = np.zeros((noOfParticles/100,))

                for i in range(noOfParticles/100):
                        # f.seek(val_pos+(i*33))
                        f.seek(val_pos + (i * 33))
                        type = f.read(1)

                        type = np.frombuffer(type, dtype='int8', offset=0, count=1)
                        energy = f.read(4)
                        energy = np.frombuffer(energy, dtype=dt, offset=0, count=1)

                        x = f.read(4)
                        x = np.frombuffer(x, dtype=dt, offset=0, count=1)
                        y = f.read(4)
                        y = np.frombuffer(y, dtype=dt, offset=0, count=1)

                        # f.seek(val_pos +5 + (i * 33))
                        # f.seek(val_pos + 9 + (i * 33))

                        if type == 1:
                            energyf[i] = abs(energy)
                            xf[i] = abs(x)
                            yf[i] = abs(y)

                energyf = [tuple([abs(ix)/5.75,0,0]) for ix in energyf]
                f.close()
                plt.scatter(xf, yf, c=energyf, alpha=0.5)
                plt.show()


            print("-----------")
            # bins = np.linspace(0.01, 6, 1000)

            # plt.hist(energy, bins, alpha=0.3, label='original')
            # plt.hist(energyp, bins, alpha=0.3, label='plus 0.1')
            # plt.hist(energym, bins, alpha = 0.3, label = 'minus 0.01')
            # plt.legend(loc='upper right')
            # plt.show()
manipulate_PSF("/home/r/Rangoli.Saxena/Downloads/ELEKTA_PRECISE_6mv_part1(1).IAEAphsp")
# manipulate_PSF("/home/r/Rangoli.Saxena/Downloads/Varian_TrueBeam6MV_01.IAEAphsp")

