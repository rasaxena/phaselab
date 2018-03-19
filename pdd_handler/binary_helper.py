import numpy as np
# /project/med/MAPDOSI/Rangoli.Saxena/
import io
# bin_file =
# bin_file = "/home/r/Rangoli.Saxena/Downloads/fake.bin"
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from fast_histogram import histogram1d, histogram2d


def plot_histogram(bin_path, val_pos=36,dtype='float64'):


    with io.open('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_mom_v1.bin','rb') as f:

        noOfParticles = len(f.read()) / 68
        # # rad = np.zeros((int(noOfParticles/100),))
        x = np.zeros((int(noOfParticles/10000),))
        y = np.zeros((int(noOfParticles/10000),))
        px = np.zeros((int(noOfParticles / 10000),))
        py = np.zeros((int(noOfParticles / 10000),))
        # pz = np.zeros((int(noOfParticles / 100),))
        # r = np.zeros((int(noOfParticles / 100),))
        # energy = np.zeros((int(noOfParticles/100),))
        # print(noOfParticles)

        # print(len(energy))
        f.seek(0)
        for i in range(int(noOfParticles/10000)):
            # val_pos = 28 for energy
            f.seek(0 + (i * 68))
            data = f.read(4)
            type= np.frombuffer(data, dtype='int8', offset=0, count=1)
            data = f.read(8)
            weight= np.frombuffer(data, dtype='float64', offset=0, count=1)
            data = f.read(8)
            xi = np.frombuffer(data, dtype='float64', offset=0, count=1)
            data = f.read(8)
            yi = np.frombuffer(data, dtype='float64', offset=0, count=1)
            data = f.read(8)
            zi = np.frombuffer(data, dtype='float64', offset=0, count=1)
            data = f.read(8)
            energyi = np.frombuffer(data, dtype='float64', offset=0, count=1)
            data = f.read(8)
            pxi = np.frombuffer(data, dtype='float64', offset=0, count=1)
            data = f.read(8)
            pyi = np.frombuffer(data, dtype='float64', offset=0, count=1)
            data = f.read(8)
            pzi = np.frombuffer(data, dtype='float64', offset=0, count=1)
            # zi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
            # f.seek(36 + (i * 68))
            # data = f.read(8)
            # energyi = np.frombuffer(data, dtype="float64", offset=0, count=1)
            # energy[i] = energyi
            # rad[i] = np.sqrt(((xi*10)**2 + (yi*10)**2))
            px[i] = pxi
            py[i] = pyi
            # pz[i] = pzi
            y[i] = yi
            x[i] = xi
            # energy[i] = energyi
            #
            # print('type :', type)
            # print('wt :', weight)
            #
            # print('x :', xi)
            #
            # print('y :', yi)
            #
            # print('z :', zi)
            #
            # print('energy :', energyi)
            #
            # print('px :', pxi)
            # print('py :', pyi)
            #
            # print('pz :', pzi)



    # print("energy",energy[:10])

    # with io.open('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_0_2_3p0.bin', 'rb') as f:
    # #
    #     noOfParticles = len(f.read()) / 68
    #     # rad = np.zeros((int(noOfParticles/100),))
    #     # y = np.zeros((int(noOfParticles / 100),))
    #     energy2 = np.zeros((int(noOfParticles / 1000),))
    #     print(noOfParticles)
    #
    #     # print(len(energy))
    #     f.seek(0)
    #     for i in range(int(noOfParticles/1000)):
    #         # val_pos = 28 for energy
    #         # f.seek(12 + (i * 68))
    #         # data = f.read(8)
    #         # xi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
    #         # data = f.read(8)
    #         # yi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
    #         # data = f.read(8)
    #         # zi = np.frombuffer(data, dtype=dtype, offset=0, count=1)
    #         f.seek(36 + (i * 68))
    #         data = f.read(8)
    #         energyi = np.frombuffer(data, dtype="float64", offset=0, count=1)
    #         energy2[i] = energyi
            # rad[i] = np.sqrt(((xi*10)**2 + (yi*10)**2))

    # bins = np.linspace(0, 10, 100)
    #
    # plt.hist(rad, bins, alpha=1)
    # plt.xlabel('radius in mm')
    # plt.ylabel('number of particles')
    # plt.hist(energy2, bins, alpha=0.3, label='manipulated Mev')

    # if energy.any() == energy10.any():
    #     print ('what the hell')
    # print(x[:10], y[:10], px[:10], py[:10], pz[:10], energy[:10])
    #
    # plt.hist2d(x, y, bins=400, norm=LogNorm())
    # plt.colorbar()
    # plt.show()
    # # plt.hist(energy, bins, alpha = 0.3, label = 'original')
    # plt.legend(loc='upper right')
    # plt.title('Cropped Phase Space histogram')
    # plt.xlabel('x direction')
    # # plt-ylabel('y direction')
    # plt.show()


    import matplotlib.pyplot as plt

    n = int(noOfParticles/10000)
    X =x
    Y=y
    U=px
    V=py
    # X, Y = np.mgrid[0:n, 0:n]
    T = np.arctan2(Y - n / 2., X - n / 2.)
    R = 10 + np.sqrt((Y - n / 2.0) ** 2 + (X - n / 2.0) ** 2)
    # U, V = R * np.cos(T), R * np.sin(T)

    # plt.axes([0.025, 0.025, 0.95, 0.95])
    plt.quiver(X, Y, U, V, R, alpha=.5)
    plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.3)
    axes = plt.gca()
    axes.set_xlim([-10, 10])
    axes.set_ylim([-10, 10])
    # plt.xlim(-5, n)
    plt.xticks(())
    # plt.ylim(-5, n)
    plt.yticks(())
    plt.axes().set_aspect('equal', 'datalim')

    plt.show()

# plot_histogram('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_original/3_PSF_0_wt_origHist.bin')

def manipulate_PSF(bin_path=None, bin_info=None, locus=None, perturbation=None, min_rad=0, max_rad=2, delta=0):

    """
    More info in the man.yaml example
    :param bin_path: a dict, paths for the binary file.
    :param bin_info: byte positions and datatypes
    :param locus: the location of change
    :param perturbation: delta added
    :return: nothing. makes the changes in the binary file and saves it in the same position
    """

    fresh = bin_path['fresh']
    src = bin_path['parent']
    final = bin_path['final']
    name = delta*10
    name = str(delta).replace(".", "p")
    # final = "/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_" + str(min_rad) + "_" + str(max_rad) + "_mm__" + str(name) +".bin"
    # delta = perturbation['delta']
    index = perturbation['index']

    qty = bin_info['qty']
    dtype = bin_info['dtype']
    no_bytes = bin_info['no_bytes']

    shape = locus['shape']

    if shape == 'rad':
        # min_rad = locus['min_rad']
        # max_rad = locus['max_rad']
        x_pos_inx = locus['x_pos_inx']
        y_pos_inx = locus['y_pos_inx']
    elif shape == 'sq':
        #TODO
        pass
    elif shape == 'rand':
        #TODO
        pass

    bytes_per_line = np.sum(no_bytes)

    assert bin_path is not None
    assert shape is not None
    assert final is not None

    val_pos = np.sum(no_bytes[:index])
    print("bin_path : ", final)

    if fresh and delta !=0:
        copyfile(src,final)

    with io.open(final, "r+b") as f:

        no_of_part = len(f.read()) / bytes_per_line
        print(no_of_part)
        print("---------------" + str(val_pos))
        f.seek(0)

        changed_particles =0

        if shape == 'rad':
            for i in range(no_of_part):
                # f.seek(np.sum(no_bytes[:x_pos_inx]) + (i * bytes_per_line))
                # data = f.read(no_bytes[x_pos_inx])
                # x = np.frombuffer(data, dtype=dtype[x_pos_inx], offset=0, count=1)
                #
                # f.seek(np.sum(no_bytes[:y_pos_inx]) + (i * bytes_per_line))
                # data = f.read(no_bytes[y_pos_inx])
                # y = np.frombuffer(data, dtype=dtype[y_pos_inx], offset=0, count=1)

                f.seek(np.sum(no_bytes[:index]) + (i * bytes_per_line))
                energy = f.read(no_bytes[index])
                energy = np.frombuffer(energy, dtype=dtype[index], offset=0, count=1)
                ev_energy = int(energy * 100000)

                # if x <= max_rad*0.1 and y <= max_rad*0.1:
                # if ev_energy != 51100 and (min_rad**2) <= ((x*10)**2 + (y*10)**2) <(max_rad**2):
                if ev_energy != 51100:
                    # print("old energy" , energy)
                    new_energy = abs(energy) + delta
                    f.seek(36 + (i * bytes_per_line))
                    f.write(np.getbuffer(new_energy))
                    # print(x)
                    # print(y)
                    # print("new energy", new_energy)
                    changed_particles += 1

        f.close()
        print("----------- # of changed particles, .....", changed_particles)


def read_PSF(bin_path=None):

    """
    More info in the man.yaml example
    :param bin_path: a dict, paths for the binary file.
    :param bin_info: byte positions and datatypes
    :param locus: the location of change
    :param perturbation: delta added
    :return: nothing. makes the changes in the binary file and saves it in the same position
    """
    if bin_path == None:
        print("Need binary File path. Given None")
        raise ValueError

    min_rad = 2
    max_rad =4
    with io.open(bin_path, "rb") as f:
        noOfParticles = len(f.read()) / 68
        print(noOfParticles)
        f.seek(0)

        changed_particles = 0

        for i in range(1000):
            f.seek(12 + (i*68))
            data = f.read(8)
            x = np.frombuffer(data, dtype="float64", offset=0, count=1)

            f.seek(20 + (i * 68))
            data = f.read(8)

            y = np.frombuffer(data, dtype="float64", offset=0, count=1)

            f.seek(36 + (i * 68))
            energy = f.read(8)
            energy = np.frombuffer(energy, dtype="float64", offset=0, count=1)


            ev_energy = int(energy * 100000)
            if ev_energy != 51100 and (min_rad ** 2) <= ((x * 10) ** 2 + (y * 10) ** 2) < (max_rad ** 2):
                print("x : ", x)
                print("y : ", y)
                print("old energy : ", energy)
        #     if ev_energy != 51100 and (0 ** 2) <= ((x * 10) ** 2 + (y * 10) ** 2) < (2 ** 2):
                changed_particles += 1
        f.close()
        print("-----------# of changed particle, .....", changed_particles)

# read_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/cropped_0p2.bin")
# read_PSF("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_original/cropped_phasespace.bin")

def manipulate_PSF_mom(bin_path=None, bin_info=None, locus=None, perturbation=None, min_rad=0, max_rad=2, delta=0):

    """
    More info in the man.yaml example
    :param bin_path: a dict, paths for the binary file.
    :param bin_info: byte positions and datatypes
    :param locus: the location of change
    :param perturbation: delta added
    :return: nothing. makes the changes in the binary file and saves it in the same position
    """

    fresh = bin_path['fresh']
    src = bin_path['parent']
    final = bin_path['final']
    name = delta*10
    name = str(delta).replace(".", "p")
    # final = "/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_" + str(min_rad) + "_" + str(max_rad) + "_mm__" + str(name) +".bin"
    # delta = perturbation['delta']
    index = perturbation['index']

    qty = bin_info['qty']
    dtype = bin_info['dtype']
    no_bytes = bin_info['no_bytes']

    shape = locus['shape']

    if shape == 'rad':
        # min_rad = locus['min_rad']
        # max_rad = locus['max_rad']
        x_pos_inx = locus['x_pos_inx']
        y_pos_inx = locus['y_pos_inx']
    elif shape == 'sq':
        #TODO
        pass
    elif shape == 'rand':
        #TODO
        pass

    bytes_per_line = np.sum(no_bytes)

    assert bin_path is not None
    assert shape is not None
    assert final is not None

    val_pos = np.sum(no_bytes[:index])
    print("bin_path : ", final)

    if fresh and delta !=0:
        copyfile(src,final)

    with io.open(final, "r+b") as f:

        no_of_part = len(f.read()) / bytes_per_line
        print(no_of_part)
        print("---------------" + str(val_pos))
        f.seek(0)

        changed_particles =0

        if shape == 'rad':
            for i in range(no_of_part):
                f.seek(np.sum(no_bytes[:x_pos_inx]) + (i * bytes_per_line))
                data = f.read(no_bytes[x_pos_inx])
                x = np.frombuffer(data, dtype=dtype[x_pos_inx], offset=0, count=1)

                f.seek(np.sum(no_bytes[:y_pos_inx]) + (i * bytes_per_line))
                data = f.read(no_bytes[y_pos_inx])
                y = np.frombuffer(data, dtype=dtype[y_pos_inx], offset=0, count=1)

                yn = (y/2)*0.9
                #
                # f.seek(np.sum(no_bytes[:index]) + (i * bytes_per_line))
                # energy = f.read(no_bytes[index])
                # energy = np.frombuffer(energy, dtype=dtype[index], offset=0, count=1)
                # ev_energy = int(energy * 100000)

                # if x <= max_rad*0.1 and y <= max_rad*0.1:
                # if ev_energy != 51100 and (min_rad**2) <= ((x*10)**2 + (y*10)**2) <(max_rad**2):
                # if ev_energy != 51100:
                #     print("old energy" , energy)
                # new_energy = abs(energy) + delta
                f.seek(20 + (i * bytes_per_line))
                f.write(np.getbuffer(yn))
                # print(x)
                # print(y)
                # print("new energy", new_energy)
                # changed_particles += 1

        f.close()
        print("----------- # of changed particles, .....")


def crop_PSF(bin_path=None, bin_info=None, locus=None, perturbation=None, min_rad=0, max_rad=2, delta=0):

    """
    More info in the man.yaml example
    :param bin_path: a dict, paths for the binary file.
    :param bin_info: byte positions and datatypes
    :param locus: the location of change
    :param perturbation: delta added
    :return: nothing. makes the changes in the binary file and saves it in the same position
    """



    with io.open("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_original/3_PSF_0_wt_origHist.bin", "rb") as f:


        no_of_part = len(f.read()) / 68
        print(no_of_part)
        # print("---------------" + str(val_pos))
        f.seek(0)

        changed_particles =0

        with io.open("/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_cropped_ellipse_1y1.bin", "wb") as fx:
            # if shape == 'rad':
            for i in range(no_of_part):
                # f.seek(np.sum(no_bytes[:x_pos_inx]) + (i * bytes_per_line))
                # data = f.read(no_bytes[x_pos_inx])
                # x = np.frombuffer(data, dtype=dtype[x_pos_inx], offset=0, count=1)

                f.seek(20 + (i * 68))
                data = f.read(8)
                y = np.frombuffer(data, dtype="float64", offset=0, count=1)
                #
                # f.seek(np.sum(no_bytes[:index]) + (i * bytes_per_line))
                # energy = f.read(no_bytes[index])
                # energy = np.frombuffer(energy, dtype=dtype[index], offset=0, count=1)
                # ev_energy = int(energy * 100000)

                if -1 <= y*10 <= 1:
                # if ev_energy != 51100 and (min_rad**2) <= ((x*10)**2 + (y*10)**2) <(max_rad**2):
                # if ev_energy != 51100:
                    # print("old energy" , energy)
                    # new_energy = abs(energy) + delta
                    f.seek(i * 68)
                    data = f.read(68)
                    fx.write(np.getbuffer(data))
                    # print(x)
                    # print(y)
                    # print("new energy", new_energy)
                    changed_particles += 1

        f.close()
        fx.close()
        print("----------- # of changed particles, .....", changed_particles)


crop_PSF('mm')