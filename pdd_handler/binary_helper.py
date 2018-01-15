import numpy as np
# /project/med/MAPDOSI/Rangoli.Saxena/
import io
# bin_file =
# bin_file = "/home/r/Rangoli.Saxena/Downloads/fake.bin"

def change_bin_file(bin_path=None, val_pos=28,delta=0, dtype='float64'):

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

    for ix in range(1):
        with io.open(bin_path, "r+b") as f:
            noOfParticles = len(f.read())/60
            f.seek(0)
            for i in range(noOfParticles):
                    # val_pos = 28 for energy
                    f.seek(val_pos+(i*60))
                    data = f.read(8)
                    y = np.frombuffer(data, dtype=dtype, offset=0, count=1)

                    if delta !=0:
                        y = y + delta
                        f.seek(val_pos + (i * 60))
                        f.write(np.getbuffer(y))

                    # print (y)
            f.close()
        # print(ix)
change_bin_file('/project/med/MAPDOSI/Rangoli.Saxena/PSFMan/PhSp_bin/3_PSF_0.bin', delta=0.00)