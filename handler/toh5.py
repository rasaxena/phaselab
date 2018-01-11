import h5py
import numpy as np

f = lambda x: x.split('   ')

x = [1,2,3,4,5,6,7,8,9,10]
y = ['a','b','c','d','e','f','g','h','i','j']
for ix in range(1):

    with open("/project/med-clhome/r/Rangoli.Saxena/PhSp/txt/phsp_full_a{0}".format(str(y[ix])) , 'r') as phsp:
        content = phsp.readlines()
    content = np.asarray([np.asarray(x.strip().replace('\t', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').split(' ')) for x in content])
    # content = np.asarray([y.split(" ") for y in x for x in content])
    # print(content[:4])
    # print(content.shape)
    #diff processes in diff steps is a tiny bit faster
    # np_total = content[:,1]
    # np_total = np.core.defchararray.rsplit(np_total, sep='   ')
    # np_total = np.asarray([np.asarray(inx, dtype=np.float32) for inx in np_total])

    np_energy = content[:,10]
    np_px = content[:,11]
    np_py = content[:,12]
    np_pz = content[:,13]

    h = h5py.File("/project/med-clhome/r/Rangoli.Saxena/PhSp/h5/phsp_trash_chunk_a{0}.h5".format(y[ix]) , 'w')
    e_dataset = h.create_dataset('energy' , data=np_energy)
    px_dataset = h.create_dataset('px' , data=np_px)
    py_dataset = h.create_dataset('py' , data=np_py)
    pz_dataset = h.create_dataset('pz' , data=np_pz)






