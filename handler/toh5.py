import h5py
import numpy as np

f = lambda x: x.split('   ')

x = [1,2,3,4,5,6,7,8,9,10]
y = ['a','b','c','d','e','f','g','h','i','j']
for ix in range(10):
    with open("phsp/phspChunka{0}".format(str(y[ix])) , 'r') as phsp:
        content = phsp.readlines()
    content = np.asarray([np.asarray(x.strip().split('\t')) for x in content])
    print(content.shape)
    #diff processes in diff steps is a tiny bit faster
    np_total = content[:,1]
    np_total = np.core.defchararray.rsplit(np_total, sep='   ')
    np_total = np.asarray([np.asarray(inx, dtype=np.float32) for inx in np_total])

    np_energy = np_total[:,0]
    np_px = np_total[:,1]
    np_py = np_total[:,2]
    np_pz = np_total[:,3]

    h = h5py.File("phsp/chunk_a{0}.h5".format(y[ix]) , 'w')
    e_dataset = h.create_dataset('energy' , data=np_energy)
    px_dataset = h.create_dataset('px' , data=np_energy)
    py_dataset = h.create_dataset('py' , data=np_energy)
    pz_dataset = h.create_dataset('pz' , data=np_energy)






