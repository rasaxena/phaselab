import sys
import yaml
cmdargs = sys.argv

mp = cmdargs[1]

document = file(mp, 'r')
parsed = yaml.load(document)
print(yaml.dump(parsed))
# from yaml import load, dump



# with io.open(mp.bin_path, "r+b") as f:
#     noOfParticles = len(f.read()) / 68
#     print(noOfParticles)
#     # dt = np.dtype('float32')
#     print("---------------" + str(val_pos))
#     f.seek(0)
#     # energy = np.zeros((noOfParticles/10,))
#
#     for i in range(noOfParticles):
#         f.seek(val_pos + (i * 68))
#         data = f.read(8)
#         x = np.frombuffer(data, dtype=dtype, offset=0, count=1)
#         data = f.read(8)
#         y = np.frombuffer(data, dtype=dtype, offset=0, count=1)
#         # print (y)
#         f.seek(36 + (i * 68))
#         energy = f.read(8)
#         energy = np.frombuffer(energy, dtype="float64", offset=0, count=1)
#         ev_energy = int(energy * 100000)
#         if ev_energy != 51100 and (x ** 2 + y ** 2) < (2.5 ** 2):
#             # energy[i] = abs(y)
#             new_energy = abs(energy) + delta
#             f.seek(36 + (i * 68))
#             if i < 1000:
#                 print("new energy", new_energy)
#
#         if i < 1000:
#             print("x...", x)
#             print("y....", y)
#             print("old energy", energy)
#     f.close()


