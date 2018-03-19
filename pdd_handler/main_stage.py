import sys
import yaml
from binary_helper import manipulate_PSF , read_PSF,manipulate_PSF_mom
cmdargs = sys.argv

mp = cmdargs[1]

with open(mp,'r') as f:
    data = yaml.load(f)

print(data)
dict_bin_path = data['bin_path']

dict_bin_info = data['bin_info']
dict_locus = data['locus']
dict_perturbation = data['perturbation']

# min_rad= [0,20,40]
# max_rad =[20,40,60]
delta = 0.3
min=2
max=4

manipulate_PSF(dict_bin_path, dict_bin_info, dict_locus, dict_perturbation, min, max, delta)