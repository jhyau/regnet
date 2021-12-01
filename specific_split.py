import os, sys

all_vids = os.listdir('./data/features/ASMR/orig_asmr_by_material_clips/OF_10s_21.5fps')

with open('./asmr_by_material_specific_split_val.txt', 'a+') as f:
    for name in all_vids:
        if name.find('wood') != -1 and name.find('surface') == -1 and name.find('box') == -1:
            f.write(name + '\n')
