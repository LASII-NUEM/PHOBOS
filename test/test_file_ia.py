from framework import file_ia

IA_data = '../data/freezerVSchiller/IA_F_03_02/Spectrum.xls'

IA_obj = file_ia.read(IA_data)
IA_C0_obj = IA_obj["c0"]
IA_C1_obj= IA_obj["c1"]
IA_Cice_obj= IA_obj["cice"]