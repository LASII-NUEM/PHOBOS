# About PHOBOS
PHOBOS is a Python toolbox developed for applying standard digital signal processing techniques to all electrical impedance spectroscopy (EIS) signals acquired within the **PHOBOS (Phase Hydrate OBservatiOn System)** context. Given the multiple types of acquisition hardware and respective data formats, PHOBOS_toolbox provides methods to automatically extract the required attributes and organize them into custom data structures. As of the latest version of the framework, the supported files are as follows: multi-electrode sensor spectroscopy/freerun CSV files (PHOBOS C# acquisition firmware) and ADMX2001 spectroscopy CSV files.

# Running the framework
To set up the toolbox, first clone this repository on your machine:

```
cd <desired_path>
git clone https://github.com/LASII-NUEM/PHOBOS.git
cd PHOBOS
```

Then, configure a Python environment. The framework has been successfully tested in Python 3.11 and 3.12. Also, in both Linux and Windows operating systems. Lastly, install the required packages listed in the requirements text file at the root of the repository with:

```
pip install -r requirements.txt
```

Once all packages are installed, the toolbox is ready for use. In the following sections, you may find some examples of how to use the PHOBOS tools. 

## Multi-electrode sensor flange files 

The infrastructure requirement is to add the CSV files from the acquisition firmware to a directory at the root of the repository. The name of the directory itself can be arbitrary. However, for standardization, it is recommended that both temperature and electrode files obey the following rules:
```
temperature -> ./<data_directory>/<acquisition_directory>/c_temp.lvm
electrode -> ./<data_directory>/<acquisition_directory>/c_test.csv
```

To read the contents of the file into a PHOBOSData structure:
```
from framework import data_types
phobos_obj = data_types.PHOBOSData(<electrode_filename>, <temperature_filename>, n_samples=<number_of_samples_per_acquired_pair>, sweeptype="flange", acquisition_mode="freq", aggregate=<aggregation_function_for_the_samples>)
```

Electrode or temperature files can also be processed individually:
```
from framework import file_lcr
electrode_obj = file_lcr.read(<electrode_filename>, n_samples=<number_of_samples_per_acquired_pair>, sweeptype="flange", acquisition_mode="freq", aggregate=<aggregation_function_for_the_samples>)
```
and
```
from framework import file_lvm
temp_obj = file_lvm.read(<temperature_filename>)
```

## Single-electrode sensor commercial cell files 

The infrastructure requirement is to add the CSV files from the acquisition firmware to a directory at the root of the repository. The name of the directory itself can be arbitrary. However, for standardization, it is recommended that both temperature and electrode files obey the following rules:
```
electrode -> ./<data_directory>/<acquisition_directory>/c_test.csv
```

To read the contents of the file into a PHOBOSData structure:
```
from framework import data_types
phobos_obj = data_types.PHOBOSData(<electrode_filename>, n_samples=<number_of_samples_per_acquired_pair>, sweeptype="cell", acquisition_mode="freq", aggregate=<aggregation_function_for_the_samples>)
```

Electrode files can also be processed individually:
```
from framework import file_lcr
electrode_obj = file_lcr.read(<electrode_filename>, n_samples=<number_of_samples_per_acquired_pair>, sweeptype="cell", acquisition_mode="freq", aggregate=<aggregation_function_for_the_samples>)
```

## Multi-electrode cell sensor spectroscopy files 

The infrastructure requirement is to add the CSV files from the acquisition firmware to a directory at the root of the repository. The name of the directory itself can be arbitrary. However, for standardization, it is recommended that the files be named based on the medium analyzed:
```
air -> ./<data_directory>/<acquisition_directory>/c0.csv
water -> ./<data_directory>/<acquisition_directory>/c1.csv
ice -> ./<data_directory>/<acquisition_directory>/cice.csv
```

To read the contents of the file into a SpectroscopyData structure:
```
from framework import file_lcr
spec_air_obj = file_lcr.read(<air_spectroscopy_filename>, n_samples=<number_of_samples_per_acquired_pair>, sweeptype="flange", acquisition_mode="spectrum", aggregate=<aggregation_function_for_the_samples>)
spec_h2o_obj = file_lcr.read(<water_spectroscopy_filename>, n_samples=<number_of_samples_per_acquired_pair>, sweeptype="flange", acquisition_mode="spectrum", aggregate=<aggregation_function_for_the_samples>)
spec_ice_obj = file_lcr.read(<ice_spectroscopy_filename>, n_samples=<number_of_samples_per_acquired_pair>, sweeptype="flange", acquisition_mode="spectrum", aggregate=<aggregation_function_for_the_samples>)
```


## Single-electrode cell sensor spectroscopy files 

The infrastructure requirement is to add the CSV files from the acquisition firmware to a directory at the root of the repository. The name of the directory itself can be arbitrary. However, for standardization, it is recommended that the files be named based on the medium analyzed:
```
air -> ./<data_directory>/<acquisition_directory>/c0.csv
water -> ./<data_directory>/<acquisition_directory>/c1.csv
ice -> ./<data_directory>/<acquisition_directory>/cice.csv
```

To read the contents of the file into a SpectroscopyData structure:
```
from framework import file_lcr
spec_air_obj = file_lcr.read(<air_spectroscopy_filename>, n_samples=<number_of_samples_per_acquired_pair>, sweeptype="cell", acquisition_mode="spectrum", aggregate=<aggregation_function_for_the_samples>)
spec_h2o_obj = file_lcr.read(<water_spectroscopy_filename>, n_samples=<number_of_samples_per_acquired_pair>, sweeptype="cell", acquisition_mode="spectrum", aggregate=<aggregation_function_for_the_samples>)
spec_ice_obj = file_lcr.read(<ice_spectroscopy_filename>, n_samples=<number_of_samples_per_acquired_pair>, sweeptype="cell", acquisition_mode="spectrum", aggregate=<aggregation_function_for_the_samples>)
```

From the processed files, it is possible to compute the dielectric parameters of the media and compare them to models from the literature. For example, to compute the parameters for ice:
```
from framework import characterization_utils
exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_ice_obj, spec_air_obj, spec_ice_obj.freqs) #experimental data
ideal_eps_real, ideal_eps_imag = characterization_utils.dielectric_params_Artemov2013(spec_ice_obj.freqs, medium="ice") #ideal curves
```

## Impedance Analyzer 4294A spectroscopy files
The infrastructure requirement is to add the Xls file from the acquisition firmware to a directory at the root of the repository. The name of the directory itself can be arbitrary. However, for standardization, it is recommended.
```
TestData -> ./<data_directory>/<acquisition_directory>/4294A_DataTransfer_0310.xls
```
To read the contents of the file into a SpectroscopyData structure:
```
from framework import file_ia
spec_ia_obj = file_ia.read(<air_spectroscopy_filename>)
```
As the 4294A IA file is generated via VBA, each measurement is stored in its own worksheet inside the .xls. After loading the file, spec_ia_obj behaves like a dictionary: the keys are the sheet names, and each value is the corresponding SpectroscopyData object. To access the sheet list and a specific test/sheet:
```
# list available sheets (tests)
sheets = list(spec_ia_obj.keys())
print(sheets)

# access a specific test by sheet name
# Air (C0)
spec_ia_air= spec_ia_obj["C0"]
```
From the processed files, it is possible to compute the dielectric parameters of the media and compare them to models from the literature. For example, to compute the parameters for ice:
```
from framework import characterization_utils
exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_ia_obj["Sheet"], spec_ia_air, spec_ia_obj["Sheet"].freqs) #experimental data
ideal_eps_real, ideal_eps_imag = characterization_utils.dielectric_params_Artemov2013(spec_ia_obj["Sheet"].freqs, medium="ice") #ideal curves
```

## ADMX2001 spectroscopy files
The infrastructure requirement is to add the CSV files from the acquisition firmware to a directory at the root of the repository. The name of the directory itself can be arbitrary. However, for standardization, it is recommended that the files be named based on the medium analyzed:
```
air -> ./<data_directory>/<acquisition_directory>/c0.csv
water -> ./<data_directory>/<acquisition_directory>/c1.csv
ice -> ./<data_directory>/<acquisition_directory>/cice.csv
```

To read the contents of the file into a SpectroscopyData structure:
```
from framework import file_admx
spec_air_obj = file_admx.read(<air_spectroscopy_filename>, sweeptype="cell", acquisition_mode="spectrum")
spec_h2o_obj = file_admx.read(<water_spectroscopy_filename>, sweeptype="cell", acquisition_mode="spectrum")
spec_ice_obj = file_admx.read(<ice_spectroscopy_filename>, sweeptype="cell", acquisition_mode="spectrum")
```

From the processed files, it is possible to compute the dielectric parameters of the media and compare them to models from the literature. For example, to compute the parameters for ice:
```
from framework import characterization_utils
exp_eps_real, exp_eps_imag = characterization_utils.dielectric_params_corrected(spec_ice_obj, spec_air_obj, spec_ice_obj.freqs) #experimental data
ideal_eps_real, ideal_eps_imag = characterization_utils.dielectric_params_Artemov2013(spec_ice_obj.freqs, medium="ice") #ideal curves
```

# User guides

In-depth guides of the available methods and data visualization options are available at the [results](https://github.com/LASII-NUEM/PHOBOS/tree/main/results) directory, and are highly recommended for first-time users!

- **Capacitance/Resistance and Temperature data (Flange)**: https://github.com/LASII-NUEM/PHOBOS/blob/main/results/PHOBOS_acquisition_flange.py
- **Capacitance/Resistance and Temperature data (Commercial Cell)**: https://github.com/LASII-NUEM/PHOBOS/blob/main/results/PHOBOS_acquisition_cell.py
- **Dielectric parameters of ice (PHOBOS acquisition flange)**: https://github.com/LASII-NUEM/PHOBOS/blob/main/results/characterization_ice_flange_phobos.py
- **Dielectric parameters of ice (PHOBOS acquisition cell)**: https://github.com/LASII-NUEM/PHOBOS/blob/main/results/characterization_ice_cell_phobos.py
- **Dielectric parameters of water (ADMX2001 acquisition)**: https://github.com/LASII-NUEM/PHOBOS/blob/main/results/characterization_water_cell_admx.py
- **Dielectric parameters of water (PHOBOS acquisition flange)**: https://github.com/LASII-NUEM/PHOBOS/blob/main/results/characterization_water_flange_phobos.py
- **Dielectric parameters of water (PHOBOS acquisition cell)**: https://github.com/LASII-NUEM/PHOBOS/blob/main/results/characterization_water_cell_phobos.py

