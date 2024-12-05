from opticalglass.glassfactory import create_glass, get_glass_catalog
import pandas as pd

pd = get_glass_catalog('SUMITA')
pd_df = pd.df

#pd_df.to_csv('sumita_glass_catalog.csv')

disp_coef = [2.5843629,-0.010743741,0.014943604,0.00020007994,4.5624595e-06,-1.2296939e-07]

import numpy as np
wavelength = np.linspace(0.35,2.5,100)

n = np.sqrt(1 + disp_coef[0]*wavelength**2/(wavelength**2 - disp_coef[3]) + disp_coef[1]*wavelength**2/(wavelength**2 - disp_coef[4]) + disp_coef[2]*wavelength**2/(wavelength**2 - disp_coef[5]))
print(n)