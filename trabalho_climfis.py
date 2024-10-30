
#pip install netCDF4
#pip install cartopy

import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.pyplot as plt
#import pandas as pd
from cartopy.util import add_cyclic_point
import xarray as xr
import netCDF4 as nc

dados = xr.open_dataset("air_global.nc") 
data = nc.Dataset("air_global.nc")
temp = data.variables["air"][:]
lat = data.variables["lat"][:]
lon = data.variables["lon"][:]
temperatura = dados["air"][:]-273.15
temperatura, lon = add_cyclic_point(temp, coord=dados['lon'])
data.close()

figure = plt.figure(figsize=(10,15))
projecao = ccrs.PlateCarree()

#média
media = np.mean(temperatura[:,2,], axis=0)
#desvio da média
desvio = temperatura[0,2,] - media
#média zonal
mz =  temperatura[0,2,].mean(1)
var = np.ma.resize(mz[:],(145,73)).T
desmz = temperatura[0,2,] - var


#média vertical
niveis_p = np.array((1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100))
cmd = niveis_p - np.array([925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70]) 
def f_vertical(tm):
  ve = np.zeros([73,145])
  for i in range(145):
    for j in range(73):
      vet=np.array([temp[tm,nvl,j,i] for nvl in range(12)])
      ve[j,i]=np.inner(vet, cmd)/(1000-70)

  return ve


#perfil de temperatura x latitude
plt.plot(lat,temperatura[0,2,].mean(1)-273.15, color='orange',)
plt.ylabel("Temperatura (ºC)")
plt.xlabel("Latitude")
plt.title("Perfil vertical de temperatura às 00UTC do dia 01/05/2020 em 30°S/54ºW")
plt.legend(loc='lower left')
plt.grid()
plt.show()


#perfil média vertical
media_v = np.inner(temperatura[0, :, -30, -54]-273.15, cmd) / cmd.sum()  # Assuming -30 and -54 are valid indices
escala_vertical = np.linspace(100, 1000, 12)

plt.axvline(media_v, color='blue', label="<T>")
plt.plot(temperatura[0, :, -30, -54]-273.15, escala_vertical, color='orange', label="T''")
plt.ylim(100, 1000)
plt.ylabel("Pressão (hPa)")
plt.yticks(np.arange(100, 1001, 100), ["{}".format(i) for i in range(1000, 0, -100)])
plt.xlabel("Temperatura (°C)")
plt.title("Perfil vertical de temperatura às 00UTC do dia 01/05/2020 em 30°S/54ºW")
plt.legend(loc='lower left')
plt.grid()
plt.show()


#plots de média global e zonal, desvios, só trocar a variável e o título
mapa = plt.axes(projection=projecao)
exemplo = mapa.contourf(lon, lat, desmz , 35, cmap='RdBu_r')
mapa.add_feature(cfeat.COASTLINE, lw=1)
mapa.add_feature(cfeat.BORDERS, lw=.5)
eixo_cbar=figure.add_axes([0.2, 0.33, 0.6, 0.02])
cbar=figure.colorbar(exemplo,eixo_cbar, orientation='horizontal')

plt.title("Desvio da média zonal de temperatura (°C) em 01/05/2020 às 00UTC".format(), size=10)
plt.show()