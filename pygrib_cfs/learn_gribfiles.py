import pygrib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
import numpy as np


grib = "/home/hpcs_rnd/George_DIAT/Daily_Model_IMD_Data/2020/gfsrain_2020060100.grib" 
grbs = pygrib.open(grib)

grb = grbs.select()[0]
data = grb.values


# need to shift data grid longitudes from (0..360) to (-180..180)
lons = np.linspace(float(grb['longitudeOfFirstGridPointInDegrees']), float(grb['longitudeOfLastGridPointInDegrees']), int(grb['Ni']) )
lats = np.linspace(float(grb['latitudeOfFirstGridPointInDegrees']), float(grb['latitudeOfLastGridPointInDegrees']), int(grb['Nj']) )
data, lons = shiftgrid(180., data, lons, start=False)
grid_lon, grid_lat = np.meshgrid(lons, lats) #regularly spaced 2D grid
 
m = Basemap(projection='cyl', llcrnrlon=-180, 
    urcrnrlon=180.,llcrnrlat=lats.min(),urcrnrlat=lats.max(), 
    resolution='c')
 
x, y = m(grid_lon, grid_lat)
 
#cs = m.pcolormesh(lons,lats,data.T,shading='flat',cmap=plt.cm.gist_stern_r)
a=plt.contourf(lons, lats, data,
                cmap="rainbow", extend = "max")
 
#m.drawcoastlines()
#m.drawmapboundary()
#m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
#m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
 
plt.colorbar()#cs,orientation='vertical', shrink=0.5)
plt.title('GFS Rain') # Set the name of the variable to plot
plt.show()