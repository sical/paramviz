""" Import Modules """

from netCDF4 import Dataset

import numpy as np
import pylab as pl

from bokeh.plotting import figure, output_file, show, HBox, VBox
from bokeh.resources import CDN, INLINE
from bokeh.embed import components

""" Read NetCDF   """

fh = Dataset('/Users/jorge/.spyder2/zoop/NATMIG_1m_20060101_20061231_grid_T_DFS5.compress.nc', 'r')
lons = fh.variables['nav_lon'][:]
lats = fh.variables['nav_lat'][:]
zdepmig = fh.variables['zdepmig'][:,:,:]
pp= fh.variables['PPPHY'][:,:,:,:]
fh.close()

""" Python Plot """

#for i in range(len(zdepmig)):
pl.clf()
#pl.contourf(zdepmig[1])
x=pp[1,2,:,200]
y=pp[2,2,:,200]
pl.scatter(x,y)
pl.show()

""" Bokeh Plot   """

output_file("bo1_lines.html",title='ZOOP MIG')

p1 = figure(title="Scatter MIG", x_axis_label='x', y_axis_label='y',plot_width=300, plot_height=300)

xa=x/1.5
ya=y/1.5
xaxis=range(1,64)
p1.scatter(y, x, legend="PP", line_width=2, color="red")
p1.scatter(ya, xa, legend="NPP", line_width=2)

show(p1)
script, div = components(p1,INLINE)
print script