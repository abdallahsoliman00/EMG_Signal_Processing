# Configure matplotlib plots
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif' # or 'sans-serif' or 'monospace'
mpl.rcParams['font.serif'] = 'cmr10'
mpl.rcParams['font.sans-serif'] = 'cmss10'
mpl.rcParams['font.monospace'] = 'cmtt10'
mpl.rcParams["axes.formatter.use_mathtext"] = True # to fix the minus signs

# Use \mathrm instead ot \text when working with latex equations and set 
# mpl.rcParams['text.usetex'] = True
