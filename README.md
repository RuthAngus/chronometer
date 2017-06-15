# chronometer
A coherent and comprehensive dating system for the stars in the Milky Way.

Things to do:

Code:
Make it easy to add data.
Make it easy to pull out the parameters of one star.
Tidy.
Use classes.




Method:
Add kinematics.
Add van Saders rotation model.
Add binaries to rotation model.
Add width to rotation model.
Add asteroseismology.
Add ability to use posteriors.
Allow automatic detection of non-convective layer stars (high mass stars, M
and brown dwarfs).
Add the option of tying together the ages and metallicities of cluster and
binary stars.

Scripts:

make_data_file.py:
    Calculates actions for stars.

match_tgas_astero.py:
    Cross-Match TGAS/Kepler/Prot with Kepler astero in van Saders.
    Also, calculate actions for these stars.

make_astero_comparison_plot.py:
    Make a plot comparing the asteroseismic ages with isochronal ages and
    chronometer ages.
