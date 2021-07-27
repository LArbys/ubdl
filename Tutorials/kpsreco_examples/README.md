# Plots

Extend the script to do the following.

## Plot nshowers

This is a minor extension of the example script. Make a histogram of the number of showers per vertex.

## Plot the track length for each track

Here, you'll have to extend the script to loop over the individual tracks in each vertex.

The tracks are in the `track_v` vector. The tracks are instances of the `larlite::track` class.
You can reference the class header for the `larlite::track` class [here](https://github.com/NuTufts/larlite/blob/nutufts/larlite/DataFormat/track.h).

## Plot the distance of the track endpoint to the detector boundary

Here, you'll need to get the position of the last track point.
You'll need to feed that end point to the following function: `ublarcvapp::dwall`

The header with the function is found [here](https://github.com/LArbys/ublarcvapp/blob/master/ublarcvapp/ubdllee/dwall.h).