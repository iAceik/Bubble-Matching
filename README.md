# Automation-of-Helium-nano-bubble-mathcing
2 algorithms for identifying bubbles across differently focused TEM images.

Direct:
Change all necessary directories.
y-axis is inverted on all graphs because in paint the coordinate 0,0 is in the top left.
Parameters-
Overlap threshold changes the minimum distance between blobs to be bubbles.
Initial corrdinates of vectors define the mapping and transformation.
Highlight point coordinates can be changed to inspect different bubbles.


Relative angle:
Change all necessary directories.
y-axis is inverted on all graphs because in paint the coordinate 0,0 is in the top left.
Run optimisation for inital guesses of parameters in different samples. 
Highlight point coordinates can be changed to inspect different bubbles.
Parameters-
Initial bubble index in over focused and underfocused define the position of the reference points.
Nearest neighbour radius defines the area in which neighbours are selected to be passed through the loop.
Distance threshold selects points within nearest neighbours to be tested only if they're closer to each other than the threshold.
Angle threshold defines the minimum anggular difference two blobs can have.
