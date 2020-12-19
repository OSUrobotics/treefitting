# TreeFitting
Fit cylinders and branch points to tree point cloud data

<h2>Dependencies</h2>

- Pymesh: *Not required anymore!* However it reads PLY files faster than the roll-your own method I found, so you can still install it for speed.
https://github.com/PyMesh/PyMesh - Clone the repository and install it with pip. If you're on a setup with both Python 2.7 and Python 3.5,
use `pip3 install . --user` to install it onto Python 3.
- Networkx: `pip3 install networkx`
- Numpy, Pandas: `sudo apt-get install python3-pandas`
- imageio: `pip3 install imageio`
- PyQt5
- PyOpenGL

<h2>How to use</h2>
To get things set up:
- Download the point clouds and skeletonization results, and put them somewhere on your computer (separate folders).
- Download the configs zip and move the configs folder to the root of the code repository. (E.g. /home/alex/python/TreeFitting/configs/[long string]/config.pickle)
- Run MainWindow.py and download any dependencies above as necessary.
- When the GUI opens, click on the Annotation Panel button on the right.
- For the Point Cloud Directory, enter the folder where the point cloud folders are extracted (so the folder which CONTAINS bag_1, bag_2, etc.)
- For the Results Directory, enter the folder where the skeletonization results are 
- Hit Refresh. If all goes well, you should see the point cloud and skeleton pop up.
- If the point cloud still has the tree in the background showing up, the config isn't loading in properly. When the point cloud loads in, the console should output a message like "Base: [string]". Tell me what that string is.