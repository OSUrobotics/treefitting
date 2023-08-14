# internship_summer_2023
Work and documentation from my summer 2023 AgAID internship.

# What it is
The following code takes in masks and depth images of trees and determines which side branches are connected to which main or "leader" branches.

# How it works
1. Builds a Bezier curve by determining the medial axis of the white space in a mask
   - This returns the radii, "t" value and the curve points at which the curve is evaluated, as well as a curve
2. Finds average depth in a cross-space of the tree branch
   - Uses numpy and opencv to do this by first drawing singular blue squares that align with the limits of the mask and the curve points, then by reading the depth image at those points
3. Determines the closest point and distance from the end points of the side branch to the curve of the leader and picks the closest two points
4. Determines the difference in depth between the two points
5. Determines the angle between the two tangent lines of each points
6. Returns a score based in those aspects that refers to how likely it is that the branches are connected (lower is better)
7. Plots curves and points it is evaluating
    
# Assumptions
- Leader and side branches are labeled as such
- Points are ordered
  
# Notes for Running
- After cloning the repository, just run connection_test.py
- Adjust any of the parameters to include your own depth and mask images
- Requires OpenCV, Matplotlib, Skimage, Scipy, and Networkx libraries

# General Notes
- The curve_fitting and camera files are contribution from [Alex You](https://github.com/osu-youa)
- The pinhole camera commented out is used to determine the size of the tree in real life, is useful for 3D generation.
- Azure camera depth data needs to be tested to determine its conversion to meters
