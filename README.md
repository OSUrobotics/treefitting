# TreeFitting
Fit cylinders and branch points to tree point cloud data

# Curve fitting
## Getting the data
Blueberries: Getting a data set from Box
This assumes that the .mkv file has been turned into folders of images (rgb and depth)
* utils/video_annotation_data, main file
  * set b_copy_blueberry to True and all others to false
  * Note: box_path is the path on your computer that is the path where Box lives on your local drive. It currently checks two places; feel free to add another
  * Change bush_name in if b_copy_blueberry to be the bush you want
    * Either before or after pulling files, you can look through them to determine the subset you want to use
    * Change the add_directory line to have start and stop values with the bush starting in the left third and moving to the right third, step size 32 ish is usually good
  * This builds two .json files, all_fnames and video_annot. The first has all the filenames, the second has every nth
  * 

## Running the gui
video_annotation_gui/data_annotation_main_window.py is the main file. 

## Re-doing the fit
* utils/video_annotation_data, main file
* Set b_redo_fit to be True and make sure the bush name and .json file are correct
* will output to foo_refit.json, and put debug images in debug (fitted curves)
* There are two primary values to change: average fit and inlier. The first is the average fit for all points, the second is how accurate the individual points need to be. In pixels


## 3D fit
* fit_routines/fit_bspline_cyl_3d.py
