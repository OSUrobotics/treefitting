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
