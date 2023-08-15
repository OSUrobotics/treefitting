import os
#extracts depth and rgb data from azure kinect video
#requires ffmpeg library: sudo apt install ffmpeg
"""
INSTRUCTIONS FOR RUNNING:
1. Install ffmpeg
2. Download folder containing video output file
3. Copy and paste this file into the downloaded folder
4. Navigate to downloaded folder in terminal
5. Run script and specify filename(excluding extension) you would like to extract data from.
"""
print("please type the file name(excluding extension) you would like to extract from: ")
inp = input()
filename = f'{inp}.mkv'
current_directory = os.getcwd()
rgb_folder = f'{inp}_rgb_data'
depth_folder = f'{inp}_depth_data'
path1 = os.path.join(current_directory, rgb_folder)
path2 = os.path.join(current_directory, depth_folder)
os.mkdir(path1)
os.mkdir(path2)

cmd1 = f'ffmpeg -i {filename} -vf "select=not(mod(n\,10))" -map 0:0 -vsync 0 {path1}/rgb%04d.png'
cmd2 = f'ffmpeg -i {filename} -vf "select=not(mod(n\,10))" -map 0:1 -vsync 0 {path2}/depth%04d.raw'
os.system(cmd1)
os.system(cmd2)
