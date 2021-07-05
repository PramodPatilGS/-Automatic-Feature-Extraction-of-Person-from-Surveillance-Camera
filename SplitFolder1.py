import splitfolders  # or import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(r"D:\pycharm\beard_detection\beard_data_modified", output=r"D:\pycharm\beard_detection\slipt data", seed=1337, ratio=(0.8,0.2))
