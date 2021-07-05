import glob
from PIL import Image

#Retriving all image names and it's path with .jpg extension from given directory path in imageNames list
imageNames = glob.glob(r"D:\pycharm\beard_detection\Beard data\No\*.jpg")

#Defining width and height of image
new_width  = 224
new_height = 224

#Count variable to show the progress of image resized
count=0

#Creating for loop to take one image from imageNames list and resize
for i in imageNames:
	#opening image for editing
	img = Image.open(i)
	#using resize() to resize image
	img = img.resize((new_width, new_height), Image.ANTIALIAS)
	#save() to save image at given path and count is the name of image eg. first image name will be 0.jpg
	img.save(r"D:\pycharm\beard_detection\beard_data_modified\No\\"+str(count)+".jpg")
	#incrementing count value
	count+=1
	#showing image resize progress
	print("Images Resized " +str(count)+"/"+str(len(imageNames)),end='\r')