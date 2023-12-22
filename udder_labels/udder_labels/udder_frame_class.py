import os
import cv2

dirpath = os.getcwd()
image_path = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_video\color_images')
label_path = os.path.join(dirpath, r'labels\class') 
label_list = [file.replace(".txt", "") for file in  os.listdir(label_path)]

# list files in cow folders
img_list = []
cow_dirs = [f.name for f in os.scandir(image_path) if  f.is_dir()]
for cow in cow_dirs:
    cow_path = os.path.join(image_path, cow)
    files =  [file.replace(".tif", "") for file in os.listdir(cow_path)]
    img_list.extend(files)

unlabeled_img = sorted(list(set(img_list).difference(set(label_list))))
todo_imgs = len(unlabeled_img)

def save_class(label, dst):
    with open(dst, "w") as f:
    # good 1: udder is there
    # bad 0: no udder
        f.write(str(label))
        
print('Instructions: Use g for class good and b for class bad \n Press s to save class,\n r to restart,\n and c to close \n\n')
print(f'There are: {todo_imgs} images to annotate\n')
# open the image with open cv and get user input
close = False
for img_num in list(range(todo_imgs)):
    window_name = 'image_' + str(img_num+1) +' _from_' + str(todo_imgs)
    image_name = unlabeled_img[img_num]
    cow_dir = os.path.join(image_path, image_name.split("_")[0])
    # image path is cow ID + image name
    src = os.path.join(cow_dir, image_name + ".tif")
    dst = os.path.join(label_path, image_name +".txt")
     # open a widow with the ith image
    img = cv2.imread(src)
    clone = img.copy()
    cv2.namedWindow(window_name)
    
    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0)
        
        if key == ord("g"):
            name = 'good'
            label = 1
            cv2.putText(img, f'Class: {name}', (100,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        elif key == ord("b"):
            label = 0
            name = 'bad'
            cv2.putText(img, f'Class: {name}', (100,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                 
        elif key == ord("c"):
            cv2.destroyAllWindows()
            close = True
            break
            
        if key == ord("s"):
                cv2.destroyAllWindows()
                # save label to text file
                save_class(label, dst)
                break
                
        elif key == ord("r"):
            img = clone.copy()
            del label
            
    if close == True:
        break

print(f'You have annotaded: {img_num +1} image(s)\n')
print(f'There are: {todo_imgs - (img_num+1)} image(s) left\n')