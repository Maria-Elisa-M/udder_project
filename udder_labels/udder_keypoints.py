import cv2
import os

# directory stuff
image_path = os.path.join(os.getcwd(), 'frames')
box_path = os.path.join(os.getcwd(), r'labels\bbox')
label_path = os.path.join(os.getcwd(), r'labels\keypoints') 

# list of images
img_list = [file.replace(".tif", "") for file in  os.listdir(image_path)]
# list of annotated images
label_list = [file.replace(".txt", "") for file in  os.listdir(label_path)]
# images to annotate
unlabeled_img = list(set(img_list).difference(set(label_list)))

keyPts = ['LF', 'RF', 'LR', 'RR']
num_kp = len(keyPts)

# print instructions
print(f'Instructions:\n Press s to save annotated points,\n r to restart,\n f to skip a point,\n and c to close \n\n')
todo_imgs = len(unlabeled_img)
print(f'There are: {todo_imgs } images to annotate\n')

# function to save keypoints
def click_and_save(event, x, y, flags, param):
    global refPt, count, img
    if event == cv2.EVENT_LBUTTONDOWN:
        im_width = img.shape[1]
        im_height = img.shape[0]
        name = keyPts[count]
        Pt = [x/im_width, y/im_height, 2]
        refPt[count] = Pt
        cv2.circle(img, (x,y), 3, (255,255,255), -1)
        cv2.putText(img, f'{name}', (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        count = count + 1 

        # fuction to save annotation to a table
def save_points(img_name, refPt):
    global label_path, box_path
    file_name = os.path.join(label_path, img_name+ ".txt")
    bbox_name =  os.path.join(box_path, img_name+ ".txt")
    with open(bbox_name, "r") as f:
        bbox = f.read().replace("\n", "")
        
    refPt_long = [str(pt) for p in refPt for pt in p]
    annotation = " ".join(refPt_long)
    
    with open(file_name, "w") as f:
        text = bbox + " "+ annotation
        f.write(text)

# this is the annotating part!
close = False
for img_num in list(range(todo_imgs)):
    refPt = [[] for x in range(num_kp)]
    count = 0
    skips = []
    window_name = 'image_' + str(img_num+1) +' _from_' + str(todo_imgs)
    image_name = unlabeled_img[img_num]
    src = os.path.join(image_path, image_name +".tif")
    img = cv2.imread(src)
    clone = img.copy()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_and_save)
    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1)
        if key == ord("f"):
            Pt = [0, 0, 0]
            refPt[count] = Pt
            skips.append([count])
            count = count +1    
        elif key == ord("s"):
            # save points 
            cv2.destroyAllWindows()
            # save points 
            save_points(image_name, refPt)
            break
        elif key == ord("r"):
            img = clone.copy()
            refPt = [[] for x in range(num_kp)]
            count = 0
            skips = []
        elif key == ord("c"):
            cv2.destroyAllWindows()
            close = True
            break
    if close == True:
        break

print(f'You have annotaded: {img_num +1} image(s)\n')
print(f'There are: {todo_imgs - (img_num+1)} image(s) left\n')