% define directories
currentFolder = pwd;
image_path = strcat(currentFolder, '\frames');
label_path = strcat(currentFolder,'\labels\segments');
box_path = strcat(currentFolder,'\labels\bbox');

img_list = split({dir(fullfile(image_path, '*.tif')).name}.', ".");
label_list = split({dir(fullfile(label_path, '*.txt')).name}.', ".");

if size(label_list, 1)>0
    unlabeled_img = setdiff(img_list(:, 1), label_list(:, 1));
else
    unlabeled_img = img_list;
end

category_id = 0;

for num_img = 1:length(unlabeled_img)

    % read depth image
    source_name = unlabeled_img(num_img);
    origA = imread(fullfile(image_path, source_name + ".tif"));

    % get image size
    im_size = size(origA);
    im_width = im_size(2);
    im_height = im_size(1);
    
    % open segmenter
    done = 0;
    while done == 0
        imageSegmenter(origA)
        done = input('Done? (1/0) : ');
    end

    % get mask polygon
    boundary = bwboundaries(BW, "noholes");
    boundary = cell2mat(boundary);
    
    % get bounding box 
    bbox = regionprops(BW,'BoundingBox');
    % miny, minx, height, width
    bbox = bbox.BoundingBox;
    % x_center y_center width height
    bbox = [(bbox(1) + (bbox(3)/2))/im_width, (bbox(2)+ (bbox(4)/2))/im_height, bbox(3)/im_width, bbox(4)/im_height];

    % write poligon as an array
    segmentation = zeros(1,length(boundary)*2);
    j = 1;
    for i = 1:length(boundary)
        segmentation(1,j) = boundary(i,2)/im_width; %x
        segmentation(1,j+1) = boundary(i,1)/im_height; %y
        j = j + 2;
    end
    
    % save annotation
    annotation= cat(2, category_id, cat(2, bbox, segmentation));
    file_name = fullfile(label_path, source_name+".txt");
    writematrix(annotation, file_name, 'Delimiter',' ')
    % save bbox
    annotation= cat(2, category_id, bbox);
    file_name = fullfile(box_path, source_name+".txt");
    writematrix(annotation, file_name, 'Delimiter',' ')

    % do nex cow?
    next = input('Continue? (1/0) : ');
    if next == 0
       break
    end
    clearvars -except currentFolder image_path label_path unlabeled_img category_id box_path

end


