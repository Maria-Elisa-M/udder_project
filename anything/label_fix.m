% open segment
% define directories
currentFolder = pwd;
label_path = strcat(currentFolder,'\udder_labels\labels\segments');
label_list = dir(fullfile(label_path, '*.txt'));
label_path = strcat(currentFolder,'\labels2\segments');
box_path = strcat(currentFolder,'\labels2\bbox');

im_height =  480;
im_width = 848;
category_id = 0;

for num_img = 1:length(label_list)
    source_name = label_list(num_img).name;

    fileID = fopen(strcat(label_list(num_img).folder,"\", label_list(num_img).name),'r');
    label = fscanf(fileID,"%f");
    fclose(fileID);
    polygon = label(6:end);
    segmentation = polygon'; 
    
    newpol = zeros(int16(length(polygon)/2), 2);
    j = 1;
    for i = 1: length(newpol)
        newpol(i, 1) = polygon(j+1)*im_height; %y
        newpol(i, 2) = polygon(j)*im_width; %x 
        j = j+2;
    end
    bw = poly2mask(newpol(:, 2), newpol(:, 1),  im_height, im_width);
    
    % get bounding box 
    bbox = regionprops(bw,'BoundingBox');
    % miny, minx, height, width
    bbox = bbox.BoundingBox;
    % x_center y_center width height
    bbox = [(bbox(1) + (bbox(3)/2))/im_width, (bbox(2)+ (bbox(4)/2))/im_height, bbox(3)/im_width, bbox(4)/im_height];
    
    % save annotation
    annotation= cat(2, category_id, segmentation);
    file_name = fullfile(label_path, source_name);
    writematrix(annotation, file_name, 'Delimiter',' ')
    % save bbox
    annotation= cat(2, category_id, bbox);
    file_name = fullfile(box_path, source_name);
    writematrix(annotation, file_name, 'Delimiter',' ')
end 