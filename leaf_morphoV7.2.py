import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from plantcv import plantcv as pcv
from functools import partial
import multiprocessing as mp
from contextlib import redirect_stdout
from tqdm import tqdm
import traceback

def rotate_image(image, angle):
    # Grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def order_points_clockwise(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="int32")

def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between 
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between 
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in 
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))
	
def leaf_morpho(input_image, template_width, naive_bayes, pix_per_marker, output_dir):
    try:
        # Read input image
        image = cv2.imread(input_image, cv2.IMREAD_COLOR) 
        original = image.copy()
        original2 = image.copy()
        
        # Read QR code
        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        qcd = cv2.QRCodeDetector()
        retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(image)

        # If QR code is read, save plot ID, if not write error
        if retval is True:
            decoded_info = str(decoded_info)\
                .replace("('", "")\
                .replace("',)", "")
        else:
            # If QR code is not read, save the pathname of image
            decoded_info = os.path.basename(input_image)\
                .rsplit('.')[0]

        ## Naive bayes classifier using partial densifty function to ID leaf and size refs from background
        bayes_mask = pcv.naive_bayes_classifier(rgb_img=image, 
                                    pdf_file=naive_bayes)

        markers = pcv.apply_mask(img=image, mask=bayes_mask["markers"], mask_color='black')
        thresh_val = 150
        transformed = None
        while transformed is None:
            try:
                # Designate ROI around template
                channel_a = pcv.rgb2gray_lab(markers, "a")
                thresh = pcv.threshold.binary(gray_img=channel_a, threshold=thresh_val, object_type='light')
                dil_marker = pcv.dilate(gray_img=thresh, ksize=5, i=1)
                fill = pcv.fill(bin_img=dil_marker, size=(pix_per_marker/4))

                # Find centroid of circle markers
                nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(fill, connectivity=8)

                # Find four stats components areas that are closest to white_pixels
                diffs = np.abs(stats[:, 4] - pix_per_marker)

                # Sort the components by the absolute differences
                sorted_indices = np.argsort(diffs)

                # Extract the centroids of the four components with the smallest absolute differences
                corners = centroids[sorted_indices[:4]]

                for corner in corners:
                    x, y = corner  # Get the coordinate of the centroied
                    x, y = int(round(x)), int(round(y))  # Round and cast to int
                    cv2.drawMarker(image, (x, y), (120, 157, 187), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_8)

                # Crop image to centroids
                c_list = []
                for corner in corners: 
                    x,y = corner.ravel()
                    c_list.append([int(x), int(y)])
                    cv2.circle(image,(int(x),int(y)),5,(36,255,12),-1)

                corner_points = np.array([c_list[0], c_list[1], c_list[2], c_list[3]])
                ordered_corner_points = order_points_clockwise(corner_points)
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [ordered_corner_points], (255,255,255))

                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]

                # Correct camera angle
                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
                    if len(approx) == 4:
                        thresh_markers = thresh_val
                        transformed = perspective_transform(original, approx)
                        break
                    else:
                        thresh_val = thresh_val - 10
                        continue
            except:
                if thresh_val <= 100:
                    transformed = "ERROR"
                    raise ValueError("Could not detect four reference markers for", decoded_info, "please inspect image quality or submit to be added to the training set.")
                else:
                    thresh_val = thresh_val - 10
                    continue

        h, w, c = transformed.shape
        if h > w:
            transformed = rotate_image(transformed, -90)
            h, w, c = transformed.shape

        # Convert pixels per inch to pixels per cm (2.54 conversion factor)
        pix_per_half_inch = int(np.round((w/template_width)/2))

        # create a retangle from the coordinates in corners
        expanded_corners = []
        x, y = ordered_corner_points[0]
        expanded_corners.append([x - pix_per_half_inch, y - pix_per_half_inch])
        x, y = ordered_corner_points[1]
        expanded_corners.append([x + pix_per_half_inch, y - pix_per_half_inch])
        x, y = ordered_corner_points[2]
        expanded_corners.append([x + pix_per_half_inch, y + pix_per_half_inch])
        x, y = ordered_corner_points[3]
        expanded_corners.append([x - pix_per_half_inch, y + pix_per_half_inch])
        # convert expanded corners as 32 int type
        expanded_corners = np.int32(np.array(expanded_corners))

        for corner in expanded_corners:
            x, y = corner  # Get the coordinate of the centroied
            x, y = int(round(x)), int(round(y))  # Round and cast to int
            cv2.drawMarker(image, (x, y), (120, 157, 187), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_8)

        c_list = []
        for corner in expanded_corners: 
            x,y = corner.ravel()
            c_list.append([int(x), int(y)])
            cv2.circle(image,(int(x),int(y)),5,(36,255,12),-1)

        corner_points = np.array([c_list[0], c_list[1], c_list[2], c_list[3]])
        ordered_corner_points = order_points_clockwise(corner_points)
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [ordered_corner_points], (255,255,255))

        # Convert mask to grayscale
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            expanded = perspective_transform(original2, approx)
        else :
            print("ERROR: Could not find 4 corners for", decoded_info)

        h, w, c = expanded.shape
        if h > w:
            expanded = rotate_image(expanded, -90)
            h, w, c = expanded.shape

        if output_dir is not False:
            img_name = decoded_info + '_perspective_adjustment.jpg'
            output_path = os.path.join(output_dir, img_name )
            cv2.imwrite(output_path, expanded, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        bayes_mask = pcv.naive_bayes_classifier(rgb_img=expanded, 
                                                pdf_file=naive_bayes)
        
        markers = pcv.apply_mask(img=expanded, mask=bayes_mask["markers"], mask_color='black')
        channel_a = pcv.rgb2gray_lab(markers, "a")
        thresh = pcv.threshold.binary(gray_img=channel_a, threshold=thresh_markers, object_type='light')
        dil_marker = pcv.dilate(gray_img=thresh, ksize=5, i=1)
        fill = pcv.fill(bin_img=dil_marker, size=(pix_per_marker/4))
        
        # Gaussian blur circles
        blur_marker = cv2.GaussianBlur(fill, (5, 5), 0)
        
        # Detect the circle in marker using hough circle in opencv
        contours, hierarchy = cv2.findContours(blur_marker, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Fit ellipses to the contours
        ellipses = []
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                ellipses.append(ellipse)

        # Extract the major and minor axes of the ellipses
        axes = np.array([ellipse[1] for ellipse in ellipses])

        # Calculate the diameters of the ellipses
        diameter_avg = np.average(np.max(axes, axis=1))
        pix_per_cm_average = diameter_avg / 1.27
        
        # invert bayes_mask["template"] to get the leaf
        background_mask = 255-bayes_mask["background"]

        # invert the marker mask
        marker_mask = 255-fill

        # Combine the masks using the bitwise OR operator
        combined_mask = pcv.apply_mask(expanded, background_mask, mask_color = 'black')
        combined_mask = pcv.apply_mask(combined_mask, marker_mask, mask_color = 'black')

        gray_mask = cv2.cvtColor(combined_mask, cv2.COLOR_BGR2GRAY)

        # thresh gray mask
        bin_leaf = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)[1]

        # Find the area of the largest object in bin_leaf
        nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_leaf, connectivity=8)
        stats = stats[1:]
        stats = stats[np.argmax(stats[:, 4])]
        area = stats[4]

        # Fill all holes smaller than half the largest object
        bin_leaf = pcv.fill(bin_leaf, size=(area/2))
        
        # Find contours in bin_leaf
        contours, hierarchy = cv2.findContours(bin_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask with the same size as the binary image
        bin_leaf = np.zeros_like(bin_leaf)

        # Draw the contours on the mask
        cv2.drawContours(bin_leaf, contours, -1, 255, -1)

        # Fill the interior of the contours with white pixels
        cv2.fillPoly(bin_leaf, contours, 255)

        # Apply the mask to the image
        leaf = pcv.apply_mask(expanded, bin_leaf, mask_color = 'black')

        # Blur and dilate to track tip points
        dil = pcv.dilate(bin_leaf, ksize=5, i=5)
        blur = pcv.median_blur(dil, ksize=(20,20))

        # Create skeleton of leaf
        skeleton = pcv.morphology.skeletonize(mask=blur)
        pruned_skel, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skeleton, size=(pix_per_cm_average*4))

        # Find leaflet tip points
        tip_pts = pcv.morphology.find_tips(skel_img=pruned_skel, mask=bin_leaf, label="default")

        # Record lowest and second lowest points as roi
        y1, x1 = np.argwhere(tip_pts > 0)[-1] # Lowest point on petiole

        # Find where petiole meets the leaf
        branch_pts = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=blur, label="default")
        
        try:
            # Find lowest point in branch_pts
            y2, x2 = np.argwhere(branch_pts > 0)[-1]
            # Find the first the coordinates of the first black pixel below y1
            y3 = y1 + np.argmax(blur[y1:, x1] == 0)
            # Find the first the coordinates of the first black pixel below y2
            y4 = y2 + np.argmax(blur[y2:, x2] == 0)-1
        # If there are no branch points, lower prune rate and define leaf further
        except:
            skeleton = pcv.morphology.skeletonize(mask=bin_leaf)
            pruned_skel, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skeleton, size=(pix_per_cm_average))
            tip_pts = pcv.morphology.find_tips(skel_img=pruned_skel, mask=bin_leaf, label="default")
            y1, x1 = np.argwhere(tip_pts > 0)[-1] # Lowest point on petiole
            # Find branch points of blur
            branch_pts = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=bin_leaf, label="default")
            # Find lowest point in branch_pts
            y2, x2 = np.argwhere(branch_pts > 0)[-1]
            # Find the first the coordinates of the first black pixel below y1
            y3 = y1 + np.argmax(bin_leaf[y1:, x1] == 0)
            # Find the first the coordinates of the first black pixel below y2
            y4 = y2 + np.argmax(bin_leaf[y2:, x2] == 0)-1

        if (y3-y4) < ((y1-y2)/2):
            dist_transform = cv2.distanceTransform(bin_leaf,cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            thresh_val = 0.05* dist_transform.max()
            rows = dist_transform[y2+1:, x2:x2+1]
            i = 0
            dist_transform = cv2.distanceTransform(bin_leaf,cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            while True:
                if rows[i, 0] <= thresh_val:
                    y4 = y2+i+1
                    break
                else:
                    i = i + 1
                    continue
        else:
            dist_transform = cv2.distanceTransform(bin_leaf,cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
           
        try:
            # Petiole tip offset
            xDiff = abs(x1 - x2)*2

            if  xDiff < 50:
                xDiff = xDiff + pix_per_half_inch

            # Create a subimage of dist_tranform with the coordinates in vertices
            subimage = dist_transform[y4:y3, x1-xDiff:x1+xDiff]

            # Calculate the threshold for white pixels
            white_threshold = subimage.shape[0] * 0.5

            # Calculate rows in subimage that contain white pixels
            white_rows = 0
            for row in subimage:
                if np.sum(row) > 0:
                    white_rows += 1

            # Initialize base dist_val to 0.09
            dist_val = 0.09

            # Loop until less than 50% of rows contain white pixels to eliminate petiole
            while white_rows >= white_threshold:

                # Calculate the thresholded image using the current dist_val
                ret, thresh = cv2.threshold(subimage, dist_val*dist_transform.max(), 255, 0)

                # Count the number of rows with white pixels
                white_rows = 0
                for row in thresh:
                    if np.sum(row) > 0:
                        white_rows += 1

                # Increase dist_val by 0.01 for the next iteration
                dist_val += 0.01
            
                thresh.shape == bin_leaf[y4:y3, x1-xDiff:x1+xDiff].shape
                bin_leaf[y4:y3, x1-xDiff:x1+xDiff] = thresh

                # Find the area of the largest object in bin_leaf
                nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_leaf, connectivity=8)
                stats = stats[1:]
                stats = stats[np.argmax(stats[:, 4])]
                area = stats[4]

                # Fill all holes smaller than stats
                bin_leaf = pcv.fill(bin_leaf, size=(area/10))

        except:
            h, w = bin_leaf.shape
            # Create a rectangular roi on bin_leaf using x1, y1, x2, y2
            roi = bin_leaf[y4:y3, 0:w]
            # Convert all white pixels in roi to black
            roi[roi > 0] = 0
            # Apply the mask to the orignal location on bin_leaf
            bin_leaf[y4:y3, 0:w] = roi
        

        leaf_trimmed = pcv.apply_mask(img=leaf, mask=bin_leaf, mask_color="black")

        # Find the area of the largest object in bin_leaf
        nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_leaf, connectivity=8)
        stats = stats[1:]
        stats = stats[np.argmax(stats[:, 4])]
        area = stats[4]

        # Fill all holes smaller than stats
        bin_leaf = pcv.fill(bin_leaf, size=(area/10))

        # Mask the outline of leaf to show only the venation
        outline = pcv.canny_edge_detect(img=bin_leaf)
        outline2 = pcv.dilate(outline, ksize=3, i = 1)
        invert = cv2.bitwise_not(outline2)

        # Use canny edge detection to show leaf veins for detecting serations
        edges = pcv.canny_edge_detect(img=leaf_trimmed)
        ven = pcv.apply_mask(edges, invert, 'black')
        dil_leaf = pcv.dilate(ven, ksize = 3, i = 1)
        ven2 = pcv.erode(gray_img=dil_leaf, ksize=3, i=1)

        # Create skeleton of leaf for venation
        skeleton = pcv.morphology.skeletonize(mask=ven2)
        pruned_skel, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=10, mask=edges)

        # Find tip points of viens (serrations)
        serr_pts = pcv.morphology.find_tips(skel_img=pruned_skel, mask=edges, label="default")

        # Calulate tip and serration points
        tip_count = np.count_nonzero(tip_pts) - 1 # Subtract petiole tip point
        serr_count = np.count_nonzero(serr_pts) - np.count_nonzero(tip_pts)

        # Measure leaf size metrics. num_objects is set to 10 in case a leaflet is separated from the overall structure
        #silence output from the below code
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull):
                vis_area = pcv.visualize.obj_sizes(img=leaf, mask=bin_leaf, num_objects=10)
        area = pcv.analyze.size(vis_area, bin_leaf, n_labels=1, label="leaf_") 

        if output_dir is not False:
            img_name = decoded_info + '_size_metrics.jpg'
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, vis_area, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Calculate area in cm2, and height and width in cm
        width_cm = pcv.outputs.observations['leaf_1']['width']['value'] / pix_per_cm_average
        height_cm = pcv.outputs.observations['leaf_1']['height']['value'] / pix_per_cm_average
        area_cm2 = pcv.outputs.observations['leaf_1']['area']['value'] / pix_per_cm_average**2

        # Find perimeter of leaf
        contours, hierarchy = cv2.findContours(bin_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Calculate the perimeter distance in pixels
            perimeter = cv2.arcLength(contour, True)
            perimeter_cm = (perimeter/pix_per_cm_average)

            # Create a dataframe of results
            data = np.array([(decoded_info, area_cm2, perimeter_cm, height_cm, width_cm, serr_count, tip_count)])
            df = pd.DataFrame(data, columns=['sampleID', 'area_cm2', 'perimeter_cm', 'height_cm', 'width_cm','serr_count', 'tip_count'])
            return df

    except Exception as e:
        decoded_info = os.path.basename(input_image)\
                .rsplit('.')[0]
        print(f'Error processing {input_image}: {e}')
        print("Error on line:", traceback.extract_tb(e.__traceback__)[0].lineno)
        data = np.array([(decoded_info, 'error', 'error', 'error', 'error', 'error', 'error')])
        df = pd.DataFrame(data, columns=['sampleID', 'area_cm2', 'perimeter_cm', 'height_cm', 'width_cm', 'serr_count', 'tip_count']) 
        return df
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description= "Analyze leaf morphometrics from images")
    parser.add_argument('-i', '--input_dir', default=None, help='Path to input directory of images to be analyzed.')
    parser.add_argument('-b', '--background', default=None, help='An integer denoting the size background template was used for these images as follows. 1: 12x12, 2: 12x16, 3: 16x16')
    parser.add_argument('-t', '--tools_dir', default=None, help='Path to leaf_morpho_tools file.')
    parser.add_argument('-o', '--output_dir', default=None, help='Path to output directory for resulting images.')
    parser.add_argument('-r', '--results_path', default=None, help='Desired path and file name for results.')
    parser.add_argument('-w','--workers', default=None, type=int, help='Number of worker processes to use, if nothing is specified half of all available workers will be used.')
    args = parser.parse_args()
    
    print("\nWelcome to leaf morphometric analysis tools version 7.2. Follow python call with [-h] for help if needed. For questions email aja294@cornell.edu.")

    if args.input_dir is not None:
        print("\nInput images found in:", args.input_dir)
        input_dir = args.input_dir
    else:
        while True:
            input_dir = input("\nPlease enter the path to the the input images to be analyzed: ")
            if os.path.exists(input_dir):
                args.input_dir = input_dir
                break
            else:
                print("Invalid path, please try again.")

    input_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and "calibrate" not in f and (not f.endswith('.DS_Store'))]
    
    for file in os.listdir(input_dir):
            if "calibrate" in file:
                print("\nPixel calibration image confirmed:", os.path.basename(file))
                calibrate = cv2.imread(os.path.join(args.input_dir, file), cv2.IMREAD_COLOR)
                #if calibrate is not found print an error and exit
                if calibrate is None:
                    print("Calibration image not found, please ensure the image is in the input directory and is titled 'calibrate'.")
                    sys.exit()
    
    template = [9.5, 14.5, 14.5]
    sizes = ['12 x 12in', '16 x 12in', '16 x 16in']
    if args.background is not None:
      index = args.template - 1
      template_width = template[index]
      print("\nBackground template size confirmed:", sizes[index])
    else:
        while True:
            template_width = int(input("\nSelect option 1, 2, or 3 that designates the size of the template used for the images being analyzed.\n1) 12 x 12in, 2) 16 x 12in, or 3) 16 x 16in:  "))
            if template_width == 1 or template_width == 2 or template_width == 3:
                index = template_width - 1
                template_width = template[index]
                print("\nBackground template size confirmed:", sizes[index])
                break
            else:
                print("Invalid input, please enter 1) for 12 x 12in, 2) for 12 x 16in, or 3) for 16 x 16in.")

    if args.tools_dir is not None:
        for root, dirs, files in os.walk(args.tools_dir):
            if 'naive_bayes.txt' in files:
                file_path = os.path.join(root, 'naive_bayes.txt')
                print("\nFound probabibility density function for naive bayes classifier:", file_path) 
                naive_bayes = file_path
    else:
        try:
            print("\nSearching for probability density function in morpho_tools directory...")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            for root, dirs, files in os.walk(script_dir):
                if 'naive_bayes.txt' in files:
                # Found the file, print its path
                    file_path = os.path.join(root, 'naive_bayes.txt')
            print("\nFound probabibility density function for naive bayes classifier in morpho_tools directory:", file_path) 
            naive_bayes = file_path
        except:
            print("No probabibility density function for naive bayes classifier found, ensure leaf_morpho.py and naive_bayes.txt exsist in the same directory, and if not, please add argument -t myPath/to/morho_tools to specify path to the morpho_tools directory that contains naive_bayes.txt.")
            sys.exit()

    if args.output_dir is not None:
        print("\nResulting ouput images will be wrote to:", args.output_dir)
    else:
        while True:
            output_dir = input("\nWould you like to save all output images (WARNING: Resulting file may be large)?\n(y/n): ")
            if output_dir == 'y':
                while True:
                    #ask user if they have a directory or would like to make one
                    output_dir = input("\nWould you like to: (a) Save the images to a known directory? or (b) Make a new folder in the current directory?\n(a/b): ")
                    if output_dir == 'a':
                        while True:
                            # Ask user the pathname to the directory
                            output_dir = input("\nPlease enter the path to the the known image output directory for resulting images: ")
                            if os.path.isdir(output_dir):
                                print("\nIntended output directory confirmed:", output_dir)
                                break
                            else:
                                print("\nIntended output directory not found, please try again.")
                                continue
                    elif output_dir == 'b':
                            output_dir = input("\nPlease enter the desired name of the new image output directory: ")
                            if output_dir != '':
                            # if name was provided do the following loop
                                os.mkdir(output_dir)
                                #get path of new output_dir
                                output_dir = os.path.join(os.getcwd(), output_dir)
                                print("\nNew directory created, images will be saved to:", output_dir)
                                break
                    else:
                        print("Invalid input, please try again.")
                        continue
                    break
                break
            elif output_dir == 'n':
                print("\nNo output directory for images designated resulting images will not be saved.") 
                output_dir = False
                break
            else:
                print("Invalid input, please try again.")
                continue

    if args.results_path is not None:
        print("\nResults will be wrote to:", args.results_path)
        results_path = args.results_path
    else:
        print("\nResults will be saved in a csv in the current directory. To save to a different directory, follow the python call with -r my/intended/directory/myResults.csv.")
        while True:
            results_path = input("\nWould you like the resulting csv to be named using the default result file name - 'leaf_morpho_results.csv'?\n(y/n): ")
            if results_path == 'y':
                results_path = os.path.join(os.getcwd(), 'leaf_morpho_results.csv')
                break
            elif results_path == 'n':
                results_path = input("\nPlease enter the desired name of the results file: ")
                if results_path.endswith('.csv'):
                    results_path = os.path.join(os.getcwd(), results_path)
                    break
                else:
                    results_path = results_path + '.csv'
                    results_path = os.path.join(os.getcwd(), results_path)
                    break
            else:
                print("Invalid input, please try again.")
                continue
            break
   
    if args.workers is not None:
    # Use the value of the --workers argument
        num_workers = args.workers
        print(f'\nUsing  {num_workers} workers for parrallel processing.')
    else:
        # Use a default value
        num_workers = int(mp.cpu_count()/2)
        print("\nUsing half of all available workers for parrallel processing, to change this, enter arguement -w myDesiredNumberOfWorkers following the python call.")

    bayes_mask = pcv.naive_bayes_classifier(rgb_img=calibrate, 
                                            pdf_file=naive_bayes)

    # Find the average pixels per marker using a calibrate image
    markers = pcv.apply_mask(img=calibrate, mask=bayes_mask["markers"], mask_color='black')
    channel_a = pcv.rgb2gray_lab(markers, "a")
    thresh = pcv.threshold.binary(gray_img=channel_a, threshold=150, object_type='light')
    dil_marker = pcv.dilate(gray_img=thresh, ksize=5, i=1)
    fill = pcv.fill(bin_img=dil_marker, size=4)
    pix_per_marker = np.sum(fill == 255)/4
        
    # Create a progress bar
    with mp.Pool(num_workers) as pool:
        # Call the leaf_morpho function for each input image
        results = []
        for input_image in input_images:
            result = pool.apply_async(leaf_morpho, args=(input_image, template_width, naive_bayes, pix_per_marker, output_dir))
            results.append(result)

        # Get the results from the multiprocessing tasks
        output = []
        for result in tqdm(results, total=len(results), desc='Processing images'):
            output.append(result.get())

        #combine all results from leaf_morpho into a pandas dataframe
        results = pd.concat(output, ignore_index=True)

    # convert results to a csv file
    results.to_csv(results_path, index=False)
    















