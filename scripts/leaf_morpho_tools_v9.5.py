# Standard library imports
import os
import multiprocessing as mp
import argparse
import traceback
import re
from itertools import combinations
import time
import concurrent.futures
import tempfile

# Third-party libraries
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode
from roboflow import Roboflow
import astropy.units as u
from fil_finder import FilFinder2D
from tqdm import tqdm
from scipy.spatial import ConvexHull

# Local application/library specific imports
from plantcv import plantcv as pcv
from qreader import QReader

# Initialize Roboflow API
rf = Roboflow(api_key='l6XPyOniqM4Ecq129cpf')

# Initialize QReader
qreader = QReader()

# Define functions
def three_pronged_qr(gray_img):
    # Save retval
    retval = False
    readby = "FAIL"
    # First attempt with OpenCV
    qcd = cv2.QRCodeDetector()
    retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(gray_img)
    if retval:
        points = points.squeeze().astype(np.int64)
        readby = "OpenCV"
        return retval, decoded_info, points, readby

    # If OpenCV fails, try pyzbar
    decoded_objects = decode(gray_img)
    if decoded_objects:
        for obj in decoded_objects:
            decoded_info = obj.data.decode("utf-8")
            points = np.array([[p.x, p.y] for p in obj.polygon])
            retval = True
            readby = "pyzbar"
            return retval, decoded_info, points, readby

    # If pyzbar fails, try qreader
    decoded_info = qreader.detect_and_decode(image=gray_img)
    decoded_info = decoded_info[0]
    decoded_list = qreader.detect(image=gray_img)
    if decoded_list:
        quad_xy_list = [info['quad_xy'] for info in decoded_list]
        points = np.array(quad_xy_list).astype(np.int64).squeeze()
        retval = True
        readby = "qreader"
        return retval, decoded_info, points, readby

    if not retval:
        return retval, None, None, readby

# Order markers in clockwise order, starting with top left
def order_points_clockwise(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right+most points from the sorted
    # x+roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y+coordinates so we can grab the top+left and bottom+left
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

# Perspective transform function
def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Convert to numpy array for easier manipulation
        corners = np.array(corners, dtype="float32").squeeze()

        # Initialize a list of coordinates that will be ordered
        rect = np.zeros((4, 2), dtype="float32")

        # The top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]  # Top-left has the smallest sum
        rect[2] = corners[np.argmax(s)]  # Bottom-right has the largest sum

        # The top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]  # Top-right has the smallest difference
        rect[3] = corners[np.argmax(diff)]  # Bottom-left has the largest difference
        top_l, top_r, bottom_r, bottom_l = rect[0], rect[1], rect[2], rect[3]
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

# find the farthest white pixels from eachother in skeleton
def find_farthest_points(skeleton):
    # Find the white pixels in the skeleton
    white_pixels = np.argwhere(skeleton == 255)

    # If there are fewer than 2 white pixels, return None
    if len(white_pixels) < 2:
        return None

    # Compute the convex hull of the white pixels
    hull = ConvexHull(white_pixels)
    hull_points = white_pixels[hull.vertices]

    # Find the farthest pair of points on the convex hull
    max_distance = 0
    farthest_points = None
    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            distance = np.linalg.norm(hull_points[i] - hull_points[j])
            if distance > max_distance:
                max_distance = distance
                farthest_points = (hull_points[i], hull_points[j])

    return farthest_points

# Find best corner points from multiple coordinates
def find_best_corner_points(coordinates_array):
    if len(coordinates_array) > 4:
        min_variance = float('inf')
        best_combination = None

        # Iterate through all combinations of four points
        for combination in combinations(coordinates_array, 4):
            combination = np.array(combination)
            # Calculate the pairwise distances
            dists = [np.linalg.norm(combination[i] - combination[j]) for i in range(4) for j in range(i + 1, 4)]
            variance = np.var(dists)
            if variance < min_variance:
                min_variance = variance
                corner_points = combination
        return order_points_clockwise(corner_points)
    else:
        return order_points_clockwise(coordinates_array)

# Get input images from directory
def get_input_images(input_dir):
    try:
        # Ensure the input directory exists
        if not os.path.isdir(input_dir):
            raise ValueError(f"The directory {input_dir} does not exist or is not a directory.")
        else:
            # List all files in the directory
            input_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            return input_images
    except Exception as e:
        return []

# Leaf image process function
def leaf_morpho(input_image, output_dir):
    error_message = None
    file_name = os.path.basename(input_image)
    file_name = os.path.splitext(file_name)[0]

    try:
        # Read image
        image = cv2.imread(input_image, cv2.IMREAD_COLOR)

        # Get segmentation model
        project = rf.workspace().project("morphometric_segmentation")
        model = project.version(2).model

        # Create a temp file path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, image)

        # Run instance segmentation
        results = model.predict(temp_file_path, confidence=50).json()

        # Draw polylines
        for prediction in results['predictions']:
            if prediction['class'] == 'Template':
                points = np.array([[p['x'], p['y']] for p in prediction['points']], dtype=np.int32).reshape((-1, 1, 2))

        copy_img = image.copy()

        # Create a mask with the same dimensions as the image
        mask = np.zeros(copy_img.shape[:2], dtype=np.uint8)

        # Draw the contours on the mask
        cv2.drawContours(mask, [points], -1, (255), thickness=cv2.FILLED)

        # Apply the mask to the image
        masked_img = cv2.bitwise_and(copy_img, copy_img, mask=mask)

        # Convert the masked image to grayscale
        gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        # Find the QR code
        retval, decoded_info, qr_points, readby = three_pronged_qr(gray_img)

        # Extract QR elements
        if not retval:
            print("No QR code found.")
        elif retval:
            if isinstance(decoded_info, tuple):
                decoded_info = decoded_info[0]
                print(decoded_info)
            elif isinstance(decoded_info, str):
                decoded_info = decoded_info
                print(decoded_info)
        else:
            print("Something went wrong trying to find the QR code.")

        # Use regex to extract all text before the first 'x'
        match1 = re.match(r'^(\d+\.?\d*)(?=[a-zA-Z])', decoded_info)
        if match1:
            width = float(match1.group())
            print(width)
        else:
            print("No match found for width.")

        # run the following regex on decoded_info (?<=x)(\d+\.?\d*)(?=[a-zA-Z])
        match2 = re.search(r'(?<=x)(\d+\.?\d*)(?=[a-zA-Z])', decoded_info)
        if match2:
            height = float(match2.group())
            print(height)
        else:
            print("No match found for height.")

        # Use regex to extract all text before the first 'x'
        if width == 6:
            dia = 0.5
        elif width == 10.5:
            dia = 0.5
        elif width == 16:
            dia = 0.75
        elif width == 24:
            dia = 1
        else:
            dia = 'error'

        # Find units
        unit = re.search(r'(in|cm)', decoded_info)
        if unit:
            unit = unit.group(1)

        # Find the corners to perform rotation using object detection - this usese a publicly avaialble roboflow repo located at https://app.roboflow.com/dataset/metrics_marker_detection
        project = rf.workspace().project("metrics_marker_detection")
        model = project.version("3").model

        # Find four markers
        results = model.predict(masked_img, confidence=50, overlap=30).json()

        # Find the center of the markers marking template centroid
        coordinates_array = []
        for prediction in results['predictions']:
            if prediction['class'] == 'Marker':
                x, y = int(prediction['x']), int(prediction['y'])
                w, h = int(prediction['width']), int(prediction['height'])
                # Add the center coordinates to the array
                coordinates_array.append((x, y))

        # Convert coordinates_array to a numpy array
        coordinates_array = np.array(coordinates_array)

        # Find best corner points
        ordered_corner_points = find_best_corner_points(coordinates_array)

        # Assign points
        x1 = np.mean(qr_points[:, 0]).astype(np.int64)
        y1 = np.mean(qr_points[:, 1]).astype(np.int64)
        qr_center = (x1, y1)  # Center of QR code
        center_x = int(np.mean(coordinates_array[:, 0]))
        center_y = int(np.mean(coordinates_array[:, 1]))
        centroid = (center_x, center_y)

        # Find the distance between qr_center and centroid
        distance = np.sqrt((qr_center[0] - centroid[0]) ** 2 + (qr_center[1] - centroid[1]) ** 2)

        # Calculate the differences in the x and y coordinates
        delta_x = centroid[1] - qr_center[1]
        delta_y = qr_center[0] - centroid[0]

        # Calculate the angle in radians
        angle_radians = np.arctan2(delta_y, delta_x)

        # Convert the angle to degrees
        degrees = np.rad2deg(angle_radians)

        # Get rotation matrix
        mat = cv2.getRotationMatrix2D(centroid, degrees, scale=1)

        # Rotate the image
        rotated_img = cv2.warpAffine(masked_img, mat, (masked_img.shape[1], masked_img.shape[0]))

        # Find the markers on the rotated image
        results = model.predict(rotated_img, confidence=50, overlap=30).json()

        # Get coordinates from rotated image
        coordinates = []
        for prediction in results['predictions']:
            if prediction['class'] == 'Marker':
                x = int(prediction['x'])
                y = int(prediction['y'])
                coordinates.append((x, y))

        # Convert the list of coordinates to a NumPy array
        coordinates_array = np.array(coordinates)

        # Convert the image to RGB format
        cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)

        # Highlight the markers on the image
        for coords in coordinates:
            cv2.circle(rotated_img, coords, 5, (0, 0, 0), -1)

        # Find best corner points from rotated image
        ordered_corner_points = find_best_corner_points(coordinates_array)

        # Crop the rotated_image to the expanded_coordinates
        mask = np.zeros(rotated_img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [ordered_corner_points], (255,255,255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            if len(approx) == 4:
                target_box = perspective_transform(rotated_img, approx)
            else:
                print("ERROR: Could not find 4 corners for" + file_name)

        if output_dir is not False:
            # Save the expanded crop to the output directory
            output_path = os.path.join(output_dir, f"{file_name}_target_box.jpg")
            cv2.imwrite(output_path, target_box)

        # Extract dimensions of template
        img_height = target_box.shape[0]
        img_width = target_box.shape[1]

        # Convert either distance to cm
        height_in = img_height / height
        width_in = img_width / width
        avg_in = (height_in + width_in) / 2
        avg_cm = avg_in / 2.54

        # INSERT YOUR OWN MODEL WITHIN THIS SECTION, OR SKIP THIS SECTION TO THRESH AGAINST TEMPALTE BACKGROUND TO EXTRACT OBJECTS
        # Semantic segmentation 
        project = rf.workspace().project("morphometric_segmentation")
        model = project.version(2).model

        # Create a temp file on which to run the model
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, target_box)

        # Run instance segmentation
        results = model.predict(temp_file_path, confidence=50).json()

        # Highlight "Leaf" on target_box
        for prediction in results['predictions']:
            if prediction['class'] == 'Leaf':
                points = np.array([[p['x'], p['y']] for p in prediction['points']], dtype=np.int32).reshape((-1, 1, 2))

        # Create a mask around everything but leaf
        copy_img2 = target_box.copy()

        # Create a mask with the same dimensions as the image
        mask = np.zeros(copy_img2.shape[:2], dtype=np.uint8)

        # Draw the contours on the mask
        cv2.drawContours(mask, [points], -1, (255), thickness=cv2.FILLED)

        # Apply the mask to the image
        masked_img = cv2.bitwise_and(copy_img2, copy_img2, mask=mask)

        # Convert the expanded crop to grayscale
        gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        # SKIP TO HERE AND USE THE FOLLOWING CODE TO THRESHOLD AGAINST THE TEMPLATE BACKGROUND TO EXTRACT OBJECTS
        # USE THE FOLLOWING CODE HERE TO THRESHOLD AGAINST OBJECTS: gray_img = cv2.cvtColor(target_box, cv2.COLOR_BGR2GRAY)
        # Trim the white pixels from around the edge of the leaf
        _, thresh1 = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)

        # Apply as a mask
        masked_data = cv2.bitwise_and(masked_img, masked_img, mask=thresh1)

        # convert all masked data pixels that are not == 0 to 255
        masked_data[masked_data != 0] = 255

        # invert the image
        masked_data = cv2.bitwise_not(masked_data)

        # If directory is not False, save the masked image
        if output_dir is not False:
            # Save the masked image to the output directory
            output_path = os.path.join(output_dir, f"{file_name}_masked_inv.jpg")
            cv2.imwrite(output_path, masked_data)

        # Apply a binary threshold to the grayscale image
        _, bin_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

        # Calculate leaf area
        white_pixels = np.sum(bin_img == 255)
        leaf_area_cm2 = white_pixels / avg_cm**2

        if output_dir is not False:
            # Save masked image to the output directory
            output_path = os.path.join(output_dir, f"{file_name}_masked.jpg")
            cv2.imwrite(output_path, masked_img)

        if error_message is None:
            error_message = "No error"
        source = 0
        data = {
            'sample_id': [file_name],
            'leaf_area_cm2': [leaf_area_cm2],
            'report': [error_message],
            'source': [0]
        }

        df = pd.DataFrame(data)
        return df

    except Exception as e:
        error_message = str(e)
        tb = traceback.extract_tb(e.__traceback__)
        source = {
            'file': tb[-1].filename,
            'line': tb[-1].lineno,
            'code': tb[-1].line
        }
        data = {
            'sample_id': [file_name],
            'leaf_area_cm2': ['error'],
            'report': [error_message],
            'source': [0]
        }

        df = pd.DataFrame(data)
        return df

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze leaf morphometrics from images")
    parser.add_argument('-i', '--input_dir', type=str, default=None, help='Path to input directory of images to be analyzed.')
    parser.add_argument('-o', '--output_dir', default=None, help='Path to output directory for resulting images.')
    parser.add_argument('-r', '--results_path', default=None, help='Desired path and file name for results.')
    parser.add_argument('-w','--workers', default=None, type=int, help='Number of worker processes to use, if nothing is specified half of all available workers will be used.')
    args = parser.parse_args()

    if args.input_dir is not None:
        input_dir = args.input_dir
    else:
        while True:
            input_dir = input("\nPlease enter the path to the the input images to be analyzed: ")
            if os.path.exists(input_dir):
                args.input_dir = input_dir
                break
            else:
                print("Invalid path, please try again.")

    input_images = get_input_images(input_dir)

    if args.output_dir is not None:
        print("\nResulting ouput images will be wrote to:", args.output_dir)
        output_dir = args.output_dir
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
        print(f'\nUsing {num_workers} workers for parallel processing.')
    else:
        # Use a default value
        num_workers = int(mp.cpu_count()/2)
        print(f"\nUsing {num_workers} workers (half of all available) for parallel processing. To change this, enter argument -w myDesiredNumberOfWorkers following the python call.")

    # Process images in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(leaf_morpho, input_image, output_dir) for input_image in input_images]

        # Get the results from the futures
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing images'):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception occurred: {e}")

        # Combine all results from leaf_morpho into a pandas dataframe
        results = pd.concat(results, ignore_index=True)

    # convert results to a csv file
    results.to_csv(results_path, index=True)
