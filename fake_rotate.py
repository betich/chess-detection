import cv2
import numpy as np
import datetime
from google.oauth2 import service_account
import json

# DRAWING

def show_cv2_image(image, title='image'):
  # plt.figure()
  # plt.title(title)
  # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  cv2.imwrite(f'output/temp/{title}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png', image)

# OCR

with open('service_gaccount.json') as source:
    info = json.load(source)
vision_credentials = service_account.Credentials.from_service_account_info(info)
  
def ocr_image(input_image, verbose=False):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient(credentials=vision_credentials)
    
    content = cv2.imencode('.jpg', input_image)[1].tobytes()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    if verbose:
        print("Texts:")

        for text in texts:
            print(f'\n"{text.description}"')

            vertices = [
                f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
            ]

            print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    
    return texts
  
def detection_to_dict(detection):
    return {
        'description': detection.description,
        'vertices': [
            (vertex.x, vertex.y) for vertex in detection.bounding_poly.vertices
        ]
    }
    
def draw_box(image, a, b, c, d):
  cv2.polylines(image, [np.array([a, b, c, d], np.int32)], True, (0, 255, 0), 2)
  
def show_image_with_ocr(image, title='ocr result'):
  result = ocr_image(image)
  
  if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

  for text in result:
    text_dict = detection_to_dict(text)
    vertices = text_dict['vertices']
    draw_box(image, vertices[0], vertices[1], vertices[2], vertices[3])

  show_cv2_image(image, title)
  
def show_image_with_ocr_labelled(image, title='ocr result'):
  result = ocr_image(image)
  
  if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  
  for text in result:
    text_dict = detection_to_dict(text)
    vertices = text_dict['vertices']
    draw_box(image, vertices[0], vertices[1], vertices[2], vertices[3])
    cv2.putText(image, text_dict['description'], vertices[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  
  show_cv2_image(image, title)

# PROCESS IMAGE
  
# Function to compute the intersection of two lines
def compute_intersection(line1, line2):
  rho1, theta1 = line1
  rho2, theta2 = line2

  # Calculate the intersection of two lines
  A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
  b = np.array([rho1, rho2])

  # Solve the linear system to find the intersection point
  intersection = np.linalg.solve(A, b)
  return int(intersection[0]), int(intersection[1])

def draw_white_board_boundaries(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of green color in HSV
    lower_green = np.array([40, 25, 40])  # Lower bound of green in HSV
    upper_green = np.array([100, 200, 200])  # Upper bound of green in HSV

    # Threshold the image to get only the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black canvas to draw contours
    contour_image = np.zeros_like(mask)

    # Draw the contours on the black canvas (255 for white contours)
    cv2.drawContours(contour_image, contours, -1, (255), 1)

    # Apply the Canny edge detector on the contour image
    edges = cv2.Canny(contour_image, 50, 150, apertureSize=3)

    # Apply Hough Line Transform to find lines in the edge-detected image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Separate the lines into vertical and horizontal based on their angle (theta)
    vertical_lines = []
    horizontal_lines = []

    # Find vertical and horizontal lines
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # Identify vertical lines (theta near 0 or 180 degrees)
            if np.abs(theta) < np.pi / 180 * 10 or np.abs(theta - np.pi) < np.pi / 180 * 10:
                vertical_lines.append((rho, theta))
            # Identify horizontal lines (theta near 90 degrees)
            elif np.abs(theta - np.pi / 2) < np.pi / 180 * 10:
                horizontal_lines.append((rho, theta))

    # Create an empty list to store intersection points
    intersection_points = []

    # Find intersection points between vertical and horizontal lines
    for v_line in vertical_lines:
        for h_line in horizontal_lines:
            intersection = compute_intersection(v_line, h_line)
            intersection_points.append(intersection)

    # Create an empty image to draw the intersection points
    intersection_image = np.zeros_like(image)

    # Draw the intersection points on the image (red points)
    for point in intersection_points:
        cv2.circle(intersection_image, point, 10, (0, 0, 255), -1)  # Red circle at intersection points

    # Convert the intersection image to grayscale
    grayscale_image = cv2.cvtColor(intersection_image, cv2.COLOR_BGR2GRAY)

    # Find contours of the red intersection points (non-zero pixels)
    contours, _ = cv2.findContours(grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty black image to draw the center points
    center_image = np.copy(image)

    # List to store the center points' coordinates
    center_points = []

    # Iterate over each contour and find the centroid (center point)
    for contour in contours:
        # Calculate the moments of the contour
        moments = cv2.moments(contour)
        
        # Calculate the centroid (center) of the contour
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Draw the center point (blue) on the original image
            cv2.circle(center_image, (cx, cy), 5, (255, 0, 0), -1)  # Blue circle at center
            
            # Store the center coordinates in the list
            center_points.append((cx, cy))
    center_points = sorted(center_points, key=lambda x: sum(x))

    # Step 1: Find the convex hull of the center points
    center_points_np = np.array(center_points, dtype=np.int32)  # Convert to NumPy array
    hull = cv2.convexHull(center_points_np)  # Compute convex hull

    # Step 2: Create a mask for the filled polygon
    mask = np.ones_like(image, dtype=np.uint8) * 255  # Create a white mask with the same dimensions as the image
    cv2.fillPoly(mask, [hull], (0, 0, 0))  # Draw the filled polygon in black on the mask
    
    image[mask == 0] = 255
    result_image = image

    return result_image

def process_image(image):
  processed_image = draw_white_board_boundaries(image)
  
  result = processed_image
  return result

# detect chess board orientation

def bounding_box(left, top, width, height):
  return [(left, top), (left + width, top), (left + width, top + height), (left, top + height)]

def draw_bounding_box(image, box, color=(0, 255, 0)):
  cv2.polylines(image, [np.array(box)], isClosed=True, color=color, thickness=2)
  return image

def get_board_orientation(bound_8, bound_h):
  # compute the center of the two bounding boxes
  center_8 = np.mean(np.array(bound_8), axis=0)
  center_h = np.mean(np.array(bound_h), axis=0)
  
  # check the relative position of the two centers
  if center_8[0] < center_h[0] and center_8[1] < center_h[1]:
    return 'UPRIGHT'
  elif center_8[0] > center_h[0] and center_8[1] > center_h[1]:
    return 'UPSIDE_DOWN'
  elif center_8[0] < center_h[0] and center_8[1] > center_h[1]:
    return 'ROTATED_RIGHT'
  elif center_8[0] > center_h[0] and center_8[1] < center_h[1]:
    return 'ROTATED_LEFT'
  

def detect_chessboard_orientation(image):
  data = ocr_image(image)
  
  bounds_8 = [] # sample: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
  bounds_h = [] # sample: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
  
  for detection in data:
    data_dict = detection_to_dict(detection)
    if data_dict['description'] == '8':
      # store the bounding box of the 8
      bounds_8.append(data_dict['vertices'])
    elif data_dict['description'] == 'h':
      bounds_h.append(data_dict['vertices'])
  
  # find the closest pair of 8 and h
  closest = None
  
  for bound_8 in bounds_8:
    for bound_h in bounds_h:
      distance = np.linalg.norm(np.array(bound_8) - np.array(bound_h))
      
      if closest is None or distance < closest[0]:
        closest = (distance, bound_8, bound_h)
        
  print(f'Closest pair: {closest}')
  
  if closest is None:
    return False, closest
  
  # compute the center of the two bounding boxes
  center_8 = np.mean(np.array(closest[1]), axis=0)
  center_h = np.mean(np.array(closest[2]), axis=0)
  
  # draw the line between the two points
  cv2.line(image, tuple(center_8.astype(int)), tuple(center_h.astype(int)), (255, 0, 0), 2)
  
  # draw the bounding boxes
  image = draw_bounding_box(image, bound_8, (0, 255, 0))
  image = draw_bounding_box(image, bound_h, (0, 0, 255))
  
  show_cv2_image(image, "detected image")
    
  return get_board_orientation(closest[1], closest[2]), closest

def get_k(image):  
  images = [image]
  
  for i in range(3):
    images.append(cv2.rotate(images[-1], cv2.ROTATE_90_CLOCKWISE))
    
  k = 0
  for i in range(4):
    result = detect_chessboard_orientation(process_image(images[i]))
    
    if result[0] and result[1]:
      bound_8 = result[1][1][0]
      bound_h = result[1][1][1]
      
      # bound_8 and bound_h are in the bottom-left window of the image
      h, w = images[i].shape[:2]
      
      if bound_8[0] < w / 2 and bound_8[1] > h / 2 and bound_h[0] < w / 2 and bound_h[1] > h / 2:
        k = i
        break
        
  return k