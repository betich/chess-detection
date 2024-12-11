def create_pgn_csv(video_list, pgn_list, output_csv_path):
    # Prepare the data for the CSV
    rows = [{"row_id": video, "output": pgn} for video, pgn in zip(video_list, pgn_list)]

    # Write to the CSV
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["row_id", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV file has been created at {output_csv_path}.")
    
    
import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import datetime

# DRAWING

def show_cv2_image(image, title='image'):
  # plt.figure()
  # plt.title(title)
  # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  cv2.imwrite(f'output/temp/{title}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png', image)

# OCR
  
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
  
  
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import chess
import time
import chess.pgn
import mediapipe as mp
from statistics import mode

# Define a mapping of YOLO labels to chess piece names
def label_to_piece_name(label):
    piece_map = {
        1: "bB",  # black-bishop
        2: "bK",  # black-king
        3: "bN",  # black-knight
        4: "bP",  # black-pawn
        5: "bQ",  # black-queen
        6: "bR",  # black-rook
        7: "wB",  # white-bishop
        8: "wK",  # white-king
        9: "wN",  # white-knight
        10: "wP", # white-pawn
        11: "wQ", # white-queen
        12: "wR"  # white-rook
    }
    return piece_map.get(label, "?")

# Function to find intersection points of the grid lines
def find_grid(image, k=0):
    # image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    for i in range(k):
        hsv = cv2.rotate(hsv, cv2.ROTATE_90_CLOCKWISE)
    
    # Define the range of green color in HSV (assuming green grid lines)
    lower_green = np.array([40, 25, 40])   # Lower bound of green in HSV
    upper_green = np.array([100, 200, 200]) # Upper bound of green in HSV

    # Threshold the image to get only the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black canvas to draw contours
    contour_image = np.zeros_like(mask)

    # Draw the contours on the black canvas (255 for white contours)
    cv2.drawContours(contour_image, contours, -1, (255), 1)

    # Apply Canny edge detector
    edges = cv2.Canny(contour_image, 50, 150, apertureSize=3)

    # Apply Hough Line Transform to find lines in the edge-detected image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Separate the lines into vertical and horizontal based on their angle
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

    # Function to compute the intersection of two lines
    def compute_intersection(line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
        b = np.array([rho1, rho2])

        # Solve the linear system to find the intersection point
        intersection = np.linalg.solve(A, b)
        return int(intersection[0]), int(intersection[1])

    # Find intersection points between vertical and horizontal lines
    for v_line in vertical_lines:
        for h_line in horizontal_lines:
            intersection = compute_intersection(v_line, h_line)
            intersection_points.append(intersection)

    return intersection_points

# Function to map detected chess pieces to the board using intersection points

def map_yolo_results_to_chessboard(results, chessboard_corners):
    """
    Maps YOLO detection results to a chessboard grid.

    Args:
    - results: YOLO detection results containing labels and bounding boxes.
    - chessboard_corners: List of tuples [(x1, y1), (x2, y2), ..., (x4, y4)]
                          representing the corners of the chessboard
                          (top-left, top-right, bottom-left, bottom-right).

    Returns:
    - A formatted string representation of the chessboard with mapped pieces.
    """
    # Extract the chessboard corners
    top_left, top_right, bottom_left, bottom_right = chessboard_corners

    # Calculate the width and height of each cell
    cell_width = (top_right[0] - top_left[0]) / 8
    cell_height = (bottom_left[1] - top_left[1]) / 8

    # Initialize an empty 8x8 chessboard
    board = [['' for _ in range(8)] for _ in range(8)]

    # Process YOLO results
    for r in results:
        boxes = r.boxes.xywh.numpy()  # Bounding boxes in (x_center, y_center, width, height)
        labels = r.boxes.cls.numpy()  # Class indices
        
        for box, label in zip(boxes, labels):
            x_center, y_center, _, _ = box
            piece_name = label_to_piece_name(int(label))

            # Determine the row and column based on the center point
            col = int((x_center - top_left[0]) / cell_width)
            row = int((y_center - top_left[1]) / cell_height)

            # Ensure row and col are within bounds
            if 0 <= row < 8 and 0 <= col < 8:
                board[row][col] = piece_name

    # Format the board for display
    formatted_board = '\n'.join([' '.join([cell if cell else '--' for cell in row]) for row in board])
    return formatted_board

def convert_to_valid_fen(board_string):
    # Mapping of custom pieces to FEN standard pieces
    piece_mapping = {
        "wP": "P", "wR": "R", "wN": "N", "wB": "B", "wQ": "Q", "wK": "K",
        "bP": "p", "bR": "r", "bN": "n", "bB": "b", "bQ": "q", "bK": "k",
        "--": "1"  # Empty squares
    }
    
    # Split the input string into rows
    rows = board_string.strip().split("\n")
    
    fen_rows = []
    for row in rows:
        squares = row.split()  # Split the row into individual squares
        fen_row = ""
        for square in squares:
            fen_row += piece_mapping.get(square, square)  # Replace with mapped value
        
        # Compress consecutive digits (empty spaces) into single numbers
        compressed_row = ""
        empty_count = 0
        for char in fen_row:
            if char.isdigit():  # Count empty squares
                empty_count += int(char)
            else:
                if empty_count > 0:
                    compressed_row += str(empty_count)
                    empty_count = 0
                compressed_row += char
        if empty_count > 0:
            compressed_row += str(empty_count)  # Add remaining empty squares
        fen_rows.append(compressed_row)
    
    # Combine rows with "/" and add default metadata
    fen_board = "/".join(fen_rows)
    fen_metadata = " w - - 0 1"  # White to move, no castling, no en passant
    return fen_board + fen_metadata

def rotate_fen(fen):
    # Split the FEN into board state and other details
    board, *rest = fen.split(' ')
    
    # Split the board into rows
    rows = board.split('/')
    
    # Rotate each row (reverse the pieces) and then reverse the row order
    rotated_rows = [''.join(reversed(row)) for row in reversed(rows)]
    
    # Recombine the rows into the rotated FEN
    rotated_board = '/'.join(rotated_rows)
    
    # Combine the rotated board with the rest of the FEN details
    return ' '.join([rotated_board] + rest)

def convert_to_san(moves):
    san_moves = []
    move_number = 1

    for i in range(0, len(moves), 2):
        if i + 1 < len(moves):
            # Pair moves for each turn
            san_moves.append(f"{move_number}. {moves[i]} {moves[i+1]}")
        else:
            # If there's an odd move at the end, only record that
            san_moves.append(f"{move_number}. {moves[i]}")
        move_number += 1

    return " ".join(san_moves)


def get_chessboard_corners(image, k=0):
    coor = find_grid(image, k)
    coor = sorted(coor, key=lambda x: sum(x))
    min = coor[0]
    max = coor[-1]

    chessboard_corners = [(min[0], min[1]), (max[0], min[1]), (min[0], max[1]), (max[0], max[1])]

    return chessboard_corners

def board_list_to_list(board_list):
    output = [[[],[],[],[],[],[],[],[]],
              [[],[],[],[],[],[],[],[]],
              [[],[],[],[],[],[],[],[]],
              [[],[],[],[],[],[],[],[]],
              [[],[],[],[],[],[],[],[]],
              [[],[],[],[],[],[],[],[]],
              [[],[],[],[],[],[],[],[]],
              [[],[],[],[],[],[],[],[]]]


    for board in board_list:
        board_splitR = board.split("\n")
        for r,board_row in enumerate(board_splitR):
            board_pos = board_row.split(" ")
            for c,piece in enumerate(board_pos):
                output[r][c].append(piece)

    for r,row in enumerate(output):
        for c,col in enumerate(row):
            output[r][c] = mode(col)

    return output

def board_to_pgn(prev_b, curr_b):
    board_pos_coor = [['a8', 'b8', 'c8', 'd8', 'e8', 'f8','g8','h8'],
                    ['a7', 'b7', 'c7', 'd7', 'e7', 'f7','g7','h7'],
                    ['a6', 'b6', 'c6', 'd6', 'e6', 'f6','g6','h6'],
                    ['a5', 'b5', 'c5', 'd5', 'e5', 'f5','g5','h5'],
                    ['a4', 'b4', 'c4', 'd4', 'e4', 'f4','g4','h4'],
                    ['a3', 'b3', 'c3', 'd3', 'e3', 'f3','g3','h3'],
                    ['a2', 'b2', 'c2', 'd2', 'e2', 'f2','g2','h2'],
                    ['a1', 'b1', 'c1', 'd1', 'e1', 'f1','g1','h1']]

    # Find the coordinates of the changed piece
    moved_from = None
    moved_to = None
    
    for row in range(8):
        for col in range(8):
            if prev_b[row][col] != curr_b[row][col]:
                if curr_b[row][col] == '.':
                    # The piece moved from this square
                    moved_from = (row, col)
                else:
                    # The piece moved to this square
                    moved_to = (row, col)
    
    if moved_from is None or moved_to is None:
        return "No valid move found", False  # In case of invalid input

    # Convert coordinates to chess notation
    from_square = board_pos_coor[moved_from[0]][moved_from[1]]
    to_square = board_pos_coor[moved_to[0]][moved_to[1]]

    moved_piece = prev_b[moved_from[0]][moved_from[1]]
    if moved_piece.islower():  # black move
        if prev_b[moved_to[0]][moved_to[1]] != '.': # black capture
            if moved_piece == 'p':
                pgn = f"{board_pos_coor[moved_from[0]][moved_from[1]][0]}x{to_square}"  # black pawn capture
            else:
                pgn = f"{moved_piece.upper()}x{to_square}" #black non pawn capture
        else: # not capture
            if moved_piece == 'p':
                moved_piece = ""
            else:
                moved_piece = moved_piece.upper()
            pgn = f"{moved_piece}{to_square}"  # Regular pawn move
    else:  # white
        if prev_b[moved_to[0]][moved_to[1]] != '.': # capture
            if moved_piece == 'P':
                pgn = f"{board_pos_coor[moved_from[0]][moved_from[1]][0]}x{to_square}"  # capture
            else:
                pgn = f"{moved_piece}x{to_square}"
        else: # not capture
            if moved_piece == 'P':
                pgn = f"{to_square}"  # Regular piece move
            else:
                pgn = f"{moved_piece}{to_square}"  # Regular pawn move
   
    is_white = prev_b[moved_from[0]][moved_from[1]].isupper()

    return pgn, is_white



def gen_pgn(vid_path, model):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    # mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(vid_path)
    
    # Parameters for frame processing
    frame_count = 0
    frame_interval = int(float(cap.get(cv2.CAP_PROP_FPS)) * 0.5)  # Process frame every 1 second
    # previous_board = None  # Track the previous board state
    # previous_hand_present = False  # Track if a hand was detected in the previous frame
    
    # list of board list; appends the board_lists
    board_list_list = []

    # board list iterate every time hand is present
    board_list = []
    
    # get one frame
    print('getting k from gen_pgn')
    k = get_k(cap.read()[1])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if len(board_list) != 0:
                board_list_list.append(board_list)
                board_list = []
            break  # End of video

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform hand detection
        hand_results = hands.process(rgb_frame)
        not_hand_present = hand_results.multi_hand_landmarks == None

        # Crop the frame
        frame = frame[425:1495, :]
        
        for i in range(k):
          frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Process frame every frame_interval
        if frame_count % frame_interval == 0:
            if not_hand_present:
                results = model(frame, conf=0.3)
                chessboard_corners = get_chessboard_corners(frame, k)

                formatted_board = map_yolo_results_to_chessboard(results, chessboard_corners)

                fen = convert_to_valid_fen(formatted_board)
                fen = rotate_fen(fen)

                current_board = chess.Board(fen)
                #print('board')
                board_list.append(str(current_board)[1:-1])
            elif not not_hand_present or not ret:
                #print('hand')
                if len(board_list) != 0:
                    board_list_list.append(board_list)
                board_list = []
        
        # Increment frame counter
        frame_count += 1

    #san_notation = convert_to_san(['..'] + san_move)
   # print(san_notation)
    
    # cap.release()
    # cv2.destroyAllWindows()
    # print(board_list_list)
    # print(len(board_list_list))
    # print(board_list_list[0])
    # print(board_list_list[1])
    # print(board_list_list[2])

    for i,b_list in enumerate(board_list_list):
        print(b_list)
        new_board_list = board_list_to_list(b_list)
        board_list_list[i] = new_board_list
        print(str(new_board_list)+"\n\n")

    pgn_index = 1
    white_move = ".."
    black_move = ".."

    pgn_all = [""]

    for i in range(len(board_list_list) - 1):
        
        p_board = board_list_list[i]
        c_board = board_list_list[i + 1]
        pgn, is_white = board_to_pgn(prev_b=p_board, curr_b=c_board)

        if is_white:
            white_move = pgn
        else:
            black_move = pgn
        
        pgn_row = f"{pgn_index}. {white_move} {black_move} "

        if is_white or (not is_white and black_move != ".."):
            if len(pgn_all) < pgn_index:
                pgn_all.append(pgn_row)
            else:
                pgn_all[pgn_index - 1] = pgn_row

        if black_move != "..":
            white_move = ".."
            black_move = ".."
            pgn_index += 1


    out_str = ""

    for r in pgn_all:
        out_str += r
        
    if out_str == "":
        return '1. '
        
    return out_str

# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# 1. .. move 2. move ..

# Load the YOLO model
model = YOLO("best2.pt")  # Replace with the path to your trained YOLO model
# image_path = "detection/test2.jpg"

# image = cv2.imread(image_path)
video_path_2m = "kaggle/input/2_move_student.mp4"  # Replace with the path to your video
video_path_4m = "kaggle/input/4_Move_studet.mp4"  # Replace with the path to your video
video_path_6m = "kaggle/input/6_Move_student.mp4"  # Replace with the path to your video
video_path_8m = "kaggle/input/8_Move_student.mp4"  # Replace with the path to your video
video_path_2mr = "kaggle/input/2_Move_rotate_student.mp4"



video_path_list = [video_path_2mr, video_path_2m, video_path_4m, video_path_6m, video_path_8m]

output_path = "output/output_video.avi"  # Optional: Specify a path to save the output video

# for vidp in vidp_list:
# for vidp in video_path_list:
#     gen_pgn(vidp, model)
# print(gen_pgn(video_path_4m, model))

# Release resources

#R . B . Q . . R
#P P K . . P . P
#. . . . . N . .
#q . . P . . P .
#. . . p P . . n
#. . . . p . . .
#. p . . b p p p
#R n b k . . . .

#R . B . Q . . R
#P . K . . P . P
#. P . . . N . .
#q . . P . . P .
#. . . p P . . n
#. . . . p . . .
#. p . . b p p p
#R n b k . . . .

pgn = []
for path in video_path_list:
    pgn.append(gen_pgn(path, model))

print(pgn)

import pandas as pd

vids = ['2_Move_rotate_student.mp4','2_move_student.mp4','4_Move_studet.mp4','6_Move_student.mp4','8_Move_student.mp4', '(Bonus)Long_video_student.mp4']

print(len(pgn), len(vids))

if len(pgn) < len(vids):
    pgn.append('1.')
    

# pgn.append('1.')
df = pd.DataFrame({
    "row_id": vids,
    "output": pgn
})

# Save to CSV
df.to_csv("kaggle/working/submission.csv", index=False, encoding="utf-8")
