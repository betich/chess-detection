from ultralytics import YOLO
from fake_chessboard import gen_pgn

def main():
  # # Initialize Mediapipe Hands
  # mp_hands = mp.solutions.hands
  # hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
  # mp_drawing = mp.solutions.drawing_utils

  # 1. .. move 2. move ..

  # Load the YOLO model
  model = YOLO("kaggle/input/models/best2.pt")  # Replace with the path to your trained YOLO model
  # image_path = "detection/test2.jpg"

  # image = cv2.imread(image_path)
  video_path_2m = "kaggle/input/videos/2_move_student.mp4"  # Replace with the path to your video
  video_path_4m = "kaggle/input/videos/4_Move_studet.mp4"  # Replace with the path to your video
  video_path_6m = "kaggle/input/videos/6_Move_student.mp4"  # Replace with the path to your video
  video_path_8m = "kaggle/input/videos/8_Move_student.mp4"  # Replace with the path to your video
  video_path_2mr = "kaggle/input/videos/2_Move_rotate_student.mp4"



  video_path_list = [video_path_2mr, video_path_2m, video_path_4m, video_path_6m, video_path_8m]

  output_path = "kaggle/output/output_video.avi"  # Optional: Specify a path to save the output video

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

if __name__ == "__main__":
  main()