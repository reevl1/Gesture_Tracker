import cv2
import mediapipe as mp
import os
import numpy as np

class SimpleImageSlider:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = self.load_images()
        self.current_index = 0

    def load_images(self):
        images = []
        if os.path.exists(self.image_folder):
            for file in os.listdir(self.image_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.image_folder, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
        return images

    def show_current_image(self):
        if not self.images:
            # Create black screen if no images are found
            img = np.zeros((400, 600, 3))
            cv2.putText(img, 'No images found in folder', (50, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return img

        img = self.images[self.current_index]
        # Scale for viewing
        h, w = img.shape[:2]
        scale = min(800 / w, 600 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h))

        # add image num info
        cv2.putText(resized_img, f'Image {self.current_index + 1}/{len(self.images)}',(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return resized_img

    def next_image(self):
        if self.images:
            self.current_index = (self.current_index + 1) % len(self.images)
            return True
        return False
    
    def prev_image(self):
        if self.images:
            self.current_index = (self.current_index - 1) % len(self.images)
            return True
        return False



def main():
    # create slider
    slider = SimpleImageSlider('images')

    # init MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    # tracking movement
    prev_hand_x = None
    right_thresh = 0.10   # tracking sensitivity(right)
    left_thresh  = -0.10  # tracking sensitivity(left)
    cooldown = 0  # cd between stitches

    while True:
        _, frame = cap.read()

        # dec cooldown
        if cooldown > 0:
            cooldown -= 1

        # show current image
        current_image = slider.show_current_image()
        cv2.imshow('Image Gallery', current_image)

        # Process frame for hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_hand_x = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # coords of index finger tip
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                current_hand_x = index_finger_tip.x

                # hand skel for view
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # middle hand point
                middle_finger_start_px = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                cv2.circle(frame, middle_finger_start_px, 10, (0, 255, 0), -1)

        # detect hand movement
        if prev_hand_x is not None and current_hand_x is not None and cooldown == 0:
            delta = current_hand_x - prev_hand_x  # >0 - right, <0 - left

            if delta >= right_thresh:
                print("Left movement detected! Previous image")
                slider.prev_image()
                cooldown = 35  

            elif delta <= left_thresh:
                print("Right movement detected! Next image")
                slider.next_image()
                cooldown = 35

        prev_hand_x = current_hand_x # update 

        cv2.putText(frame, 'Move hand RIGHT / LEFT to change image', (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Cooldown: {cooldown}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #memory clear
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()