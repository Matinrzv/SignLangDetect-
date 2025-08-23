import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # ['A', 'B', ..., 'Z']
dataset_size = 300

cap = cv2.VideoCapture(0)

for label in classes:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {label}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        cv2.putText(frame, f'Show {label} - Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    existing_files = os.listdir(class_dir)
    existing_count = len([f for f in existing_files if f.endswith('.jpg')])

    counter = existing_count
    while counter < existing_count + dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f'Capturing {label}: {counter-existing_count+1}/{dataset_size}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25)
        if key == ord('e'):
            print('Exit requested!')
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
