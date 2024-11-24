import cv2
from tracker1 import ObjectCounter  # Importing ObjectCounter from tracker.py

# Open the video file
cap = cv2.VideoCapture('cattlecount.mp4')

# Define region points for counting
region_points = [(569, 5), (569, 499)]

# Initialize the object counter
counter = ObjectCounter(
    region=region_points,  # Pass region points
    model="best.pt",  # Model for object counting
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=2,  # Adjust line width for display
)

count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        # If video ends, reset to the beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count += 1
    if count % 2 != 0:  # Skip odd frames
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Process the frame with the object counter
    frame = counter.count(frame)
    

    # Show the frame
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Small delay to visualize the frame
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
 