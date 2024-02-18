import cv2
import numpy as np
import clip
import torch
from PIL import Image, ImageTk
import tkinter as tk
from mss import mss
import torch.nn.functional as F


global input_labels_X, snapshot, snapshot_mode, second_monitor_coordinates


input_labels_X = "Happy Face, Sad Face, Angry Face, Fear Face, Nervous Face, Disgust Face, Contempt Face, Curious Face, Flirtatious Face, Ashamed Face, Bored Face, Confused Face, Proud Face, Guilty Face, Shy Face, Sympathetic Face, Infatuated Face, Neutral Face"


device = "cuda"
model, preprocess = clip.load("ViT-L/14", device=device)


root = tk.Tk()
root.title("EmotionTrack")
root.attributes('-topmost', True)
root.resizable(False, False)

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(fill=tk.BOTH, expand=True)

label0 = tk.Label(frame)
label0.pack()

controls_frame = tk.Frame(root, bg="#f0f0f0")
controls_frame.pack(fill=tk.X, expand=True)



# New global variables for capture area selection
global capture_area
capture_area = {'x': 0, 'y': 0, 'width': 640, 'height': 360}

# Function to start the capture area selection process
def select_capture_area():
    global capture_area

    def on_mouse_click(event):
        global start_x, start_y
        start_x, start_y = event.x, event.y
        canvas.create_rectangle(start_x, start_y, start_x + 1, start_y + 1, outline='red', tag='area', width=2)

    def on_mouse_drag(event):
        # Temporary rectangle to visualize while dragging
        end_x, end_y = event.x, event.y
        # Calculate aspect ratio constrained dimensions
        width = end_x - start_x
        height = end_y - start_y
        aspect_ratio = 16 / 9
        # Adjust width or height based on the aspect ratio
        if width / height > aspect_ratio:
            # Width is too wide for the height; adjust width
            width = height * aspect_ratio
        else:
            # Height is too tall for the width; adjust height
            height = width / aspect_ratio
        # Update rectangle coordinates
        canvas.coords('area', start_x, start_y, start_x + width, start_y + height)

    def on_mouse_release(event):
        global capture_area
        # Finalize the rectangle size with aspect ratio constraint
        width = event.x - start_x
        height = event.y - start_y
        aspect_ratio = 16 / 9
        # Adjust width or height based on the aspect ratio
        if width / height > aspect_ratio:
            # Width is too wide for the height; adjust width
            width = height * aspect_ratio
        else:
            # Height is too tall for the width; adjust height
            height = width / aspect_ratio
        capture_area['x'] = start_x
        capture_area['y'] = start_y
        capture_area['width'] = int(width)
        capture_area['height'] = int(height)
        selection_window.destroy()

    selection_window = tk.Tk()
    selection_window.attributes('-fullscreen', True)
    selection_window.attributes('-alpha', 0.3)  # Adjust if transparency does not work
    selection_window.attributes('-topmost', True)  # Ensure window is on top

    canvas = tk.Canvas(selection_window, cursor="cross")
    canvas.pack(fill=tk.BOTH, expand=True)

    canvas.bind('<Button-1>', on_mouse_click)
    canvas.bind('<B1-Motion>', on_mouse_drag)
    canvas.bind('<ButtonRelease-1>', on_mouse_release)

    selection_window.mainloop()


# Global variable for capture area
# Button to select capture area
select_area_button = tk.Button(controls_frame, text="Select Capture Area", command=select_capture_area)
select_area_button.pack(side=tk.LEFT, padx=5)



# Model selection setup
model_label = tk.Label(controls_frame, text="Model:")
model_label.pack(side=tk.LEFT, padx=(20, 5))


# Model selection drop-down menu
def load_selected_model(selected_model):
    global model, preprocess
    model, preprocess = clip.load(selected_model, device=device)


model_var = tk.StringVar(value="ViT-L/14")  # Default model
model_options = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']  # Example model options
model_menu = tk.OptionMenu(controls_frame, model_var, *model_options, command=load_selected_model)
model_menu.pack(side=tk.LEFT, padx=5)



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def update_frame():
    global current_label, snapshot_mode, capture_area, frame_count, start_time


    # Use capture_area for capturing the screen
    with mss() as sct:
        monitor = {"top": capture_area['y'], "left": capture_area['x'], "width": capture_area['width'], "height": capture_area['height']}
        sct_img = sct.grab(monitor)
        cv2_frame = np.array(sct_img)
        cv2_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGBA2RGB)

    # Detect faces in the frame
    gray = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Find the largest face detected
    largest_area = 0
    largest_face = None
    for (x, y, w, h) in faces:
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_face = (x, y, w, h)


    # Draw rectangles around the faces and select the largest face for cropping
    if largest_face is not None:
        x, y, w, h = largest_face
        cv2.rectangle(cv2_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Crop the largest face from the frame
        cropped_face = cv2_frame[y:y + h, x:x + w]
        # Convert the cropped face's color from BGR to RGB
        frame_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

        # Resize the cropped face to a smaller size for display
        small_face = cv2.resize(cropped_face, (300, 300))

        # Calculate position for the small face in the lower right corner of the main frame
        face_pos_x = cv2_frame.shape[1] - small_face.shape[1] - 10  # 10 pixels from the right edge
        face_pos_y = cv2_frame.shape[0] - small_face.shape[0] - 10  # 10 pixels from the bottom edge

        # Overlay the small face onto the main frame
        # Convert small_face to grayscale
        small_face_gray = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
        # Convert the grayscale image back to BGR before overlaying (to match cv2_frame's color space)
        small_face_bgr = cv2.cvtColor(small_face_gray, cv2.COLOR_GRAY2BGR)
        cv2_frame[face_pos_y:face_pos_y + small_face_bgr.shape[0],
        face_pos_x:face_pos_x + small_face_bgr.shape[1]] = small_face_bgr

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Resize the grayscale image to your desired size
        frame_gray = cv2.resize(frame_gray, (1920, 1080))

        frame_tensor = preprocess(Image.fromarray(frame_gray)).unsqueeze(0).to(device)

        # Tokenize input labels and prepare for model
        input_labels = input_labels_X.split(", ")
        text = clip.tokenize(input_labels).to(device)

        with torch.no_grad():
            # Encode the frame and text
            image_features = model.encode_image(frame_tensor)
            text_features = model.encode_text(text)

            # Calculate logit
            logit_per_image, logit_per_text = model(frame_tensor, text)

            # Apply softmax to convert logits to probabilities
            probabilities = F.softmax(logit_per_image[0], dim=0)

            logits = []
            for idx, single_probability in enumerate(probabilities):
                logits.append(single_probability.item())

        combined_labels_logits = list(zip(input_labels, logits))
        combined_labels_logits.sort(key=lambda x: x[1], reverse=True)
        top_five_labels_logits = combined_labels_logits[:5]

        overlay_start_x = 10
        overlay_start_y = 10
        overlay_width = 400  # Adjusted based on frame size
        overlay_height = 350  # Adjusted to fit the text

        # Create a semi-transparent overlay
        overlay = cv2_frame[overlay_start_y:overlay_start_y + overlay_height,
                  overlay_start_x:overlay_start_x + overlay_width].copy()
        transparent_layer = np.zeros_like(overlay, dtype=np.uint8)
        cv2.rectangle(transparent_layer, (0, 0), (overlay_width, overlay_height), (255, 255, 255), -1)

        alpha = 0.65  # Transparency factor
        cv2.addWeighted(transparent_layer, alpha, overlay, 1 - alpha, 0, overlay)
        cv2_frame[overlay_start_y:overlay_start_y + overlay_height,
        overlay_start_x:overlay_start_x + overlay_width] = overlay

        font_scale = 1.0
        thickness = 2
        text_spacing = 65  # Space between lines

        # Adjust text positioning dynamically
        num_texts = len(top_five_labels_logits)
        total_text_height = num_texts * text_spacing
        start_y_offset = (overlay_height - total_text_height) // 2  # Center texts vertically within the overlay

        # Increase the start_y_offset to move text lower
        start_y_offset += 35  # Increase this value to move the text lower within the overlay

        for idx, (label, logit) in enumerate(top_five_labels_logits):
            text = f"{label.strip()}: {logit * 100:.1f}%"
            text_position = (
                overlay_start_x + 10, overlay_start_y + start_y_offset + idx * text_spacing)  # Adjust text position
            cv2.putText(cv2_frame, text, text_position, cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 0, 0), thickness)



    # Define border width in pixels
    top, bottom, left, right = [15]*4  # Example: 50 pixels border on all sides

    # Define border color in BGR (Black in this example)
    border_color = [192, 192, 192]

    # Add the border to the frame
    cv2_frame_with_border = cv2.copyMakeBorder(cv2_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

    # Original aspect ratio for 1920x1080 is 16:9
    aspect_ratio = 16 / 9

    # Adjust the new desired width if needed or keep it to maintain the aspect ratio
    new_width = 1024  # You can adjust this if you want a different size

    # Decrease the new height to make the display frame smaller
    new_height = int(new_width / aspect_ratio)  # Calculate new height based on the desired width to maintain aspect ratio

    # Resize the frame for GUI display with new dimensions
    display_frame = cv2.resize(cv2_frame_with_border, (new_width, new_height))

    # Update the Tkinter window size
    # Decrease the added height in window_height to make the overall window smaller
    window_width = new_width + 40  # Adjust window width as necessary
    window_height = new_height + 100  # Decrease extra height to make the window shorter

    # Update the root window's geometry
    root.geometry(f"{window_width}x{window_height}")

    # Optionally adjust the minimum size to prevent resizing to a smaller size
    root.minsize(window_width, window_height)

    # Convert the cropped frame for Tkinter display
    frame_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    frame_photo = ImageTk.PhotoImage(image=frame_pil)

    # Update the label with the new PhotoImage
    label0.config(image=frame_photo)
    label0.image = frame_photo

    # Schedule the next update
    root.after(10, update_frame)


update_frame()
root.mainloop()

