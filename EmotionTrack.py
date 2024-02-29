import cv2
import numpy as np
import clip
import torch
from PIL import Image, ImageTk
import tkinter as tk
from mss import mss
import torch.nn.functional as F
from facenet_pytorch import MTCNN


global input_labels_X, snapshot, snapshot_mode, second_monitor_coordinates


input_labels_X = "Happy Face, Sad Face, Angry Face, Fear Face, Disgust Face, Contempt Face, Nervous Face, Curious Face, Flirtatious Face, Ashamed Face, Bored Face, Confused Face, Calm Face, Proud Face, Guilty Face, Annoyed Face, Desperate Face, Jealous Face, Embarrassed Face, Impatient Face, Uncomfortable Face, Bitter Face, Helpless Face, Shy Face, Infatuated Face, Betrayed Face, Shocked Face, Relaxed Face, Apathetic Face, Neutral Face"


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


button_font = ('Helvetica', 25)  # Example font family and size
button_padx = 10  # Horizontal padding
button_pady = 5  # Vertical padding

select_area_button = tk.Button(controls_frame, text="Select Capture Area", command=select_capture_area, font=button_font, padx=button_padx, pady=button_pady)
select_area_button.pack(side=tk.LEFT, padx=5)

# Model selection setup
model_label = tk.Label(controls_frame, text="Model:", font=button_font)
model_label.pack(side=tk.LEFT, padx=(20, 5))


# Model selection drop-down menu
def load_selected_model(selected_model):
    global model, preprocess
    model, preprocess = clip.load(selected_model, device=device)


model_var = tk.StringVar(value="ViT-L/14")  # Default model
model_options = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']  # Example model options
model_menu = tk.OptionMenu(controls_frame, model_var, *model_options, command=load_selected_model)

model_menu.config(font=button_font)  # Increase font size for the dropdown
model_menu.pack(side=tk.LEFT, padx=5)
model_menu["menu"].config(font=button_font)


# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

def update_frame():
    global capture_area
    # Use capture_area for capturing the screen
    with mss() as sct:
        monitor = {"top": capture_area['y'], "left": capture_area['x'], "width": capture_area['width'], "height": capture_area['height']}
        sct_img = sct.grab(monitor)
        cv2_frame = np.array(sct_img)
        cv2_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGBA2RGB)

    # Convert the frame to PIL Image for MTCNN
    frame_pil = Image.fromarray(cv2_frame)

    # Detect faces
    boxes, _ = mtcnn.detect(frame_pil)

    # Find the largest face detected
    largest_area = 0
    largest_face = None
    if boxes is not None:
        for box in boxes:
            x, y, w, h = box[0], box[1], box[2], box[3]
            area = (w - x) * (h - y)
            if area > largest_area:
                largest_area = area
                largest_face = box

    # Crop and overlay the largest face
    if largest_face is not None:
        x, y, w, h = largest_face
        cv2.rectangle(cv2_frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)
        # Crop the largest face from the frame
        cropped_face = cv2_frame[int(y):int(h), int(x):int(w)]

        # Resize the cropped face to a smaller size for display
        small_face = cv2.resize(cropped_face, (400, 500))
        small_face_gray = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
        small_face_bgr = cv2.cvtColor(small_face_gray, cv2.COLOR_GRAY2BGR)

        # Calculate position for the small face in the lower right corner of the main frame
        face_pos_x = cv2_frame.shape[1] - small_face_bgr.shape[1] - 10  # 10 pixels from the right edge
        face_pos_y = cv2_frame.shape[0] - small_face_bgr.shape[0] - 10  # 10 pixels from the bottom edge

        # Overlay the small grayscale face onto the main frame
        cv2_frame[face_pos_y:face_pos_y + small_face_bgr.shape[0],
        face_pos_x:face_pos_x + small_face_bgr.shape[1]] = small_face_bgr

        # Assuming cropped_face is the largest face cropped from the frame
        # Convert the cropped face's color from BGR to RGB and then to grayscale
        frame_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Resize the grayscale image to your desired size for the model
        # Assuming the model expects a 224x224 input
        frame_gray_resized = cv2.resize(frame_gray, (224, 224))

        # Convert the resized grayscale image to a tensor
        frame_tensor = preprocess(Image.fromarray(frame_gray_resized)).unsqueeze(0).to(device)

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
        overlay_start_y = 80
        font_scale = 2.5
        thickness = 2
        text_spacing = 100

        for idx, (label, logit) in enumerate(top_five_labels_logits):
            # Format the label and its probability
            text = f"{label.strip()}: {logit * 100:.1f}%"
            # Calculate the position of the text to be drawn, adjust as necessary
            text_position = (overlay_start_x, overlay_start_y + idx * text_spacing)
            # Draw the text directly onto the frame
            cv2.putText(cv2_frame, text, text_position, cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), thickness)

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

