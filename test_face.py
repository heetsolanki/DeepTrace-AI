from backend.services.detector import extract_face

# Change this to the path of a real photo with a visible face
image_path = "test_face.jpg"

print(f"Looking for a face in: {image_path}")
face = extract_face(image_path)

if face:
    face.save("extracted_face.jpg")
    print("Success! Face detected and saved as 'extracted_face.jpg'")
else:
    print("No face detected. Make sure the image has a clearly visible face.")