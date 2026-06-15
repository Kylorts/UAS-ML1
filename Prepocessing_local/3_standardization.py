import cv2
import os

base_path = 'C:\\Users\\User\\Documents\\Kuliah\\Semester 6\\ML 1\\UAS ML\\dataset'
input_root = os.path.join(base_path, 'deduplicated_dataset')
output_root = os.path.join(base_path, 'standardized_dataset')

categories = ['Kertas', 'Kardus', 'Plastik']

IMG_SIZE = (224, 224)

for cat in categories:
    class_input = os.path.join(input_root, f'Sampah {cat}')
    class_output = os.path.join(output_root, f'Sampah {cat}')

    if not os.path.exists(class_output):
        os.makedirs(class_output, exist_ok=True)

    if not os.path.exists(class_input):
        print(f"Skip, folder {class_input} not found.")
        continue

    files = [f for f in os.listdir(class_input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n process Sampah {cat} ({len(files)} file)")

    for f in files:
        path = os.path.join(class_input, f)
        img = cv2.imread(path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_resized = cv2.resize(img_rgb, IMG_SIZE)

        name = os.path.splitext(f)[0]
        save_path = os.path.join(class_output, name + ".jpg")

        cv2.imwrite(save_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

print("\nStandardization done")