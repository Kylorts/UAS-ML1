import os
import shutil 
from imagededup.methods import CNN


base_path = 'C:\\Users\\User\\Documents\\Kuliah\\Semester 6\\ML 1\\UAS ML\\dataset'
categories = ['Kertas', 'Kardus', 'Plastik'] # Nama folder kategori kamu
threshold = 0.9


phasher = CNN()

for cat in categories:
    source_dir = os.path.join(base_path, 'cleaned_dataset', f'Sampah {cat}')
    target_dir = os.path.join(base_path, 'deduplicated_dataset', f'Sampah {cat}')
    base_label = cat.lower()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"\n process categories: {base_label} ")
    

    duplicates = phasher.find_duplicates(image_dir=source_dir, min_similarity_threshold=threshold)

    all_files = sorted(os.listdir(source_dir))
    processed_files = set()
    unique_counter = 1

    for file_name in all_files:
        if file_name in processed_files or not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        dup_list = duplicates.get(file_name, [])
        ext = os.path.splitext(file_name)[1]


        new_name = f"{base_label}_{unique_counter}{ext}"
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(target_dir, new_name))
        
        processed_files.add(file_name)
        for dup_file in dup_list:
            processed_files.add(dup_file)
                
        unique_counter += 1

    print(f"done for {cat}:")
    print(f"total original image : {len(all_files)}")
    print(f"total unique images : {unique_counter - 1}")
    print(f"total duplicate images : {len(processed_files) - (unique_counter - 1)}")