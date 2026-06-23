import json

nb_path = 'C:/order/Desktop/Gun/Shoot_Catcher/01_1D_CNN/1d_cnn_gunshot_detector.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell.get('source', []):
            if "DATA_DIR = Path" in line and "Shoot_Catcher" in line:
                line = "DATA_DIR = Path(r'C:\\order\\Desktop\\Gun\\Shoot_Catcher\\Data\\READY_1D_CNN')\n"
            new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
print("Notebook patched successfully!")
