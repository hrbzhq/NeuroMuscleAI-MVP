import io
from PIL import Image, ImageDraw
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
ART.mkdir(parents=True, exist_ok=True)

def make_image(color, size=(224,224)):
    im = Image.new('RGB', size, color=color)
    d = ImageDraw.Draw(im)
    d.text((10,10), color, fill=(255,255,255))
    return im

def main():
    imgs = [make_image('red'), make_image('green'), make_image('blue')]
    names = ['red.jpg','green.jpg','blue.jpg']
    zip_path = ART / 'test_images.zip'
    csv_path = ART / 'test_labels.csv'
    with zipfile.ZipFile(zip_path, 'w') as z:
        for n, im in zip(names, imgs):
            buf = io.BytesIO()
            im.save(buf, format='JPEG')
            z.writestr(n, buf.getvalue())
    # create CSV: label 0 for red/normal, 1 for others
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('red.jpg,0\n')
        f.write('green.jpg,1\n')
        f.write('blue.jpg,1\n')
    print('Wrote', zip_path, csv_path)

if __name__ == '__main__':
    main()
