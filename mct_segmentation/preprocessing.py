from pathlib import Path
import zipfile

import wget
import nibabel as nib
from fastai.vision.all import *


def download_and_unpack_data(rawpath):
    filename = wget.download('https://ndownloader.figstatic.com/files/12981293', out=str(rawpath/'data.zip'))

    with zipfile.ZipFile(dfilename, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    if os.path.exists(filename):
        os.remove(filename)


class Mouse:
    """Some scans have 2 organ segmentations 'Organ1' and 'Organ2'"""
    def __init__(self, mouse_path):
        self.path = mouse_path
        self.find_files()

    def find_files(self):
        lbl_name = self.path/'Organ.img'
        if lbl_name.exists():
            clsfile = self.path/'Organ.cls'
        else:
            lbl_name = self.path/'Organ1.img'
            clsfile = self.path/'Organ1.cls'
        self.lbl = lbl_name
        self.cls = clsfile
        self.img = self.path/'CT280.img'
        return self
        
    def exists(self):
        return np.asarray([self.lbl.exists(), self.cls.exists(), self.img.exists()]).all()
    
    def classes(self):
        filename = open(self.cls)
        d = dict()
        for f in filename:
            k, v = f.split('=')
            d[k] = v

        cleanind = d['ClassIndices'][:-1]
        cleannames = d['ClassNames'][:-1]
        return {n: int(i) for i, n in zip(cleanind.split('|'), cleannames.split('|'))}
    
    def load_img(self):
        return nib.load(self.img).get_fdata().transpose((1,0,2))
        
    def load_lbl(self):
        return nib.load(self.lbl).get_fdata().transpose((1,0,2))
    
    def assign_class_new_value(self, img, old_value, new_value):
        return np.asarray(img==old_value)*new_value
    
    def convert_lbl(self, new_class_dict):
        """While all of the files are valid, the lablels have different values so we'll have to save out a new mask set with consistent labeling and as a directory of pngs for easier DataLoading"""
#         img = self.load_img()
        lbl = np.copy(self.load_lbl())
        old_class_dict = self.classes()
        new_lbl = np.stack([self.assign_class_new_value(lbl, old_class_dict[class_name], new_class_dict[class_name]) for class_name in old_class_dict]).sum(axis=0)
        max_lbl = len(new_class_dict)-1
        new_lbl[new_lbl>max_lbl] = 0
        return new_lbl
    
    
def make_paths():
    mask_path = Path('labels')
    image_path = Path('images')

    if not mask_path.exists():
        mask_path.mkdir()

    if not image_path.exists():
        image_path.mkdir()
        
    return mask_path, image_path


def save_images_and_labels(mouse, class_dict):
    mask_path, image_path = make_paths()
    
    imgs = mouse.load_img()
    lbls = mouse.convert_lbl(class_dict)
    
    img_names = []
    msk_names = []
    for i in range(imgs.shape[2]):
        image_name = image_path/f'{mouse.path.stem}_{i}.png'
        mask_name = mask_path/f'{mouse.path.stem}_{i}_P.png'
        img_names += [str(image_name)]
        msk_names += [str(mask_name)]

        Image.fromarray(((imgs[:,:,i]+1000)/10000*255).astype(np.uint8)).save(image_name) # see if fastai has better solution for this 
        Image.fromarray(lbls[:,:,i].astype(np.uint8)).save(mask_name)
        
    return img_names, msk_names


def write_classes_to_txt(class_dict):
    fname = 'codes.txt'
    file = open(fname, 'w')
    for k in class_dict:
        file.write(f'{k} \n')
    file.close()
    return fname


def make_images_and_labels(path):
    mice = (path/'1_nativeCTdata').ls()
    mice_list = [Mouse(m) for m in mice]
    class_dict = mice_list[0].classes()
    
    img_list = []
    msk_list = []
    for mouse in mice_list:
        img_names, mask_names = save_images_and_labels(mouse, class_dict)
        img_list += img_names
        msk_list += mask_names
    
    codes_file = write_classes_to_txt(class_dict)
    df = pd.DataFrame(data={'Images': img_list, 'Masks': msk_list})
    data_file = 'data.csv'
    df.to_csv(data_file)
    
    return Path(data_file), Path(codes_file)