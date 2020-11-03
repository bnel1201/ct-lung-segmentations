from pathlib import Path
import nibabel as nib
from fastai.vision.all import *

class Mouse:
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
        return nib.load(self.img).get_fdata()
        
    def load_lbl(self):
        return nib.load(self.lbl).get_fdata()
    
    def assign_class_new_value(self, img, old_value, new_value):
        return np.asarray(img==old_value)*new_value
    
    def convert_lbl(self, new_class_dict):
        """While all of the files are valid, the lablels have different values so we'll have to save out a new mask set with consistent labeling and as a directory of pngs for easier DataLoading"""
        img = self.load_img()
        lbl = np.copy(self.load_lbl())
        old_class_dict = self.classes()
        return np.stack([self.assign_class_new_value(lbl, old_class_dict[class_name], new_class_dict[class_name]) for class_name in old_class_dict]).sum(axis=0)
    
    
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
        
        Image.fromarray(imgs[:,:,i].astype(np.uint16)+1000).save(image_name)
        Image.fromarray(lbls[:,:,i].astype(np.uint16)).save(mask_name)
        
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
    df = pd.DataFrame(list(zip(img_list, msk_list)), columns = ['Images', 'Masks'])
    
    data_file = 'data.csv'
    df.to_csv(data_file)
    
    return Path(data_file), Path(codes_file)