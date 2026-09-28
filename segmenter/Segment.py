import pydicom
from tkinter import Frame, Button, Label, Tk, BOTH
from tkinter import filedialog, messagebox
import os
import datetime
import pandas as pd
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.segmentation import DiceScore
from pathlib import Path


def makeKey(text):
    parts = text.split("_")
    return (parts[3]+"_"+parts[4]+"_"+parts[-2])


def make_df(dir=r"C:\Segment_Temp"):
    files = os.listdir(dir)
    annotations = [i for i in files if i.lower().endswith(('.png',
                                                           '.jpg',
                                                           '.jpeg',
                                                           ".dcm",
                                                           ".tif",
                                                           ".tiff"))]
    if len(annotations) == 0:
        raise Exception("No Images found in folder")
    inputs = [i for i in files]
    images = pd.DataFrame({"Input": inputs})
    return images


class TestImageDataset(Dataset):
    def __init__(self, dir_, dataframe, size=256,
                 transform=None, target_transform=None):
        self.img_labels = dataframe
        self.num = self.img_labels.shape[0]
        self.img_dir = dir_
        self.transform = transform
        self.target_transform = target_transform
        self.size = size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        if ".dcm" in img_path:
            dcm = pydicom.dcmread(img_path)
            image = dcm.pixel_array
        else:
            image = Image.open(img_path)
        image = np.array(image)
        if image.shape[2] == 3:
            pass
        else:
            new_image = np.zeros([image.shape[0],
                                 image.shape[1], 3],
                                 dtype='uint8')
            new_image[:, :, 0] = image
            new_image[:, :, 1] = image
            new_image[:, :, 2] = image
            image = new_image
        image = Image.fromarray(image)
        image = self._fixit(image)
        image = image.resize((self.size, self.size))
        image = np.array(image)
        if self.transform:
            out = self.transform(image=image, mask=None)
            image = out['image']
        image = torch.from_numpy(image/255).float()
        image = image.permute(2, 0, 1)
        return image, 'a', img_path

    def _fixit(self, image):
        if image.height == 868 and image.width == 768:
            image = image.crop((0, 0, 768, 768))
        elif image.height == 1636 and image.width == 1536:
            image = image.crop((0, 0, 1536, 1536))
        return image

    def _makeMask(self, image):
        image = (
            (image[:, :, 0] > 200) &
            (image[:, :, 1] < 140) &
            (image[:, :, 2] < 140)
        )
        return image


class StrongModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name,
            in_channels=in_channels, classes=out_classes,
            **kwargs
        )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std",
                             torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean",
                             torch.tensor(params["mean"]).view(1, 3, 1, 1))
        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE,
                                           from_logits=True)
        self.dice_score = DiceScore(ignore_index=None)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        pred = self.model(image)
        return pred

    def training_step(self, batch, batch_idx):
        x, y, ZZ = batch
        image = x
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        mask = y
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(),
                                               mask.long(),
                                               mode="binary")
        loss = self.loss_fn(logits_mask, y)
        dice = self.dice_score(logits_mask, y)
        self.log("train_loss", loss)
        self.log("train_Dice", dice)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, ZZ = batch
        image = x
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        mask = y
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(),
                                               mask.long(),
                                               mode="binary")
        loss = self.loss_fn(logits_mask, y)
        dice = self.dice_score(logits_mask, y)
        self.log("val_loss", loss)
        self.log("Val_Dice", dice)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, ZZ = batch
        image = x
        mask = y
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        dice = self.dice_score(logits_mask, y)
        self.log("test_loss", loss)
        self.log("test_Dice", dice)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


def makeBinaryMask(img, thresh):
    thresh = torch.tensor([thresh])
    img = (img > thresh).float()*1
    return img


def look(model, dataloader, save_path):
    scale = ((200/10.6667)**2)/1e6
    targets = []
    preds = []
    dc512 = []
    fnames = []
    batch = {}
    for batch['image'], batch['mask'], fns in dataloader:
        with torch.no_grad():
            model.eval()
            logits = model(batch["image"])
            pr_masks = logits.sigmoid()
        for image, gt_mask, pr_mask, f in zip(batch["image"],
                                              batch["mask"],
                                              pr_masks, fns):
            image_name_txt = f
            image_name_txt = image_name_txt.split("\\")[-1]
            tmp_image = image.numpy().transpose(1, 2, 0).copy()
            # just squeeze classes dim, because we have only one class
            mm = makeBinaryMask(pr_mask, 0.5).squeeze()
            tmp_image = image.numpy().transpose(1, 2, 0).copy()
            mm_image = Image.fromarray(mm.numpy()*255)
            mm_image = mm_image.convert("L")
            print(f"Measure {image_name_txt}")
            tmp_image[:, :, 0] = np.maximum(tmp_image[:, :, 0], mm)
            tmp_name = "Report_"+image_name_txt
            tmp_image = Image.fromarray((tmp_image*255).astype('uint8'))
            preds.append(makeBinaryMask(pr_mask, 0.5).squeeze().sum().numpy())
            targets.append(0)
            fnames.append(image_name_txt)
            report = Image.new(mode="RGB",
                               size=(1044, 1044),
                               color=(255, 255, 255))
            report.paste(tmp_image, (0, 220))
            image = image*255
            image = Image.fromarray(
                image.permute(1, 2, 0).numpy().astype('uint8'))
            report.paste(image, (532, 220))
            I1 = ImageDraw.Draw(report)
            myFont = ImageFont.truetype('arial.ttf', 24)
            myFont2 = ImageFont.truetype('arial.ttf', 36)
            myFont3 = ImageFont.truetype('arial.ttf', 28)
            pixels = makeBinaryMask(pr_mask, 0.5).squeeze().sum().numpy()
            area = pixels*scale
            I1.text((36, 0), "A-EYE Research Unit",
                    fill=(0, 0, 0), font=myFont2)
            I1.text((36, 40), "Dept of Ophthalmology and Visual Sciences",
                    fill=(0, 0, 0), font=myFont2)
            I1.text((290, 80), "Geographic Atrophy Report",
                    fill=(0, 0, 0), font=myFont3)
            current_time = datetime.now()
            formatted_time = current_time.strftime('%m-%d-%Y %H:%M:%S')
            I1.text((36, 145), f"Report Date:{formatted_time}",
                    fill=(0, 0, 0), font=myFont)
            I1.text((36, 120), f"Image: {image_name_txt}",
                    fill=(0, 0, 0), font=myFont)
            I1.text(
                (36, 745),
                "\u2022 Do not use the above images for diagnostic purposes. "
                "Images have been resized to 512x512 ",
                fill=(0, 0, 0),
                font=myFont,
                )
            I1.text(
                (36, 790),
                "\u2022 AI based measurements assume that the input image is",
                "standard 786 x786. If the image \ndoes not have standard",
                "resolution, please convert pixel measurement to \nconvert",
                "to mm2 for known calibration",
                fill=(0, 0, 0),
                font=myFont)
            I1.text(
                (36, 900), "\u2022 This report is AI-generated and intended",
                "to support healthcare professionals. \nIt is not a"
                "substitute for clinical judgment, and medical decisions",
                "should be based on \nprofessional evaluation of the",
                " patient's condition",
                fill=(255, 0, 0),
                font=myFont)
            I1.text(
                (36, 168), f"AI measured Area:{area:0.2f} mm\u00b2 or ",
                "{pixels} total pixels",
                fill=(0, 0, 0),
                font=myFont)
            I1.text((36, 195), "AI Segmented GA", fill=(0, 0, 0), font=myFont)
            I1.text((636, 195), "Original", fill=(0, 0, 0), font=myFont)
            report.save(os.path.join(save_path, tmp_name+".pdf"))
            dc512.append(area)
    return dc512, preds, targets, fnames


class Window(Frame):
    def __init__(self, master=None, app_dir=None, source_dir=None):
        Frame.__init__(self, master)
        self.master = master
        self.img_dir = source_dir
        self.app_dir = app_dir
        # widget can take all window
        self.pack(fill=BOTH, expand=1)
        # create button, link it to clickExitButton()
        exitButton = Button(self, text="Exit", command=self.clickExitButton)
        # place button at (0,0)
        exitButton.place(x=0, y=0)
        text = Label(self, text=self.img_dir)
        text.place(x=240, y=50)
        text2 = Label(self, text=self.app_dir)
        text2.place(x=240, y=110)
        imgPathButton = Button(self, text="Select Img Dir",
                               command=self.setImagePath)
        imgPathButton.place(x=70, y=50)
        locPathButton = Button(self, text="Select App Dir",
                               command=self.setAppPath)
        locPathButton.place(x=70, y=110)
        runButton = Button(self, text="Run", command=self.runModel)
        runButton.place(x=70, y=250)

    def clickExitButton(self):
        exit()

    def setImagePath(self):
        self.img_dir = filedialog.askdirectory()
        text = Label(self, text=self.img_dir)
        text.place(x=240, y=50)

    def setAppPath(self):
        self.app_dir = filedialog.askdirectory()
        text2 = Label(self, text=self.app_dir)
        text2.place(x=240, y=110)

    def runModel(self):
        text3 = Label(self, text="Running")
        text3.place(x=240, y=250)
        messagebox.showinfo(title=None, message="Model will begin running")

        ckpt_path = os.path.join(self.app_dir, "epoch=18-step=1406.ckpt")
        main(ckpt=ckpt_path, encoder="efficientnet-b5",
             arch="FPN", save_dir="results", source=self.img_dir)
        messagebox.showinfo(title=None, message="Finished")
        text3 = Label(self, text="Finished")
        text3.place(x=240, y=250)


def main(ckpt=None, encoder=None,
         arch=None, save_dir=None, source=None):
    runtime = datetime.now().strftime("%m-%d-%YAT%H-%M-%S")
    data = make_df(source)
    save_path = os.path.join(app.app_dir,
                             save_dir, "_".join(["COCOGA",
                                                 arch,
                                                 encoder,
                                                 runtime]))
    result_path = os.path.join(app.app_dir,
                               save_path, "_".join(["COCOGA",
                                                    arch,
                                                    encoder,
                                                    runtime])+".csv")
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path, exist_ok=True)
    test_data = TestImageDataset(source, dataframe=data, size=512)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = StrongModel.load_from_checkpoint(ckpt,
                                             arch=arch,
                                             encoder_name=encoder,
                                             in_channels=3,
                                             out_classes=1,
                                             map_location='cpu')

    d, p, t, f = look(model, test_loader, save_path)
    results = pd.DataFrame({"Image": f, "Prediction": p, "Area": d})
    results.to_csv(result_path)
    print("FINISHED YOU MAY EXIT NOW")
    with open(my_fp, "w") as f:
        f.write(app.img_dir)
        f.write("\n")
        f.write(app.app_dir)
    return 0


my_fp = r"C:\Users\rslater.FPRC.002\Desktop\seg_settings.txt"
my_file = Path(my_fp)
if my_file.is_file():
    with open(my_file, "r") as f:
        items = f.readlines()
        items = [i.strip() for i in items]
        init_app = items[1]
        init_source = items[0]
else:
    init_app = ""
    init_source = ""
root = Tk()
app = Window(root, app_dir=init_app, source_dir=init_source)
root.wm_title("Segmentation App")
root.geometry("640x400")
root.mainloop()
