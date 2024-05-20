import torch
import torch.nn as nn
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import random
import torch.nn.functional as F
import gdal
import json
from CCUnet3plus.UNet_3Plus import UNet_3Plus_DeepSup_FS2W

#GDAL数据类型映射表
NP2GDAL_CONVERSION = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11
}

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset

def writeTiff(im_data, im_geotrans, im_proj, savepath):
    #获取数据类型
    datatype = im_data.dtype.name
    gdaltype = NP2GDAL_CONVERSION[datatype]
    #获取波段、宽高数
    im_bands, im_height, im_width = im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(savepath, int(im_width), int(im_height), int(im_bands), gdaltype)
    if (dataset!= None):
        dataset.SetGeoTransform(im_geotrans)    #写入仿射变换参数
        dataset.SetProjection(im_proj)          #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i]) #写入数据
    del dataset
    
class Unet_dataset(Dataset.Dataset):
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir          
        self.names_list = []
        self.size = 0
    
        self.transform = transforms.ToTensor()

        if not os.path.isfile(self.csv_dir):
            print(self.csv_dir + ':txt file does not exist!')

        file = open(self.csv_dir)
        
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        image_path = self.names_list[idx].split(',')[0]
        image = readTif(image_path)

        label_path = self.names_list[idx].split(',')[1].strip('\n')
        label = readTif(label_path)
        
        img_w = image.RasterXSize
        img_h = image.RasterYSize
        
        image = image.ReadAsArray(0, 0, img_w, img_h).astype('float32')
        label = label.ReadAsArray(0, 0, img_w, img_h).astype('float32')
        
        image1 = image[0:3]/255
        label1 = label[0:3]/255
        
        sample = {'image': image1, 'label': label1}
        
        sample['image'] = torch.from_numpy(image1).float().cuda()
        sample['label'] = torch.from_numpy(label1).float().cuda()
        
        return sample

def criterion(loss1, loss2):
    loss =  (loss1 + loss2*(loss1/loss2).int()) if ((loss1/loss2).int()) > 1 else (loss1 + loss2)
    return loss

def calcCD(imgdata1, imgdata2):
    img1 = imgdata1.detach()
    img2 = imgdata2.detach()
    hist1R = torch.histc(img1[:,0,:,:], bins=256)
    hist1G = torch.histc(img1[:,1,:,:], bins=256)
    hist1B = torch.histc(img1[:,2,:,:], bins=256)
    hist2R = torch.histc(img2[:,0,:,:], bins=256)
    hist2G = torch.histc(img2[:,1,:,:], bins=256)
    hist2B = torch.histc(img2[:,2,:,:], bins=256)
    num = sum(hist1R)
    CDR = 0.0
    CDG = 0.0
    CDB = 0.0
    for i in range(256):
        CDR = CDR + hist1R[i]/num * abs(hist1R[i]-hist2R[i])/256
        CDG = CDG + hist1G[i]/num * abs(hist1G[i]-hist2G[i])/256
        CDB = CDB + hist1B[i]/num * abs(hist1B[i]-hist2B[i])/256
    return (CDR + CDG + CDB)/(3*16384)

def train():     
    #以下为训练部分
    #########################################################################
    configs = json.loads(open('./config.json', 'r').read())
    inputfolder = configs["trainfilefolder"] + '/input/'
    labelfolder = configs["trainfilefolder"] + '/label/'
    epochnum = int(configs["epochnum"])
    modelname = configs["tmodelname"]
    inputfilenum = 0
    labelfilenum = 0
    for file in os.listdir(inputfolder):
        if file.endswith(".tif"):
            inputfilenum +=1
    for file in os.listdir(labelfolder):
        if file.endswith(".tif"):
            labelfilenum +=1
    if inputfilenum != labelfilenum:
        print("The number of input images and label images does not correspond.")
        return
    num_list = list(range(0,inputfilenum))
    train_rate = 0.7
    train_list = random.sample(num_list, int(len(num_list)*train_rate))
    csvfilename = configs["trainfilefolder"] + '/trainre128.csv'
    with open(csvfilename,'w') as train_csv:
        for item in train_list:
            train_csv.write(inputfolder + str(item) + '.tif,' + labelfolder + str(item) + '.tif\n')   
    
    data = Unet_dataset(csvfilename)               
    Unet_dataloader = DataLoader(data, batch_size=10, shuffle=True)
    unet=UNet_3Plus_DeepSup_FS2W(3,3).cuda()
    print(unet)
    
    Loss_function1 = nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    
    for epoch in range(epochnum):
        loss_sum = 0
        loss_sum1 = 0
        loss_sum2 = 0
        loss_sum3 = 0
        for i, sample in enumerate(Unet_dataloader, 0):
            optimizer.zero_grad()

            images, labels = sample['image'], sample['label']

            outputi, outputo = unet(images)            

            loss1 = Loss_function1(outputi, labels)
            loss2 = Loss_function1(outputo, images)              
            loss3 = calcCD(outputi, labels)
            loss = criterion(loss2, criterion(loss1, loss3))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            loss_sum1 +=loss1.item()
            loss_sum2 +=loss2.item()            
            loss_sum3 +=loss3.item()
        scheduler.step()
        lossprint = 'epoch is %d , loss is %f, loss1 is %f, loss2 is %f, loss3 is %f'%(epoch, loss_sum, loss_sum1, loss_sum2, loss_sum3)
        print(lossprint)
        torch.save(unet, modelname)
    torch.save(unet, modelname)
    print("Done!")
    
def eval():
    #以下为验证部分
    #########################################################################
    configs = json.loads(open('./config.json', 'r').read())
    modelname = configs["pmodelname"]
    inputfolder = configs["prefilefolder"] + '/predict/'
    inputfilenum = 0
    for file in os.listdir(inputfolder):
        if file.endswith(".tif"):
            inputfilenum +=1
    if inputfilenum == 0:
        print("The number of input images is 0.")
        return
    
    csvfilename = configs["prefilefolder"] + '/preallre128.csv'
    with open(csvfilename,'w') as pre_csv:
        for item in range(0, inputfilenum):
            pre_csv.write(inputfolder + str(item) + '.tif,' + inputfolder + str(item) + '.tif\n')    
    
    datatest = Unet_dataset(csvfilename)
    Unettest_dataloader = DataLoader(datatest, batch_size=1, shuffle=False)
    unet = torch.load(modelname)
    unet.eval()
    resultdirs = configs["prefilefolder"] + '/result'

    for i, sample in enumerate(Unettest_dataloader, 0):

        imgs, labels = sample['image'], sample['label']
        
        outputs, _ , = unet(imgs)

        predects = (outputs * 255).round().int()
        
        predects = predects.cpu().detach().numpy()
        predects = predects.reshape((-1,128,128))

        file = open(csvfilename)
        names_list = []
        for f in file:
            names_list.append(f)
        image_path = names_list[i].split(',')[0]
        image = readTif(image_path)
        proj = image.GetProjection()
        geotrans = image.GetGeoTransform()
        writeTiff(predects[:,:,:], geotrans, proj, configs["prefilefolder"] + '/result/pre' + str(i) + '.tif')
    print("Done!")


if __name__ == '__main__':
    train()
    eval()