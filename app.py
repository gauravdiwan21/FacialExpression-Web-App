import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import models
import streamlit as st



# target classes for predictions
data_classes = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

## Model class
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, "last_lr: {:.5f},".format(result['lrs'][-1]) if 'lrs' in result else '', 
            result['train_loss'], result['val_loss'], result['val_acc']))

class FERModel(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=pretrained)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

model = FERModel(7)
device = torch.device('cpu')
PATH = "fer.pth"

model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()


# image -> tensor
def transform_image(image_bytes):

    ## reading the image from the input stream 
    ## ocnverting to PIL image
    ## converting to RGB as requred by the model using PIL convert
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  


    ## applying transformation
    ## Image already converted to RGB
    ## Resizing to the size the originialImages were in Before training
    ## converting to tensaor
    transform = transforms.Compose([
                                    transforms.Resize((48,48)),
                                    # transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()])
    ## returning Image tensor with 4 dimensions -- batch dimension added                                
    return transform(image).unsqueeze(0)


# predict
def get_prediction(image_tensor):
    # images = image_tensor.reshape(-1, 48*48)
    images= image_tensor
    outputs = model(images)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted



def about():
	st.write(
		'''
		
		
		In this project I wanted to build an algorithm that could automatically detect the facial expression within the Image. For that, I used the RESNET34 
		architecture and the Facial Expression Recog Image Ver of (FERC)Dataset [Dataset Link](https://www.kaggle.com/manishshah120/facial-expression-recog-image-ver-of-fercdataset), which consists of approximately 
		37 thousand pictures of mid-upper body selfies, with only one person in each picture.
		
		Please take into account that the model was trained on mid-upper body selfies, with only one person in the picture.
        
		
		**If you want to see how I did it, [here](https://github.com/gauravdiwan21/FER-WebApp/tree/master) is the repo.** 



		''')
    
    

def main():
  st.title("FACIAL EXPRESSION RECOGNITION")

  activities = ["About","App"]
  choice = st.sidebar.selectbox("Pick something:", activities)

  if choice == "About":
      about()
      st.write("Here's an example of the kind of pictures with wich it works best:")

      st.image("images/SelfieImage.jpg")

      st.write("**Model Creation [Link](https://www.kaggle.com/gauravdiwan/facial-expression-pytorch-resnet34)**. Please upvote if you find useful :)")


  elif choice == "App":
	
    st.write("**Please note that it will work best with mid-upper body selfies, ideally with only one person in the picture**")
    st.write("Here's an example of the kind of pictures with wich it works best:")


    st.image("images/SelfieImage.jpg")

    
    image_file = st.file_uploader("Upload an Image", type=['jpeg', 'png', 'jpg', 'webp'])

    if image_file is None:
        st.write("No file or File is not in correct format")
    
    if image_file is not None:

        img_bytes = image_file.read()
        tensor = transform_image(img_bytes)
        
    
    if st.button("Process"):
        prediction = get_prediction(tensor)
        data = {'prediction': prediction.item(), 
                'Emotion': str(data_classes[prediction.item()])}
        st.image(image_file)
        st.write(data)



if __name__ == "__main__":
    main()