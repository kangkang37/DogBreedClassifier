
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# dataset
import pandas as pd
from PIL import Image
from torchvision import transforms

MAX_PROB=0.2 # if the max prob smaller than MAX_PROB, it's not a dog

class PretrainViT(nn.Module):

    def __init__(self):
        super(PretrainViT, self).__init__()

        model = models.vit_l_16() ###
        
        num_classifier_feature = model.heads.head.in_features
        
        model.heads.head = nn.Sequential(
            nn.Linear(num_classifier_feature, 120)
        )
        
        self.model = model

        for param in self.model.named_parameters():
            if "heads" not in param[0]:
                param[1].requires_grad = False

    def forward(self, x):
        return self.model(x)


# load the trained model
torch.manual_seed(2022)
try:
    device = torch.device("mps")
except:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


net = PretrainViT()
net.load_state_dict(torch.load("./net.pt", map_location="cpu"))
net.to(device)
net.eval()

channel_mean = torch.Tensor([0.485, 0.456, 0.406])
channel_std = torch.Tensor([0.229, 0.224, 0.225])

csv_path = "./dog-breed-identification/labels.csv"
label_df = pd.read_csv(csv_path)

label_idx2name = label_df['breed'].unique()

label_names_en = label_idx2name.tolist()
for i in range(len(label_names_en)):
    name=label_names_en[i].replace('_',' ')
    label_names_en[i]=name

input_file = 'label_zh.txt'

with open(input_file, 'r') as file:
    label_names_zh= [line.strip() for line in file]



myServer = Flask(__name__, template_folder='templates')
# Put index.html in the templates folder


@myServer.route('/')
def index():
    return render_template('index.html')

@myServer.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    isDog=True

    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_mean, std=channel_std)
    ])
    image = Image.open(file).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = net(image)
        probabilities = torch.softmax(output, dim=1)
        # print(probabilities)
        max_prob, predicted_label = torch.max(probabilities, dim=1)
        # print(predicted_label)
    
    if max_prob<MAX_PROB:
        isDog=False
    
    if isDog:
        predicted_class_en=label_names_en[predicted_label]
        predicted_class_zh=label_names_zh[predicted_label]

    if isDog:
        return jsonify({'predicted_class': str(predicted_class_en+', '+predicted_class_zh)})
    else:
        return jsonify({'predicted_class': "It's not a dog!"})

# run the app
if __name__ == '__main__':
    # myServer.run()
    myServer.run(host='0.0.0.0', port=8000, debug=True)

