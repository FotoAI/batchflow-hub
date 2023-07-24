
from typing import Dict, Tuple, Union
from batchflow.processors.core import ModelProcessor
from typing import List, Any
import numpy as np
from batchflow.decorators import log_time
from batchflow.storage import get_storage
import os
from batchflow.constants import BATCHFLOW_HOME
import torch
from torchvision import transforms
from PIL import Image
from argparse import Namespace
import random
import torch
from torchvision import transforms
import timm
import urllib.request




#def get_path(model_name):
#    return '../../models/affectnet_emotions/'+model_name+'.pt'

def get_model_path(model_name):
    model_file=model_name+'.pt'
    cache_dir = os.path.join(os.path.expanduser('~'), '.hsemotion')
    os.makedirs(cache_dir, exist_ok=True)
    fpath=os.path.join(cache_dir,model_file)
    if not os.path.isfile(fpath):
        url='https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/'+model_file+'?raw=true'
        print('Downloading',model_name,'from',url)
        urllib.request.urlretrieve(url, fpath)
    return fpath        
    

class HSEmotion(ModelProcessor):
    #supported values of model_name: enet_b0_8_best_vgaf, enet_b0_8_best_afew, enet_b2_8, enet_b0_8_va_mtl, enet_b2_7
    def __init__(self, *args, device=None,  model_source: Dict[str, str] = None,
        model_path=None, **kwargs):

        super().__init__(
            model_path=model_path,
            model_source=model_source,
            *args,
            **kwargs,
        )

        model_name = os.path.splitext(os.path.basename(self.model_path))[0]

        if not device:
            self.device= "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.is_mtl='_mtl' in model_name
        if '_7' in model_name:
            self.idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
        else:
            self.idx_to_class={0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}

        self.img_size=224 if '_b0_' in model_name else 260
        
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size,self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            ]
        )
        
    def open(self):
        if self.device == 'cpu':
            model=torch.load(self.model_path, map_location=torch.device('cpu'))
        else:
            model=torch.load(self.model_path)
        if isinstance(model.classifier,torch.nn.Sequential):
            self.classifier_weights=model.classifier[0].weight.cpu().data.numpy()
            self.classifier_bias=model.classifier[0].bias.cpu().data.numpy()
        else:
            self.classifier_weights=model.classifier.weight.cpu().data.numpy()
            self.classifier_bias=model.classifier.bias.cpu().data.numpy()
        
        model.classifier=torch.nn.Identity()
        model=model.to(self.device)
        self.model=model.eval()
        self._logger.info(f"model opened: {self.model_path}")

    def close(self):
        self.model = None
    
    def get_probab(self, features):
        x=np.dot(features,np.transpose(self.classifier_weights))+self.classifier_bias
        return x
    
    def extract_features(self,face_img):
        img_tensor = self.test_transforms(Image.fromarray(face_img))
        img_tensor.unsqueeze_(0)
        features = self.model(img_tensor.to(self.device))
        features=features.data.cpu().numpy()
        return features
        
    def predict_emotions(self,face_img, logits=True, return_features=False):
        features=self.extract_features(face_img)
        scores=self.get_probab(features)[0]
        if self.is_mtl:
            x=scores[:-2]
        else:
            x=scores
        pred=np.argmax(x)
        
        if not logits:
            e_x = np.exp(x - np.max(x)[np.newaxis])
            e_x = e_x / e_x.sum()[None]
            if self.is_mtl:
                scores[:-2]=e_x
            else:
                scores=e_x
        
        if return_features:
            return self.idx_to_class[pred],scores, features
        else:
            return self.idx_to_class[pred],scores
        
    def extract_multi_features(self,face_img_list):
        imgs = [self.test_transforms(Image.fromarray(face_img)) for face_img in face_img_list]
        features = self.model(torch.stack(imgs, dim=0).to(self.device))
        features=features.data.cpu().numpy()
        return features
        
    def predict_multi_emotions(self,face_img_list, logits=True, return_features=False):
        features=self.extract_multi_features(face_img_list)
        scores=self.get_probab(features)
        if self.is_mtl:
            preds=np.argmax(scores[:,:-2],axis=1)
        else:
            preds=np.argmax(scores,axis=1)
        if self.is_mtl:
            x=scores[:,:-2]
        else:
            x=scores
        pred=np.argmax(x[0])
        
        if not logits:
            e_x = np.exp(x - np.max(x,axis=1)[:,np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:,None]
            if self.is_mtl:
                scores[:,:-2]=e_x
            else:
                scores=e_x

        if return_features:
            return [self.idx_to_class[pred] for pred in preds],scores, features
        else:
            return [self.idx_to_class[pred] for pred in preds],scores
        

    def process(self, ctx: Dict[str, any]) -> any:
        image = ctx["image"]
        face_bbox = ctx["face_bbox"]
        face_crop = ctx.get("face_crop", None)
        if not face_crop:
            x1,y1,x2,y2 = face_bbox
            face_crop = image[y1:y2, x1:x2]
        emotions, scores, features = self.predict_emotions(face_crop,return_features=True )
        ctx["emotions"] = emotions
        ctx["emotion_scores"] = scores
        ctx["emotion_embedding"] = features
        return ctx
    

    def process_batch(self, batch_ctx: List[Dict[str, any]]) -> Any:
        face_crop_batch = []
        for ctx in batch_ctx:
            image = ctx["image"]
            face_bbox = ctx["face_bbox"]
            face_crop = ctx.get("face_crop", None)
            
            if not face_crop:
                x1,y1,x2,y2 = face_bbox
                face_crop = image[y1:y2, x1:x2]
            face_crop_batch.append(face_crop)
        emotions_b, scores_b, features_b = self.predict_multi_emotions(face_crop_batch,return_features=True )

        if features_b.ndim == 1:
                features_b = np.expand_dims(features_b, 0)

        features_b = features_b / np.expand_dims(
            np.sqrt(np.sum(np.multiply(features_b, features_b), axis=1)), axis=1
        )
       


        for emotions, scores, features,ctx in zip(emotions_b, scores_b, features_b , batch_ctx):
            ctx["emotions"] = emotions
            ctx["emotion_scores"] = scores
            ctx["emotion_embedding"] = np.expand_dims(features,axis=0)

        return batch_ctx

