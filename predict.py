
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import numpy as np


input_shape = (224, 224)  

max_length = 35

def remove_first_last_word(sentence):
    words = sentence.split()
    if len(words) >= 3: 
        words = words[1:-1]  
        return ' '.join(words)
    else:
        return ''
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    
    for i in range(max_length):
        
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        
        sequence = pad_sequences([sequence], max_length)
       
        yhat = model.predict([image, sequence], verbose=0)
        
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        
        if word is None:
            break
        in_text += " " + word
        
        if word == 'endseq':
            break

    return in_text


class indoor_class:
    def __init__(self,filename):
        self.filename =filename
    
    
   


    def predictiondogcat(self):

        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        vgg_model = VGG16()

        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
        saved_model_path = "best_model1.h5"
        model = load_model(saved_model_path)
        # image_path = '10815824_2997e03d76.jpg'
        imagename = self.filename
        image = load_img(imagename, target_size=input_shape)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = vgg_model.predict(image, verbose=0)
        res=predict_caption(model, feature, tokenizer, max_length)
        res=remove_first_last_word(res)
        print(res)





        
        
        
        result={"prediction_scores":str(""),
                "predicted_class_label":res.capitalize()
                }

        return result







       


