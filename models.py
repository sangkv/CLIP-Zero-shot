import torch
import clip 

class ImageClassifier():
    def __init__(self, categories, model='ViT-B/32'):
        self.categories = categories

        # Load the model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.preprocess = clip.load(model, device=self.device)

        # Prepare the inputs
        text_inputs = torch.cat([clip.tokenize(text) for text in categories]).to(self.device)

        # Calculate features
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)

        # Normal
        self.text_features = text_features / (text_features.norm(dim=-1, keepdim=True))
    
    def predict(self, image):
        # Prepare the inputs
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        # Calculate features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        
        # Normal
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Predict
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

        #values, indices = similarity[0].topk(5)
        values, indices = similarity[0].topk(1)

        value = values[0]
        index = indices[0]

        probability = value.item()
        result = self.categories[index.item()]

        return result, probability
