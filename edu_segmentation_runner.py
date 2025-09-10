#https://pypi.org/project/edu-segmentation
from edu_segmentation.download import download_models
download_models()
from edu_segmentation.main import EDUSegmentation, ModelFactory, BERTUncasedModel, BERTCasedModel, BARTModel
from edu_segmentation.main import DefaultSegmentation, ConjunctionSegmentation
model_type = "bert_uncased"  # or "bert_cased", "bart"
model = ModelFactory.create_model(model_type)
edu_segmenter = EDUSegmentation(model)

text = "Developed by Peking University's Tangent Lab, the toolkit is for segmenting Elementary Discourse Units. "
"It implements an end-to-end neural segmenter based on a neural framework, "
"addressing data insufficiency by transferring a word representation model trained on a large corpus"

granularity = "conjunction_words"  # or "default"
conjunctions = ["and", "but", "however"]  # Customize conjunctions if needed
device = 'cpu'  # Choose your device, e.g., 'cuda:0'

segmented_output = edu_segmenter.run(text, granularity, conjunctions, device)
print(segmented_output)