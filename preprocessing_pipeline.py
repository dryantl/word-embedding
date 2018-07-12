import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import string
import tqdm

class preprocessing:
	
	def __init__(self,dimension,embedder):
		self.dimension=dimension
		self.embedder=embedder
	
    # fungsi untuk menghapus karakter tidak penting
	def remove_parentheses(self,input_string):
		##input_string=''.join(i for i in input_string if not i.isdigit())
		result_string=input_string.lower()
		target_parentheses=string.punctuation
		for parentheses in target_parentheses:
			result_string=result_string.replace(parentheses, ' ')
		result_string=result_string.strip(' ').split()
		return result_string

    # fungsi untuk mengubah kata menjadi vektor
	def vectorize_word(self,product_title):
		try:
			result=self.embedder[product_title]
		except KeyError:
			result=0
		return result

    # fungsi untuk mengubah kalimat menjadi vektor
	def vectorize_sentence(self,input_sentence):
		result_vector=np.zeros(self.dimension)
		for word in input_sentence:
			result_vector+=self.vectorize_word(word)
		return result_vector

	def preprocess_data(self,features,labels,encoder=None):
		
		embedded_data=pd.DataFrame()

		if(encoder==None):
			label_encoder=LabelEncoder()
			embedded_data["Labels"]=label_encoder.fit_transform(labels)
		else:
			label_encoder=encoder
			embedded_data["Labels"]=label_encoder.transform(labels)
		
		print("REMOVING PUNCTUATIONS")
		embedded_data["Features"]=[self.remove_parentheses(title) for title in tqdm.tqdm(features)]
		print("CONVERTING SENTENCE TO VECTOR")
		embedded_data["Features Vector"]=[self.vectorize_sentence(title) for title in tqdm.tqdm(embedded_data["Features"])]
    
		print("SAVE VECTOR TO PANDAS DATAFRAME")
		for i in tqdm.tqdm(range(self.dimension)):
			embedded_data[i]=[value[i] for value in embedded_data["Features Vector"]]
    
		embedded_data = embedded_data[[*range(self.dimension),"Labels"]]
		
		if(encoder==None):
			return embedded_data, label_encoder
		else:
			return embedded_data
