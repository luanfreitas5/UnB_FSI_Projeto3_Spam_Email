# Nome: Luan Mendes Gonçalves Freitas
# Disciplina: Fundamentos de Sistemas Inteligentes
# Projeto Final - Detecção de Spam baseado na suposição de um conteúdo diferente de um e-mail legítimo
# Módulo RedesNeurais

import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class RedesNeurais:
	
	def __init__(self):
		self.head = None
		self.df = None
		self.caracteristicas = None
		self.rotulos = None
		self.contador=0
		
	def leitura(self):
		
		self.head = ['0', 'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
					'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
					'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
					'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
					'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
					'char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam']
		
		self.df = pd.read_table('spambase.data', delim_whitespace=True, header=None, names=self.head)  # lendo todos os atributos do arquivo
	
		for item in self.head:
			if item != '0': 
				self.df[item] = ''
				
		self.df[['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business',
		'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
		'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
		'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
		'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
		'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
		'char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam']] = self.df['0'].str.split(',', expand=True)
		del self.df['0']
		
		del self.df['word_freq_our']
		del self.df['word_freq_over']
		del self.df['word_freq_email']
		del self.df['word_freq_addresses'] 		
		del self.df['word_freq_you']
		del self.df['word_freq_your']
		del self.df['word_freq_000']
		del self.df['word_freq_hp']
		del self.df['word_freq_hpl']
		del self.df['word_freq_george']
		del self.df['word_freq_650']
		del self.df['word_freq_lab']
		del self.df['word_freq_labs']
		del self.df['word_freq_telnet']
		del self.df['word_freq_857']
		del self.df['word_freq_415']
		del self.df['word_freq_85']
		del self.df['word_freq_1999']
		del self.df['word_freq_parts']
		del self.df['word_freq_pm']
		del self.df['word_freq_direct']
		del self.df['word_freq_cs']
		del self.df['word_freq_original']
		del self.df['word_freq_re']
		del self.df['word_freq_table']
		del self.df['char_freq_;']
		del self.df['char_freq_(']
		del self.df['char_freq_[']
		del self.df['capital_run_length_average']
		del self.df['capital_run_length_longest']
		del self.df['capital_run_length_total']
		
		self.rotulos = self.df['spam']
		self.caracteristicas = self.df.drop(['spam'], axis=1)
	
class PerceptronMulticamadas(RedesNeurais):
	
	def __init__(self):
		super().__init__()
		
	def relatorioClassificaçãoPontuaçãoPrecisao(self, y_true, y_pred):
		self.contador =self.contador +1
		print("Sequencia: "+ str(self.contador))
		print("classification_report(y_true, y_pred)")
		print(classification_report(y_true, y_pred))	
		print("\n")
		return accuracy_score(y_true, y_pred) 

	def processamento(self):
		self.leitura()
		self.contador=0
		print('Classificador Perceptron Multicamadas')
		classificadorMLP = MLPClassifier(hidden_layer_sizes=(57, 57, 57))  # funcao para Perceptron Multicamadas
		self.predicao = cross_val_predict(classificadorMLP, self.caracteristicas, self.rotulos, cv=10)
		self.score = cross_val_score(classificadorMLP, self.caracteristicas, self.rotulos, cv=10, scoring=make_scorer(self.relatorioClassificaçãoPontuaçãoPrecisao))
		
		print("Perceptron Multicamadas Impressão de todos os scores dos testes")
		print(self.score)
		print("\n")
		
		print("Perceptron Multicamadas score medio: ", str(self.score.mean() * 100) + ' %')
		print("Perceptron Multicamadas Acurácia (precisão média): ", str(accuracy_score(self.rotulos, self.predicao) * 100) + ' %')
		print("\n")
		
		matrizConfusao = confusion_matrix(self.rotulos, self.predicao)
		
		print("Matriz de Confusão Perceptron Multicamadas")
		print(pd.crosstab(self.rotulos, self.predicao, rownames=['Rotulo'], colnames=['Predição'], margins=True))
		print("\n")
		
		plt.matshow(matrizConfusao)
		plt.ylabel('Rotulo')
		plt.xlabel('Predição')
		plt.title('Perceptron Multicamadas Matriz de Confusão')
		plt.colorbar()
		plt.show()		


class FuncaoBaseRadial(RedesNeurais):
	
	def __init__(self):
		super().__init__()
		
	def processamento(self):
		
		self.leitura()
		print('Classificador Função Base Radial')
		classificadorFBR = GaussianProcessClassifier(kernel=1.0 * RBF(1.0)).fit(self.caracteristicas, self.rotulos)  # funcao para fbr
		self.predicao = classificadorFBR.predict(self.caracteristicas)
		validacaoCruzada = ShuffleSplit(n_splits=10, test_size=0.1)
		self.score = cross_val_score(classificadorFBR, self.caracteristicas, self.rotulos, cv=validacaoCruzada)
		
		print("Função Base Radial Impressão de todos os scores dos testes")
		print(self.score)
		print("\n")
		
		print("Função Base Radial score medio: ", str(self.score.mean() * 100) + ' %')
		print("Função Base Radial Acurácia (precisão média): ", str(accuracy_score(self.rotulos, self.predicao) * 100) + ' %')
		print("\n")
		
		print('Matriz de Confusão Função Base Radial')
		print(pd.crosstab(self.rotulos, self.predicao, rownames=['Conhecido'], colnames=['Perdição'], margins=True))
		
		matrizConfusao = confusion_matrix(self.rotulos, self.predicao)
		plt.matshow(matrizConfusao)
		plt.ylabel('Rotulo')
		plt.xlabel('Perdição')
		plt.title('Função Base Radial Matriz de Confusão')
		plt.colorbar()
		plt.show()
