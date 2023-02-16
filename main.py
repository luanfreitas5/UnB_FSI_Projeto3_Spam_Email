# Nome: Luan Mendes Gonçalves Freitas
# Matricula: 15/0015585
# Disciplina: Fundamentos de Sistemas Inteligentes
# Projeto Final - Detecção de Spam baseado na suposição de um conteúdo diferente de um e-mail legítimo
# Módulo main
import os

from redesNeurais_150015585 import PerceptronMulticamadas, FuncaoBaseRadial

def main():
    opcao = 0
    while opcao < 4:
        print('1 - Executar Algoritmo Perceptron Multicamadas')
        print('2 - Executar Algoritmo Função Base Radial')
        print('3 - Executar Ambos')
        print('4 - Sair')
        opcao = int(input('Digite uma opção '))
        if opcao == 1:
            os.system('clear') or None
            mpl = PerceptronMulticamadas()
            mpl.processamento()
        elif opcao == 2:
            os.system('clear') or None
            fbr = FuncaoBaseRadial()
            fbr.processamento()
        elif opcao == 3:
            os.system('clear') or None
            mpl = PerceptronMulticamadas()
            mpl.processamento()
            fbr = FuncaoBaseRadial()
            fbr.processamento()
        elif opcao != 4:
            os.system('clear') or None
            print('Invalido opção digite novamente ')
            opcao = 0    
    
    print("Fim")
    
if __name__ == '__main__':
    main()
