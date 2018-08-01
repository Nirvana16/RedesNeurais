######################################################################################################################

#Este exemplo faz a identificação de numeros de 0 a 9 em uma resolução de 5x5 conforome demonstrad ona planilha de
#Apio "HORA DO SHOW", a ideia é fazer com que a rede aprenda até o ponto de que seja capaz de reconhecer os pixels
#correspondente aos numeros da matriz e dizer com precisão qual numero foi inserido
# Se voce chegou aqui de Google sugiro que antes de tentar criar um programa que resolva uma Epoca completa voce consiga
# ser capaz de resolver uma unica rodada (no meu programa chamei rodada de instante) então eu sugiro que voce comece
# a ler este código a partir da linha 116 e suba para ver as declarações de variaveis conforme elas forem aparecendo.

######################################################################################################################
# importa o plugin numpy que manipula as matrizes de modo mais facil no python
import numpy as np
# O numpy tem uma limitação na hora de printar os arrays longs, então uso a configuração abaixo pra definir
# o comprimento maximo do meu array antes que ele quebre a linha no terminal. Firula.
np.set_printoptions(linewidth=127)

# Estou criando um array de arrays, é como se fosse uma matriz, porém dentro deste array eu tenho outro(vetores)
# linha deste array, ou seja cada elemnto deste array x, contem um vetor de amostras de X, é como se em cada linha
# correspondesse a X1, X2, X3, X....X9.
#Cada vetor(linha) deste array(matriz) é a representação em pixel de um numero de zero a 9, estou dizendo pra minha
#rede que o numero O, 2, 3, 5 etc são representados na composição abaixo:

x = np.array([[1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1],
              [1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1],
              [1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1],
              [1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1],
              [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1],
              [1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1],
              [1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1],
              [1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1],
              [1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1],
              [1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1]])

# Segunda amostra, a ser utilizada apos treinar com a primeira
x_segunda_amostra = np.array([[1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1],
                              [1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1],
                              [1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1],
                              [1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1],
                              [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1],
                              [1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1],
                              [1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1],
                              [1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1],
                              [1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1],
                              [1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1]])

controlador = 1

# No Y desjado eu estou dizendo oomo deve estar ativo (1) o resultado da minha função para cada numero, ou seja
# cada linha do array abaixo representa um numero de 0 a 9 respectivamente, quando o numero eu quero que minha rede
# retorne o numero 9, por exemplo, significa que só vai estar ativa no ultimo digito e sendo intaiva (-1) par todo o
#restante.
yd = np.array([[1, -1, -1, -1, -1, -1, -1, -1, -1, -1], # 0
               [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1], # 1
               [-1, -1, 1, -1, -1, -1, -1, -1, -1, -1], # 2
               [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1], # 3
               [-1, -1, -1, -1, 1, -1, -1, -1, -1, -1], # 4
               [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1], # 5
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1], # 6
               [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1], # 7
               [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1], # 8
               [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1]]) # 9

#Gerar W aleatorio de forma elegante
#gera uma matriz 10 linhas com 26 colunas (ou neste caso 10 vetores com 26 elementos) contendo apenas 2 valores (0 ou 1)
w = np.random.randint(2, size=(10, 26))

#troca a matriz(array) w aonde o valor dentro dela for zero por -1
np.place(w, w==0, [-1])

# Geração "na mao" do W, que é a sinapse aleatória usada para treinar minha rede. Caso voce não manje nada de coisa
# alguma de Python e chegou aqui via Google depois que seu professor te pediu um exemplo de Perceptron sugiro que voce
# comente a as linhas de cima, onde é gerado o vetor W de forma aleatória, e descomente o vetor W abaixo.

# w = np.array([[1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1],
#               [1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1],
#               [-1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1],
#               [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1],
#               [-1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
#               [1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1],
#               [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1],
#               [1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1],
#               [-1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
#               [-1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1]])

#Variaveis de apoio #############################################################################################

# O erro médio é resetado a cada rodada do meu loop, então a variavel erroMedioEpocaGeral armazenda esse erro sem perder
# as informações
erroMedioEpocaGeral = np.array([[]])

# minha constante Sigma
sigma = 0.5

# Minha constante para erro maximo que aceito em minha rede.
# Troquei de 0.04 para ZERO para que a rede treine o maximo possivel antes de testa-la.
#erroMaximo = 0.04
erroMaximo = 0.00000

# Epoca inicial
epoca = 1

# Rodada / instante inicial
instante = 0

# Auto explicativo
escolha = 6

while escolha != 3:
    print("1 - Treinar")
    print("2 - Testar")
    print("3 - Sair")
    escolha = int(input("Choose your Destiny \n"))

    if escolha == 1:
        if epoca > 1:
            instante = 0
        while instante < len(x):

            # Inicia todos os arrays que irei usar de forma vazai
            vetorSomado = np.array([[]])
            yEncontrado = np.array([[]])
            eroou = np.array([[]])
            erroMedioEpoca = np.array([[]])
            deltaAux = np.array([[]])
            deltaW = np.array([[]])
            if instante == 0:
                erroMedioEpocaGeral = np.array([[]])

            # COMECE AQUI!!!  treinamento #######################################################################

            # o Array V vai receber o resultado da multiplicação de x por W
            # Aqui eu fiz um for in line, mas nao gosto de usar assim posi nao acho que fica algo facil
            # de ler ou interpretar. Cada linha de V ira receber o resultado da multiplicação linha a linha de W por
            # X "fixo" em 0 a 10.
            v = np.array([w * x[instante] for w in w])

            # Nesta fase eu preciso do valor somado de cada uma das linhas acima, então eu faço um novo for
            # onde o vetorSomado irá ser construido com os valores da soma de cada linha de v
            i = 0
            for apoio in v:
                vetorSomado = np.append(vetorSomado, [np.sum(v[i])])
                i = i + 1

            # Aqui eu aplico a função para verificar de os valores são positivos ou negativos
            # para cada caso o novo vetor Y recebera os valores 1 ou -1
            # Encontrar Y desejado ######################################################
            i = 0
            for apoio in vetorSomado:
                if vetorSomado[i] > 0:
                    yEncontrado = np.append(yEncontrado, [1])
                else:
                    yEncontrado = np.append(yEncontrado, [-1])
                i = i + 1

            # Agora a rede irá calcular o erro, ou seja, verificar quanto errado foi o valor do
            # Vetor Y que encontramos acima em comparação ao valor do Y desejado qeu definimos lá no começo

            # calcular erro ######################################################
            i = 0
            for apoio in yEncontrado:
                formulaDoErro = yd[i][instante] - (yEncontrado)[i]
                eroou = np.append(eroou, [formulaDoErro])
                i = i + 1

            # O calculo do erro médio para esta rodada é realizado agora
            # Calcular erro médio ######################################################
            somaDosErros = np.sum(eroou)
            totalDeerros = len(eroou)
            calculaerroMedioEpoca = somaDosErros / totalDeerros
            erroMedioEpoca = np.append(erroMedioEpoca, [calculaerroMedioEpoca])
            erroMedioEpocaGeral = np.append(erroMedioEpocaGeral, [erroMedioEpoca])


            # Atualizar Sinapse ######################################################
            # O For a seguir calcula separadamente o valor de sigma multiplicado pelos  erros e armazenda em SigamaXerro
            # em seguida uma variavel auxiliar sigmaXerro e multiplica pelo X
            # por ultimo o array deltaW armazenda os valores de deltaAux a cada ciclo do for
            i = 0
            tamanhodeW = len(w)
            for apoio in range(0, tamanhodeW):
                sigmaXerro = sigma * eroou[i]
                deltaAux = sigmaXerro * x[instante]
                deltaW = np.append(deltaW, [deltaAux])
                i = i + 1

            # Gambiarra para dividir o array
            # Por alguma razão a qual eu estava sem paciencia para descobrir quando adicionei elementos ao array
            # deltaW ele criou 1 unico array ao invés de criar varios, afinal o np.array cria um "array de arrays"
            # ou seja, um vetor de vetores, só que no deltaW ele fez 1 unico vetor de um cassetada de posições ao
            # invés de ter feito 10 vetores de 25 elementos (que era o que eu esperava) então precisei utilizar de
            # meios tecnicos de resolução emergencial para resultados não esperados em produção, Gambiarra!
            # a linha a seguir faz um SPLIT, divide, meu array deltaW a cada "comprimento de w" elementos
            deltaW = (np.split(deltaW, len(w)))

            # Soma o vetor W original com o valor de Delta W, sem a gambiarra isso não funciona.
            w = (np.add(w, deltaW))

            # Indica o final desta rodada
            instante = instante + 1

            ############################################################################################################

            # Se voce entendeu o que acontece até este ponto, beleza, volte até o inicio para entender como fiz o meu
            # laço de repetição e assim ser capaz de compreender, olhando as linhas a seguir, como meu programa muda as
            # rodadas (que por alguma razão que somente eu e Deus sabemos chamei de instante) e como ele muda as épocas.
            # se voce não entendeu, leia tudo de novo antes de ir pro proximo bloco!

            ############################################################################################################

            # Verifica se chegou a ultima rodada, caso verdadeiro calcular o erro médio quadratico.
            if instante == len(x):
                emqQuadrado = (erroMedioEpocaGeral * erroMedioEpocaGeral)
                emq = (np.sum(emqQuadrado)) / len(x)
                print("Erro medio Quadratico da Epoca N: ", epoca)
                print('= ', emq)
                print("")
                epoca = epoca + 1

                if emq > erroMaximo:
                    # epoca = epoca + 1
                    instante = 0
                    print("EMQ maior que erro maximo aceitavel, nova rodada.")

                if emq == 0 and controlador == 1:
                    print("\n\n\n###########################################")
                    print("treinamento para a segunda amostra")
                    controlador = 0
                    x = x_segunda_amostra
                    instante = 0
                    epoca = 1


    ############################################################################################################

    # O bloco a seguir é exclusivo para teste da rede, até o momento anterior a rede foi capaz de aprender,
    # chegou a hora de testar, se o EMQ foi igual a Zero isso significa que a rede é OBRIGADA a acertar, ou seja,
    # ela é obrigada a retornar o Y encontrado igual ao Y desejado quando voce inseri um vetor X já treinado
    # Essa rede faz isso, para mostrar que ela funciona e fazer com que o usuario não precise digitar 25 elementos
    # do vetor criei o menu a seguir o qual pode ser pulado caso o usuario queira.

    ############################################################################################################

    elif escolha == 2:
        print("Voce deseja usar vetores de treinamento ou informar um novo?")
        resposta = int(input("\nDigite 1 para usar os de treinamento \nDigite 2 para informar novo vetor \n"))
        if resposta == 1:

            # O bloco abaixo para testar, nele coloquei um menu onde voce escolhe vetore de 0 a 9 para testar se a rede funciona
            # sao os mesmos valores do vetor X LÁ DO COMEÇO, só que separado, com a rede funcionando perfeitamente o resultado W deverá ser
            # sempre o mesmo vetor inserido X, no caso o erro será zero e o Y encontrado igual ao Y desejado indicando que a rede nao aprendeu
            # mais nada, pois nao errou

            #############################################################################################
            debugar = int(input(" Escolha um numero de 0 até 9 \n"))
            if debugar == 0:
                vetorUsuario = np.array([[1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1]])
            if debugar == 1:
                vetorUsuario = np.array([[1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1]])
            if debugar == 2:
                vetorUsuario = np.array([[1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1]])
            if debugar == 3:
                vetorUsuario = np.array([[1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1]])
            if debugar == 4:
                vetorUsuario = np.array([[1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1]])
            if debugar == 5:
                vetorUsuario = np.array([[1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1]])
            if debugar == 6:
                vetorUsuario = np.array([[1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1]])
            if debugar == 7:
                vetorUsuario = np.array([[1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1]])
            if debugar == 8:
                vetorUsuario = np.array([[1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1]])
            if debugar == 9:
                vetorUsuario = np.array([[1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1]])

            ################################################################################################################

            #O Bloco a seguir recebe os valores do usuario e aloca e um vetor, trata-se de uma inserção chata pra caceta, mas nao tem outro jeito


            ############################################################################################################
            #Preencher vetor X
        else:
            vetorUsuario = np.array([[]])
            i = 0
            for apoio in range(0, 26):
                if i == 0:
                    valorDeX = int(1)
                else:
                    valorDeX = int(input("Informe o valor de x%d " % i))
                vetorUsuario = np.append(vetorUsuario, [valorDeX])
                i = i+1


        #############################################################################################################

        # A logica aqui é  a seguinte voce precisa inserior o vetor correspondente ao numero que voce quer reconher na rede
        # isso significa informar um novo vetor X personalizado, aqui chamei esse vetor de "vetorUsuario" o que a rede faz
        # é pegar esse vetor multiplicar pelo W treinado e obter o Y e ver se esse Y é igual ao Desejado, se for ele acertou
        # ou deduziu com base no que a rede aprende uaté agora.
        #
        # O certo seria ter jogado tudo em funções e chamar essas funções aqui, mas entender redes neurais já é complexo
        # tentar entender isso com uma programação estruturada ou Orientada objetos é mais complexo ainda, isso aqui
        # é para aprendizado e só

        #############################################################################################################
       # Iniciar vetores ------------------------------------------------------------
        instante = 0
        vetorSomado = np.array([[]])
        yEncontrado = np.array([[]])

        # Jogar o vetor informado na rede (treinar)-------------------------------------
        v = np.array([w * vetorUsuario for w in w])
        i = 0
        for apoio in v:
            vetorSomado = np.append(vetorSomado, [np.sum(v[i])])
            i = i + 1
        # Encontrar Y  -------------------------------------------------------------

        i = 0
        for apoio in vetorSomado:
            if vetorSomado[i] > 0:
                yEncontrado = np.append(yEncontrado, [1])
            else:
                yEncontrado = np.append(yEncontrado, [-1])
            i = i + 1
        #print(yEncontrado)

        numero = yEncontrado.tolist().index(1)
        print("o Vetor informado correponde ao numero: ", numero)


        # Exibir desenho

        matriz = np.array(vetorUsuario)
        remover = [0]
        matriz = np.delete(matriz, remover)
        matriz = matriz.astype(str)

        if resposta == 1:
            np.place(matriz, matriz == "-1", [" "])
            np.place(matriz, matriz == "1", ["X"])
        else:
            np.place(matriz, matriz == "-1.0", [" "])
            np.place(matriz, matriz == "1.0", ["X"])
        #print(matriz)

        print("")
        print(np.reshape(matriz, (5,5)))
        print("")

