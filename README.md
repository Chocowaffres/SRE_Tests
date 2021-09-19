# Máquina para Security Requirements Elicitation (SRE)

Inicialmente, o objetivo era arranjar um algoritmo de Machine Learning (ML) que conseguisse substituir o conjunto de if's e else's que existiam no módulo SRE. Posto isto, é necessário criar um dataset para que o modelo possa aprender. O que o modelo recebe são as respostas de um utilizador ao módulo e o que o modelo deve produzir é o conjunto de requisitos de segurança. Para modelos ML é necessário que toda a informação esteja sob o formato numérico e é necessário um conjunto de dados. 

## Criar o conjunto de dados

Criámos um programa em Python que gera todas as possíveis respostas e todos os possíveis outputs ao módulo SRE:
	- "reqs_B.py" (criado pelo Bernardo)
	- "reqs_normal.py" (criado por mim, com base no Bernardo)

## Analisar o problema

Posteriormente, analisámos quais os modelos que poderíamos utilizar para este tipo de problema, que é multi-label (representado pelas várias colunas de output) e binário (representado pela possibilidade de ser 0 ou 1 para cada coluna de output), e chegámos a conclusão que existiam os seguintes modelos possíveis:
	- Classifier Chains;
	- Binary Relevance;
	- Label Powersets;
	- Multi-label k-Nearest Neighbors;
	- Binary Relevance k-Nearest Neighbors;
	- Label Space Partitioning;
	- Majority Voting.

Os datasets de multi-label têm um formato específico denominado de ".arff", pelo que foi necessário transformar os dados gerados anteriormente para este formato:
	1) Pegar no ficheiro CSV gerado pelos reqs_normal.py (dataset_normal_train.csv);
	2) Criar um ficheiro ".arff" e colocar o formato necessário (dataset_normal_train.arff). Este formato tem de especificar a quantidade de inputs e de outputs;

O código associado aos classificadores apresentados em cima está contida no ficheiro "best_classifiers_old.py".

## Resultados

### Testar classificadores-base

No entanto, estes classificadores necessitam de um classificador base para poderem ser aplicados. Então optamos por utilizar alguns dos classificadores disponibilizados pelo sklearn e averiguar qual apresenta melhor performance. Os classificadores base utilizados foram:
	- AdaBoost (AB);
	- GradientBoostClassifier (GBC);
	- ExtraTreesClassifier (ETC);
	- RandomForestClassifier (RFC);
	- DecisionTreeClassifier (DTC);
	- ExtraTreeClassifier (ETC2);
	- KNeighborsClassifier (KNC);
	- RadiusNeighborClassifier (RNC);
	- SGDClassifier (SGD);
	- LogisticRegression (LR);

Como não conseguia correr no dataset completo, optei por correr com 3000 dados para treino e 600 dados para testes. Obtive os seguintes resultados (accuracy):

| Modelos/Classificadores             |  AB  |  GBC  |  ETC  |  RFC  |  DTC  |  ETC2 |  KNC  |  RNC  |  SGD  |  LR  |
|-------------------------------------|------|-------|-------|-------|-------|-------|-------|-------|-------|------|
| Classifier Chains                   | 96.8 |  N/A  |  95.8 |  95.7 |  95.0 |  89.3 |  94.3 |  80.7 |  N/A  |  N/A |  
| Binary Relevance                    | 96.8 |  N/A  |  95.7 |  96.5 |  95.2 |  91.7 |  94.2 |  80.7 |  N/A  |  N/A |
| Label Powerset                      | 96.8 |  N/A  |  96.8 |  96.8 |  95.0 |  91.0 |  94.0 |  80.7 |  N/A  |  N/A |
| Label Space Partitioning            | 96.8 |  N/A  |  95.7 |  96.7 |  95.3 |  90.8 |  94.2 |  80.7 |  N/A  |  N/A |
| Majority Voting                     | 96.8 |  N/A  |  96.2 |  96.5 |  95.5 |  92.2 |  94.2 |  80.7 |  N/A  |  N/A |

Quanto aos classificadores, Multi-label k-Nearest Neighbot e Binary Relevance k-Nearest Neighbor, como não precisam de um classificador base só foram testados uma única vez e obtive os resultados 90.4 e 93.1, respetivamente.

Os resultados que contêm N/A significam que não consegui correr para o conjunto de dados atual. Apresentava um problema a dizer que não havia mais do que uma class para certas labels (i.e., havia colunas do output que tinham sempre todos os resultados a "1"). Isto levou a que eu analisasse o dataset para tentar perceber de onde vinha este problema. Criei um ficheiro na drive (https://docs.google.com/spreadsheets/d/1NxfSC-64cSzSPUQGY9eiVE9jNTy7kq4_duCECfc9ZrE/edit?usp=sharing) onde coloquei tabelas que mostravam a partir de que altura um requisito de segurança passava a estar "ativo" (célula a verde quer dizer que foi ativado; célula a vermelho quer dizer que foi desativado). Consegui concluir que havia requisitos de segurança que estavam sempre a "1" e como tal não faria sentido estarem no conjunto de dados desta forma. Surgiram então duas soluções:
	- Colocar uma opção extra na 1ª questão do módulo, onde se assumia que nenhum requisito ficava ativo ("reqs_7options.py");
	- Remover a 1ª questão do módulo, visto ser a fonte do problema que surgiu ("reqs_16questions.py").

### Lidar com problemas no dataset

Posteriormente, também verificámos que o dataset era highly imbalanced (e.g., a proporção de "1" para uma determinada classe é muito superior à proporção de "0" da mesma classe). Posto isto, a métrica que estava a ser utilizada previamente ("Accuracy") não era representativa dos resultados que estavam a ser obtidos, pelo que se optou por alterar a métrica para "mean Accuracy". Adicionalmente, também era preciso que o modelo atribuí-se uma importância diferente a cada um dos resultados que produzia ("0" ou "1") de modo a conseguir aprender melhor num dataset highly imbalanced. Isto foi possível através da introdução da opção "class_weight='balanced'" no classificador base utilizado nos restantes classificadores. Estas alterações constam no ficheiro "best_classifier_all_balanced.py". Todos os testes realizados de agora em diante tiveram em consideração o conjunto de dados completo.

### Utilizar mean accuracy para avaliar imbalanced datasets

Analisámos os valores obtidos para "mean Accuracy", para cada uma das possíveis labels, entre diferentes classificadores-base. Destes classificadores-base excluímos KNC e RNC pelo custo temporal e de memória associado aos mesmos e também considerando a "Accuracy" obtida ser inferior aos restantes métodos.

**Classifier Chains**
| Labels/Classificadores |  DTC  |  ETC2 |  SGD  |  LR   |
|------------------------|-------|-------|-------|-------|
| Confidentiality        |  100  |  99.9 |  100  |  100  |
| Integrity              |  50   |  50   |  50   |  50   |
| Availability           |  50   |  50   |  50   |  50   |
| Authentication         |  100  |  96.8 |  100  |  100  |
| Authorization          |  100  |  76.7 |  94.5 |  95.2 |
| Non-Repudiation        |  100  |  98   |  99.3 |  99.3 |
| Accountability         |  100  |  93.7 |  96.9 |  93.6 |
| Reliability            |  100  |  90.5 |  96.5 |  97.0 |
| Privacy                |  100  |  99.1 |  97.7 |  97.2 |
| Physical Security      |  100  |  73.3 |  98.5 |  98.5 |
| Forgery Resistance     |  100  |  100  |  100  |  100  |
| Tamper Detection       |  100  |  96   |  88.6 |  89   |
| Data Freshness         |  100  |  99.2 |  100  |  100  |
| Confinement            |  100  |  97.3 |  100  |  100  |
| Interoperability       |  100  |  100  |  100  |  100  |
| Data Origin            |  100  |  100  |  100  |  100  |

**Binary Relevance**
| Labels/Classificadores |  DTC  |  ETC2 |  SGD  |  LR   |
|------------------------|-------|-------|-------|-------|
| Confidentiality        |  100  |  100  |  100  |  100 |
| Integrity              |  50   |  50   |  50   |  50   |
| Availability           |  50   |  50   |  50   |  50   |
| Authentication         |  100  |  96.4 |  100  |  100  |
| Authorization          |  100  |  95.7 |  94.7 |  95.1 |
| Non-Repudiation        |  100  |  93   |  98.6 |  99.3 |
| Accountability         |  100  |  97   |  93.8 |  93.6 |
| Reliability            |  100  |  95.2 |  95.6 |  94.2 |
| Privacy                |  100  |  98.8 |  97.3 |  97   |
| Physical Security      |  100  |  100  |  97.2 |  98.4 |
| Forgery Resistance     |  100  |  98.4 |  100  |  100  |
| Tamper Detection       |  100  |  98   |  97.2 |  92.7 |
| Data Freshness         |  100  |  99.3 |  100  |  100  |
| Confinement            |  100  |  98.8 |  92.4 |  100  |
| Interoperability       |  100  |  100  |  100  |  100  |
| Data Origin            |  100  |  100  |  100  |  100  |

**Label Powerset**
| Labels/Classificadores |  DTC  |  ETC2 |  SGD  |  LR   |
|------------------------|-------|-------|-------|-------|
| Confidentiality        |  100  |  99.1 |  46.7 |  100  |
| Integrity              |  50   |  50   |  50   |  50   |
| Availability           |  50   |  50   |  50   |  50   |
| Authentication         |  100  |  91.3 |  56.1 |  96.2 |
| Authorization          |  100  |  82.8 |  61.8 |  73.3 |
| Non-Repudiation        |  99.9 |  94.2 |  56.9 |  93.8 |
| Accountability         |  100  |  85.1 |  65.5 |  90.8 |
| Reliability            |  100  |  76.7 |  63   |  54.1 |
| Privacy                |  100  |  98.1 |  46.6 |  98.3 |
| Physical Security      |  100  |  78.6 |  57.8 |  74   |
| Forgery Resistance     |  100  |  94.6 |  55.7 |  99.5 |
| Tamper Detection       |  99.9 |  94.4 |  55.9 |  88.7 |
| Data Freshness         |  100  |  98.1 |  60.1 |  100  |
| Confinement            |  100  |  97.5 |  53   |  97.9 |
| Interoperability       |  100  |  98.8 |  59.3 |  100  |
| Data Origin            |  100  |  98.5 |  59.1 |  100  |

**Multi-label k-Nearest Neighbors**
| Labels/Classificadores |   -   |
|------------------------|-------|
| Confidentiality        |  97.4 |
| Integrity              |  50   |
| Availability           |  50   |
| Authentication         |  84.4 |
| Authorization          |  87.7 |
| Non-Repudiation        |  89.8 |
| Accountability         |  95.7 |
| Reliability            |  92.0 |
| Privacy                |  97   |
| Physical Security      |  84.4 |
| Forgery Resistance     |  90.8 | 
| Tamper Detection       |  90.8 |
| Data Freshness         |  91.6 |
| Confinement            |  91.7 |
| Interoperability       |  92.6 |
| Data Origin            |  92.5 |

**Binary Relevance k-Nearest Neighbors**
| Labels/Classificadores |   -   |
|------------------------|-------|
| Confidentiality        |  100  |
| Integrity              |  50   |
| Availability           |  50   |
| Authentication         |  88.9 |
| Authorization          |  94.3 |
| Non-Repudiation        |  95.3 |
| Accountability         |  100  |
| Reliability            |  98.0 |
| Privacy                |  99.9 |
| Physical Security      |  88.1 |
| Forgery Resistance     |  98.1 |
| Tamper Detection       |  99   |
| Data Freshness         |  99.1 |
| Confinement            |  99.2 |
| Interoperability       |  100  |
| Data Origin            |  100  |

**Label Space Partitioning**
| Labels/Classificadores |  DTC  |  ETC2 |  SGD  |  LR   |
|------------------------|-------|-------|-------|-------|
| Confidentiality        |  100  |  99.8 |  100  |  100  |
| Integrity              |  50   |  50   |  50   |  50   |
| Availability           |  50   |  50   |  50   |  50   |
| Authentication         |  100  |  96.9 |  100  |  100  |
| Authorization          |  100  |  91.8 |  95.5 |  95.1 |
| Non-Repudiation        |  100  |  97.6 |  98.8 |  99.3 |
| Accountability         |  100  |  93.7 |  96.4 |  93.6 |
| Reliability            |  100  |  97.4 |  95.8 |  94.2 |
| Privacy                |  100  |  98.3 |  97.9 |  97   |
| Physical Security      |  100  |  88.2 |  97.7 |  98.4 |
| Forgery Resistance     |  100  |  98.1 |  100  |  100  |
| Tamper Detection       |  100  |  98.4 |  92.7 |  92.7 |
| Data Freshness         |  100  |  99.6 |  100  |  100  |
| Confinement            |  100  |  98.8 |  92.1 |  100  |
| Interoperability       |  100  |  99.8 |  100  |  100  |
| Data Origin            |  100  |  100  |  100  |  100  |

**Majority Voting**
| Labels/Classificadores |  DTC  |  ETC2 |  SGD  |  LR   |
|------------------------|-------|-------|-------|-------|
| Confidentiality        |  100  |  99.7 |  100  |  100  |
| Integrity              |  50   |  50   |  50   |  50   |
| Availability           |  50   |  50   |  50   |  50   |
| Authentication         |  100  |  97.4 |  99.5 |  100  |
| Authorization          |  100  |  93.8 |  95.5 |  95.1 |
| Non-Repudiation        |  100  |  95.7 |  99.3 |  99.3 |
| Accountability         |  100  |  96   |  93.6 |  93.6 |
| Reliability            |  100  |  90   |  94.2 |  94.2 |
| Privacy                |  100  |  98.1 |  97.9 |  97   |
| Physical Security      |  100  |  80.6 |  97.3 |  98.4 |
| Forgery Resistance     |  100  |  97.5 |  98.8 |  100  |
| Tamper Detection       |  100  |  97   |  92.7 |  92.7 |
| Data Freshness         |  100  |  99.5 |  99.9 |  100  |
| Confinement            |  100  |  99.8 |  92   |  100  |
| Interoperability       |  100  |  100  |  100  |  100  |
| Data Origin            |  100  |  100  |  100  |  100  |

Da análise das tabelas anteriores é possível concluir que os métodos que obtiveram melhores resultados foram: Classifier Chains, Binary Relevance, Label Space Partitioning e Majority Voting. Também é possível concluir que o classificador-base DecisionTreeClassifier é o que obtém melhores resultados ao longo de todos os classificadores. Os valores de 50% para a Integrity e Availability são justificados pelo uso do conjunto de dados normais e que não contém nenhum valor "0".

### Análise temporal dos modelos utilizados

De seguida, fizemos uma análise temporal, isto é, quanto tempo demorava a treinar e testar cada um dos classificadores escolhidos, mantendo o mesmo classificador base (DecisionTreeClassifier). Estes resultados estão armazenados nos ficheiros com o formato "*_time_dectree_normal.txt", onde "*" corresponde a um nome similar ao classificador analisado. 

| Modelos                             |  Tempo (s)  |
|-------------------------------------|-------------|
| Classifier Chains                   |   16.445    |
| Binary Relevance                    |   14.863    |
| Label Powerset                      |   21.614    |
| Label Space Partitioning            |   21.242    |
| Majority Voting                     |   29.295    |
| Multi-label k-Nearest Neighbor      |   847.902   |
| Binary Relevance k-Nearest Neigbor  |   227.357   |

Através da tabela é possível concluir que os 3 classificadores mais rápidos são: Binary Relevance, Classifier Chains e Label Space Partitioning, por ordem de rapidez. Com base no que foi apresentado anteriormente, pode-se afirmar que os valores para DecisionTreeClassifier são iguais ao longo deste três classificadores. Assim, os resultados analisados de seguida focaram-se principalmente nos três classificadores que apresentavam melhores resultados e melhor performance temporal: Binary Relevance, Classifier Chains e Label Space Partitioning, utilizando o classificador-base DecisionTreeClassifier (ficheiro "best_classifiers_balanced.py").

### Análise do número mínimo de dados necessários para o modelo ter performance razoável

Visto que o objetivo era conseguir utilizar o menor conjunto de dados possíveis para o modelo aprender facilmente quais os outputs corretos, analisámos quais os resultados que eram obtidos treinando o modelo com 250 até 1000 dados, testando sempre o modelo no conjunto de teste completo. Dado que os valores de "mean accuracy" para a Integrity e Availability são 50 para todos os testes efetuados, estes não foram considerados para o cálculo da média dos valores de "mean accuracy" apresentados na tabela abaixo. Referência: "bal_acc_dectree_normal_train_diff_size_*.txt".

| Modelos/Tamanho do treino           |  250  |  300  |  350  |  400  |  450  |  500  |  550  |  600  |  650  |  700  |  750  |  800  |  850  |  900  |  950  |  1000 |
|-------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Classifier Chains                   |  98.4 |  99.3 |  99.1 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
| Binary Relevance                    |  98.4 |  99.0 |  99.1 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
| Label Space Partitioning            |  98.4 |  99.1 |  98.6 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
|-------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Classifier Chains                   |  98.3 |  99.3 |  99.1 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
| Binary Relevance                    |  99.6 |  99.0 |  98.6 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
| Label Space Partitioning            |  98.4 |  99.6 |  98.4 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
|-------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Classifier Chains                   |  99.5 |  99.1 |  98.4 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
| Binary Relevance                    |  99.6 |  99.1 |  98.6 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
| Label Space Partitioning            |  99.6 |  99.3 |  98.5 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
|-------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Classifier Chains                   |  98.3 |  99.1 |  99.1 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
| Binary Relevance                    |  98.4 |  99.3 |  98.4 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
| Label Space Partitioning            |  98.4 |  99.1 |  99.1 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
|-------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Classifier Chains                   |  99.6 |  99.7 |  99.1 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
| Binary Relevance                    |  98.4 |  98.4 |  98.5 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |
| Label Space Partitioning            |  99.6 |  99.7 |  98.5 |  99.7 |  99.8 |  99.8 |  99.8 |  99.8 |  99.9 |  99.9 |  99.9 |  100  |  100  |  100  |  100  |  100  |

Analisando a tabela acima, é possível observar que sistematicamente os modelos têm 99.7 a partir do tamanho 400. Dado que o conjunto de treino é fixo, isto é, o modelo analisa sempre as mesmas 400 instâncias, pode-se especular que o modelo ao observar estas 400 instâncias consegue aprender casos que tinha observado em reduzida quantidade. Observando as primeiras 3 colunas da tabela é possível perceber que o modelo que apresenta melhor capacidade de generalização é o Classifier Chains, que também é o 2º modelo mais rápido. Classifier Chains divide problemas multi-label em vários problemas de uma única label e cria ligações entre os diferentes problemas (https://www.google.com/search?q=classifier+chains&sxsrf=ALeKk01DygR--kTufocfrgQQCp3uO8bYnw:1621522706916&source=lnms&tbm=isch&sa=X&ved=2ahUKEwi1lcPnwtjwAhWQFMAKHRDpDFAQ_AUoAXoECAEQAw&biw=1853&bih=953#imgrc=3CUHzaD1gH36YM). Quanto ao Binary Relevance, este somente divide os problemas multi-label em vários problemas de uma única label (https://xang1234.github.io/images/multilabel/binary-relevance.jpg). O aumento de accuracy em conjuntos de dados de menores dimensões associado ao Classifier Chains permite perceber que o conjunto de dados utilizado apresenta alguma ligação entre as diferentes labels apresentadas, e daí o melhor desemepnho por parte de Classifier Chains.

Por fim, surgiu a "máquina" ("machine_SRE.py") que tem a capacidade de aprender com o utilizador, isto é, o utilizador dá um conjunto de respostas ao módulo SRE e o modelo apresenta um conjunto de requisitos de segurança para as respostas dadas. De seguida, o utilizador é questionado sobre a pertinência dos requisitos de segurança que o modelo conseguiu prever. Deste modo, constrói-se um dataset com os inputs (respostas às questões) dadas pelo utilizador e com os ouputs (produzidos pelo modelo e confirmados pelo utilizador). Treina-se o modelo de modo a aprender com base neste conjunto de dados e de seguida as respostas dadas pelo modelo já devem seguir as orientações dadas pelo utilizador. Esta máquina tem o modelo Classifier Chains com DecisionTreeClassifier como base, que foram os modelos que apresentaram melhores resultados no final de todas estas experiências.

## NOTAS

É necessário ter um artigo que suporte o que é dito sobre Classifier Chains e Binary Relevance. Tal pode constar numa secção denominada "Background", onde também apresentas os restantes modelos que falamos nesta experiência.
Mais casas decimais e métrica f1_score também é apresentada nos ficheiros respeitantes a cada tabela.
Pode ser necessário fazer uma descrição do conjunto de dados utilizados. 
	Normal: Treino: 70093 / Teste: 17457.
	7 opções: Treino: 81667 / Teste: 20475.
	16 questões: Treino: 11668 / Teste: 2922.
Pode ser importante realçar que, para o caso da última experiência (análise do tamanho), é possível através de 400 instâncias de treino o modelo acertar 17405 (round_up(17457 * 0.997)) instâncias.
Os conjuntos de dados 7 opções e 16 questões são apenas uma solução para a mean accuracy de Integrity e Availability não darem 50%. A melhor solução é o uso de 7 opções na primeira questão, em que é adicionada a opção "Outro" e que não "ativa" nenhum requisito de segurança. Isto permite ao modelo aprender quando é que estas duas labels têm o valor "0". 