Ik wil naast mijn bestaande C++ VectorDatabase een apart leerproject bouwen waarin ik zelf begrijp en implementeer hoe een neural network en uiteindelijk een embedding model werken.

## Mijn huidige VectorDatabase

Ik heb een C++ VectorDatabase gebouwd die numerieke vectoren opslaat en vergelijkt.

Belangrijke onderdelen:

* `Vector`
* `KnowledgeItem`
* `VectorDatabase`
* `findNearest()`
* `findNearestK()`
* KD-tree
* verschillende graph/KNN-functionaliteit

Een `KnowledgeItem` ziet er ongeveer zo uit:


struct KnowledgeItem
{
    Vector vector;
    std::string text;
};


De database weet niets van de betekenis van tekst. Hij werkt alleen met numerieke vectoren en afstanden.

De huidige pipeline is:

```text
tekst
 ↓
pretrained embedding model
 ↓
384-dimensionale vector
 ↓
KnowledgeItem
 ↓
VectorDatabase
 ↓
nearest-neighbor search
```

Ik gebruik momenteel Python + `sentence-transformers/all-MiniLM-L6-v2` om tekst om te zetten naar vectors van 384 dimensies.

Bijvoorbeeld:

```text
"Elektrische auto's gebruiken een elektromotor en een batterij."
        ↓
[0.12, -0.43, 0.87, ...]
        ↓
384 getallen
```

De C++ VectorDatabase kan deze vectors vervolgens opslaan en zoeken.

## Het nieuwe project

Ik wil nu een apart project bouwen om zelf te leren hoe een neural network werkt.

Dit project hoeft in eerste instantie absoluut geen serieus concurrerend embeddingmodel te worden.

Het doel is:

**begrijpen wat er onder de motorkap gebeurt.**

Ik wil het stap voor stap bouwen.

## Gewenste leerroute

Begin klein en bouw steeds verder:

### 1. Eén neuron

Ik wil begrijpen:

* inputs
* weights
* bias
* weighted sum
* activation function
* output

Bijvoorbeeld conceptueel:

```text
inputs
  ↓
weights
  ↓
weighted sum + bias
  ↓
activation
  ↓
output
```

### 2. Meerdere neuronen

Daarna begrijpen hoe meerdere neuronen samen een layer vormen.

### 3. Meerdere layers

Daarna een klein neural network:

```text
input layer
     ↓
hidden layer
     ↓
output layer
```

Ik wil hierbij begrijpen wat er tijdens een forward pass gebeurt.

### 4. Loss function

Daarna wil ik leren hoe het netwerk meet hoe fout de voorspelling is.

### 5. Backpropagation

Daarna wil ik zelf begrijpen en implementeren hoe het netwerk bepaalt welke weights aangepast moeten worden.

Ik wil dus niet alleen een library gebruiken, maar begrijpen waar de gradients vandaan komen.

### 6. Gradient descent

Daarna wil ik leren hoe de weights daadwerkelijk worden aangepast.

### 7. Training

Vervolgens een klein netwerk daadwerkelijk trainen op eenvoudige data.

Bijvoorbeeld:

```text
input → gewenste output
```

en het netwerk steeds laten verbeteren.

### 8. Embeddings

Pas daarna wil ik onderzoeken hoe we van een neural network naar een embedding kunnen gaan.

Het uiteindelijke idee:

```text
input
 ↓
neural network
 ↓
embedding
 ↓
bijvoorbeeld 384 getallen
```

Een embedding is dus een numerieke representatie die later door mijn VectorDatabase gebruikt kan worden.

## Belangrijk architectuuridee

Ik wil het neural-network-project voorlopig als een apart project houden.

Bijvoorbeeld:

```text
Development/Public/
├── VectorDatabase/
├── EmbeddingModel/
└── SearchEngine/
```

Het embeddingproject produceert uiteindelijk vectors.

Mijn bestaande VectorDatabase consumeert die vectors.

Dus:

```text
Embedding Model
      ↓
    Vector
      ↓
KnowledgeItem
      ↓
VectorDatabase
```

Het VectorDatabase-project hoeft niet te weten hoe de vector is gemaakt.

Het kan dezelfde `Vector` gebruiken ongeacht of de vector afkomstig is van:

* mijn eigen neural network
* `sentence-transformers`
* een ander embeddingmodel
* een API

## Belangrijk onderscheid

Mijn huidige `all-MiniLM-L6-v2` is een bestaand pretrained embeddingmodel.

Het nieuwe project is juist bedoeld om zelf te leren hoe neural networks en embeddings werken.

Ik verwacht dus niet dat mijn eerste eigen model meteen vergelijkbaar is met MiniLM.

Het doel is eerst:

**één neuron → meerdere neuronen → layers → training → kleine embeddings → uiteindelijk tekstembeddings.**

## Hoe ik graag leer

Ik leer het beste door het zelf te bouwen.

Geef daarom kleine stappen en leg iedere belangrijke regel code uit.

Niet meteen een enorm framework of complete neural-network-library neerzetten.

Ik wil begrijpen:

* waarom iets nodig is
* wat iedere variabele betekent
* wat er mathematisch gebeurt
* hoe de code overeenkomt met de wiskunde
* waarom we een bepaalde architectuur kiezen

Gebruik bij voorkeur C++ voor de neural-networkimplementatie, omdat mijn VectorDatabase ook in C++ is gebouwd.

Python mag gebruikt worden voor experimenten of visualisaties als dat nuttig is.

Begin helemaal bij **één neuron** en bouw vanaf daar langzaam verder.

Later wil ik het eigen embeddingmodel uiteindelijk kunnen aansluiten op mijn bestaande C++ VectorDatabase.
