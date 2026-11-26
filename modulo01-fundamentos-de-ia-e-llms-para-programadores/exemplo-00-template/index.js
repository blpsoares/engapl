import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();
    // Primeira camada da rede
    // entrada de 7 neurônios (idade, cor e localização)

    // Camada oculta com 80 neurônios e função de ativação ReLU
    // Quanto mais neuronios na camada oculta, mais complexa a rede pode ser
    //  mas também pode levar a overfitting se for exagerado.
    // inputShape: [7] indica que cada exemplo de entrada tem 7 características (idade, cor e localização)

    // A ReLU age como um filtro:
    // é como se ela deixasse somente os dados mais relevantes passarem para a próxima camada
    // ajudando a rede a aprender padrões mais complexos.
    // na categorizacao, se for negativo a relu desconsidera, se for positivo ela passa adiante.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Camada de saída com 3 neurônios (pq temos 3 categorias)
    // Um para cada categoria (premium, medium, basic)
    // A função de ativação softmax é usada para classificação multiclasse
    // pois ela transforma as saídas em probabilidades que somam 1.
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // Compilamos o modelo definindo a função de perda e o otimizador
    // Optmizer 'adam' é um algoritmo de otimização eficiente e amplamente utilizado para treinar redes neurais.
    // Ele ajusta os pesos da rede para minimizar a função de perda durante o treinamento.
    // Ele vai aprender com o historico de erros e acertos para melhorar a precisão do modelo.

    // A função de perda 'categoricalCrossentropy' é usada para problemas de classificação multiclasse
    // Ele compara o que o modelo "acha" que é a categoria correta (as probabilidades)
    // com o que realmente é (as labels one-hot encoded)
    //? one-hot encoded: é uma representação onde cada categoria é representada por um vetor binário
    //? onde apenas um elemento é 1 (indica a categoria correta) e os outros são 0.

    // maior erro (loss) ocorre quando o modelo atribui alta probabilidade para a categoria errada
    // Exemplo classico: classificacao de imagens, recomendação de produtos (entre outras recomendações), categorização
    model.compile({
        optimizer: 'adam', // algoritmo de otimização para ajustar os pesos da rede
        // (pesos = valores que a rede aprende durante o treinamento para fazer previsões)
        loss: 'categoricalCrossentropy', // função de perda para classificação multiclasse
        // (em resumo simples, é a medida de quão bem o modelo está se saindo em relação às categorias corretas)
        metrics: ['accuracy'] // Métrica para avaliar o desempenho do modelo durante o treinamento
    })


    const onEpochEnd = (epoch, logs) => console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc}`)
    // Treinamento do modelo
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0, // controla a quantidade de informações exibidas durante o treinamento (0 = sem logs, 1 = barra de progresso, 2 = uma linha por época)
            epochs: 100000, // número de vezes que o modelo vai passar por todo o conjunto de dados durante o treinamento
            shuffle: true, // embaralha os dados a cada época para melhorar a generalização do modelo e evitar overfitting
            //? overfitting: quando o modelo aprende tão bem os dados de treinamento que ele não consegue generalizar para novos dados
            //? (ele "decora" os dados de treinamento ao invés de aprender os padrões gerais)
            //? ele evita um viés no modelo
            callbacks: {
                onTrainEnd: () => console.log("Treinamento finalizado!"), // callback para quando o treinamento terminar
             }
        }
    )

    return model;
}

async function predict(model, pessoaTensorNormalizado) {
   // transformar o array js para o tensor tfjs
   const tfInput = tf.tensor2d(pessoaTensorNormalizado);

   // Faz a predicao (sera um vetor de 3 probabilidades)
   const pred = model.predict(tfInput);
   const predArray = await pred.array();
    return predArray[0].map((prob, index) => ({ prob, index }))
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded) - (one-hot encoded = representação binária onde cada categoria é representada por um vetor com um único valor 1 e os demais 0)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// Quanto mais dados melhor.
// assim o algoritmo consegue aprender melhor os padrões para novos casos.
const model = await trainModel(inputXs, outputYs);

model.save("file://./meu-modelo"); // Salva o modelo treinado localmente
// Exemplo de nova pessoa para previsão
const pessoa = {nome: "Bryan", idade: 28, cor: "verde", localizacao: "Curitiba"}
// Normalizando a idade da nova  pessoa usando o mesmo padrao do treino.
// Exemplo: idade_min = 28, idade_max = 40
// idade_normalizada = (idade - idade_min) / (idade_max - idade_min)

// const pessoaTensorNormalizado = [
//     [
//         0.2, // idade,
//         0,   // azul
//         0,   // vermelho
//         1,   // verde
//         0,   // São Paulo
//         0,   // Rio
//         1    // Curitiba

//     ]
// ];

// const predictions = await predict(model, pessoaTensorNormalizado);
// const results = predictions.map((({prob, index}) => {
//     return `Categoria: ${labelsNomes[index]} - Probabilidade: ${(prob * 100).toFixed(2)}%`
// })).join(`\n`);
