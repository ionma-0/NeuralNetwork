#include "../Headers/NeuralNetwork.h"
#include <iostream>
#include <cmath>
#include <algorithm>

// CONSTRUCTORS
NeuralNetwork::NeuralNetwork(std::vector<int> arch, float eta, Activation hiddenActivation, Activation lastActivation, unsigned int seed){
    architecture = arch;
    hiddenLayerActivation = hiddenActivation;
    lastLayerActivation = lastActivation;
    learningRate = eta;
    examples = 0;

    input.resize(arch[0], 0.0f);
    networkDepth = arch.size();

    for(int i = 1; i < networkDepth; i++){
        layers.emplace_back(arch[i-1],arch[i],seed + i);
    }
}


// PUBLIC METHODS
void NeuralNetwork::setLearningRate(float newLearningRate){
    learningRate = newLearningRate;
}

float NeuralNetwork::calculateLoss(const std::vector<float>& inData,const std::vector<float>& outData){
    float loss = 0.0f;
    int lastLayerSize = architecture[networkDepth - 1];

    if(inData.size() == (size_t)architecture[0] && outData.size() == (size_t)lastLayerSize){
        std::vector<float> prediction = forwardPropagation(inData);


        switch(lastLayerActivation){
            default:
            case Activation::Identity:
                for(int i = 0; i < lastLayerSize; i++){
                    loss += (outData[i] - prediction[i])*(outData[i] - prediction[i])/2.0f;
                }

                loss = loss/(float)lastLayerSize;
                break;

            case Activation::Sigmoid:{
                float epsilon = 1e-7f;
                float p;
                for(int i = 0; i < lastLayerSize; i++){
                    p = std::max(epsilon,std::min(1.0f-epsilon,prediction[i]));
                    loss += (-1)*(outData[i]*logf(p) + (1.0f-outData[i])*logf(1.0f-p));
                }
                loss = loss/(float)lastLayerSize;
                break;
            }

            case Activation::Softmax:{
                float epsilon = 1e-7f;
                float p;
                for(int i = 0; i < lastLayerSize; i++){
                    p = std::max(epsilon,std::min(1.0f-epsilon,prediction[i]));
                    loss += (-1)*outData[i]*logf(p);
                }
                break;
            }
        }
    }else{
        loss -= -1.0f;
    }
    
    return loss;
}

void NeuralNetwork::applyAllGradients(){
    if(examples != 0){
        for(int i = 0; i < networkDepth - 1; i++){
            layers[i].applyGradient(learningRate/(float)examples);
        }
        examples = 0;
    }
}

TrainingResult NeuralNetwork::train(const std::vector<std::vector<float>>& inputDataset,const std::vector<std::vector<float>>& outputDataset, float tolerance, unsigned int maxEpoch){
    TrainingResult result;
    float loss = 1.0f;
    unsigned int epoch = 0;
    size_t size = outputDataset.size();

    if(inputDataset.size() == size){
        while(loss > tolerance && epoch < maxEpoch){
            loss = 0.0f;

            for(size_t i = 0; i < size; i++){
                loss += calculateLoss(inputDataset[i], outputDataset[i]);
                backwardPropagation(outputDataset[i]);
                }
            

            applyAllGradients();

            epoch ++;
        }
    }else{
        exit(1);
    }

    result.epochs = epoch;
    result.loss = loss;
    result.converged = loss < tolerance;

    
    return result;
}

SaturatedNeuronReport NeuralNetwork::checkForSaturatedNeurons(const std::vector<std::vector<float>>& dataset, float tolerance){
    SaturatedNeuronReport report;

    bool anySaturated = false;
    std::vector<unsigned int> saturatedPerLayer;
    std::vector<std::vector<bool>> saturatedNeuronMask;
    if(tolerance > 0.25f){
        tolerance = 0.25f;
    }

    
    for(size_t i = 0; i < networkDepth - 2; i++){
        saturatedPerLayer.emplace_back(0);
        saturatedNeuronMask.emplace_back();

        for(size_t j = 0; j < architecture[i+1]; j++){
            saturatedNeuronMask[i].emplace_back(true);
        }
    }

    for(size_t i = 0; i < dataset.size(); i++){
        forwardPropagation(dataset[i]);

        for(size_t j = 0; j < networkDepth - 2; j++){
            const std::vector<float>& postAc = layers[j].postActivationValues();
            for(size_t k = 0; k < architecture[j+1]; k++){
                switch(hiddenLayerActivation){
                    default:
                    case Activation::ReLU:
                    case Activation::LeakyReLU:
                        if(postAc[k] > 0.0f){
                            saturatedNeuronMask[j][k] = false;
                        }
                        break;

                    case Activation::Sigmoid:
                        if(postAc[k] > tolerance && postAc[k] < (1.0f - tolerance)){
                            saturatedNeuronMask[j][k] = false;
                        }
                        break;

                    case Activation::Tanh:
                        if(postAc[k] < 1.0f - tolerance/2.0f && postAc[k] > tolerance/2.0f - 1.0f){
                            saturatedNeuronMask[j][k] = false;
                        }

                        break;


                }
            }
        }
    }


    for(size_t i = 0; i < networkDepth - 2; i++){
        for(size_t j = 0; j < architecture[i+1]; j++){
            if(saturatedNeuronMask[i][j]){
                anySaturated = true;
                saturatedPerLayer[i]++;
                report.howManySaturated++;
            }
        }
    }

    report.anySaturated = anySaturated;
    report.saturatedPerLayer = saturatedPerLayer;
    report.saturatedNeuronMask = saturatedNeuronMask;

    return report;
}


std::vector<float> NeuralNetwork::forwardPropagation(const std::vector<float>& in){
    examples++;
    input = in;
    layers[0].forward(input,hiddenLayerActivation);

    for(int i = 1; i < networkDepth - 2; i++){
        layers[i].forward(layers[i - 1].postActivationValues(),hiddenLayerActivation);
    }

    layers[networkDepth - 2].forward(layers[networkDepth - 3].postActivationValues(), lastLayerActivation);


    return layers[networkDepth - 2].postActivationValues();
}


std::vector<float> NeuralNetwork::output(){
    return layers[networkDepth-2].postActivationValues();
}

// PRIVATE METHODS
void NeuralNetwork::backwardPropagation(const std::vector<float>& data){
    std::vector<float> auxDelta;
    auxDelta = layers[networkDepth - 2].lastLayerBackProp(data, lastLayerActivation);
    

    for(int i = networkDepth - 3; i >= 0; i--){
        layers[i+1].updateGradient(auxDelta,layers[i].postActivationValues());
        auxDelta = layers[i+1].backProp(auxDelta, layers[i].postActivationValues(), hiddenLayerActivation);
    }

    layers[0].updateGradient(auxDelta,input);
}