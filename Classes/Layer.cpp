#include "../Headers/Layer.h"
#include <random>
#include <cmath>
#include <iostream>
#include <string.h>

// CONSTRUCTORS
//
Layer::Layer(int input, int output,unsigned int seed){
    inputSize = input;
    outputSize = output;

    biases.resize(output, 0.0f);
    preActivation.resize(output, 0.0f);
    postActivation.resize(output, 0.0f);
    accumulatedWeightGrad.resize(input*output, 0.0f);
    accumulatedBiasGrad.resize(output, 0.0f);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> uniform(-sqrt(6/(float)input),sqrt(6/(float)input));

    for (int i = 0; i < input*output; i++)
        weights.push_back(uniform(gen));

}

// PUBLIC METHODS
// Prints out weigths and biases and passes sample vectors through using relu activation
void Layer::testLayer(){
    std::vector<float> sample;

    // Print weights
    std::cout << "Weights:" << std::endl;
    for (int i = 0; i < outputSize; i++){
        for(int j = 0; j < inputSize; j++){
            std::cout << weights[i * inputSize + j] << "   ";
        }
        std::cout << std::endl;
    }



    // Forward and print sample vectors
    sample.resize(inputSize, -1.0f);
    sample[0] = 1.0f;

    std::cout << std::endl << std::endl << "Samples:" << std::endl;
    for (int i = 0; i < inputSize; i++){
        if(i>0){
            sample[i] = 1;
            sample[i-1] = -1;
        }

        forward(sample,Activation::ReLU);

        for (int j = 0; j < outputSize; j++){
            std::cout << postActivation[j] << "   ";
        }
        std::cout << std::endl;
    }

    return;
}



// Passes a vector through the layer and stores pre and post activation
void Layer::forward(const std::vector<float>& previousValues, Activation activationType){
    preActivation = biases;

    if(previousValues.size() == inputSize){
        for(int i = 0; i < outputSize; i++){
            for(int j = 0; j < inputSize; j++){
                preActivation[i] += weights[i * inputSize + j] * previousValues[j];
            }
        }
    }else{
        exit(2);
    }

    activation(activationType);
}



// Returns the pre activation of the layer
std::vector<float> Layer::preActivationValues(){
    return preActivation;
}



// Returns the post activation of the layer
std::vector<float> Layer::postActivationValues(){
    return postActivation;
}



//Takes in the previous postActivation vector and calculates the back propagation for the hidden layers according to the activation type
//Coded for Identity, ReLU, LeakyReLU, Standard Sigmoid and Tanh (excludes softmax as a hidden layer activation possibility)
//Returns delta^(l-1) vector
std::vector<float> Layer::backProp(const std::vector<float>& otherDelta,const std::vector<float>& aux, Activation activationType){
    std::vector<float> delta;
    delta.resize(inputSize, 0.0f);
    if(aux.size() == inputSize && otherDelta.size() == outputSize){
        for(int i = 0; i < outputSize; i++){
            for(int j = 0; j < inputSize; j++){
                delta[j] += otherDelta[i]*weights[i*inputSize+j];
            }
        }

        switch(activationType){
            default:
            case Activation::Identity:
                break;

            case Activation::ReLU:
                for(int i = 0; i < inputSize; i++){
                    if(aux[i] == 0.0f){
                        delta[i] = 0.0f;
                    }
                }
                break;

            case Activation::LeakyReLU:
                for(int i = 0; i < inputSize; i++){
                    if(aux[i] <= 0.0f){
                        delta[i] = delta[i]*0.1f;
                    }
                }
                break;

            case Activation::Sigmoid:
                for (int i = 0; i < inputSize; i++){
                    delta[i] = delta[i]*aux[i]*(1.0f - aux[i]);
                }
                break;

            case Activation::Tanh:
                for (int i = 0; i < inputSize; i++){
                    delta[i] = delta[i]*(1.0f - aux[i]*aux[i]);
                }
                break;
        }
    }else{
        exit(3);
    }
    return delta;
}



//Takes in the example as a vector of the same size as the output layer and calculates the back propagation of the last layer according to the activation type
//Coded for Identity + MSE (Regressions), Sigmoid + Mean Binary Cross Entropy and Softmax + Cross Entropy (Excludes Tanh, ReLU and LeakyRelu as last layer activation possibilities)
//Returns delta^l vector
std::vector<float> Layer::lastLayerBackProp(const std::vector<float>& y, Activation activationType){
    std::vector<float> delta;

    if(y.size() == outputSize){
        switch(activationType){
            default:
            case Activation::Identity: //Identity as the last layer activation + Mean Squared Error as the loss function
            case Activation::Sigmoid:  //Standard sigmoid as the last layer activation function + Mean binary croos entropy between all output neurons as loss function
                for(int i = 0; i < outputSize; i++){
                    delta.emplace_back((postActivation[i]-y[i])/(float)outputSize);
                }
                break;
            case Activation:: Softmax: //Softmax as the last layer activation function + Cross entropy as the loss function
                for(int i = 0; i < outputSize; i++)
                    delta.emplace_back((postActivation[i]-y[i]));
                break;
        }
    }else{
        exit (4);
    }

    return delta;
}



// Takes in a delta vector and updates the weights gradient and the biases gradient
void Layer::updateGradient(const std::vector<float>& delta,const std::vector<float>& previousPostActivation){
    if(delta.size() == outputSize && previousPostActivation.size() == inputSize){
        for (int i = 0; i < outputSize; i++){
            for (int j = 0; j < inputSize; j++){
                accumulatedWeightGrad[i*inputSize + j] += delta[i]*previousPostActivation[j];
            }
            accumulatedBiasGrad[i] += delta[i];
        }

    }else{
        if(delta.size() != outputSize && previousPostActivation.size() == inputSize){
            exit(6);
        }else if (delta.size() == outputSize && previousPostActivation.size() != inputSize){
            exit(7);
        }else{
            exit(8);
        }
        exit(5);
    }
}



// Takes in the learning rate, applies the gradient to the weights and biases and sets the gradient to zero
void Layer::applyGradient(float eta){
    for(int i = 0; i < outputSize*inputSize; i++){
        weights[i] = weights[i] - eta*accumulatedWeightGrad[i];
        accumulatedWeightGrad[i] = 0.0f;
    }

    for(int i = 0; i < outputSize; i++){
        biases[i] = biases[i] - eta*accumulatedBiasGrad[i];
        accumulatedBiasGrad[i] = 0.0f;
    }
}

// PRIVATE METHODS
void Layer::activation(Activation activation){
    switch(activation){
        default:
        case Activation::Identity:
            postActivation = preActivation;
            break;

        case Activation::ReLU:
            for (int i = 0; i < outputSize; i++){
                postActivation[i] = preActivation[i] > 0.0f ? preActivation[i] : 0.0f;
            }
            break;

        case Activation::LeakyReLU:
            for(int i = 0; i < outputSize; i++){
                postActivation[i] = preActivation[i] > 0.0f ? preActivation[i] : 0.1f*preActivation[i];
            }
            break;

        case Activation::Sigmoid:
            for(int i = 0; i < outputSize; i++){
                postActivation[i] = 1.0f/(1.0f + expf(-1.0f*preActivation[i]));
            }
            break;

        case Activation::Softmax:{
            float partition = 0.0f;
            for (int i = 0; i < outputSize; i++){
                partition += expf(preActivation[i]);
            }

            for (int i = 0; i < outputSize; i++){
                postActivation[i] = expf(preActivation[i])/partition;
            }
            break;
}

        case Activation::Tanh:
            for(int i = 0; i < outputSize; i++){
                postActivation[i] = tanhf(preActivation[i]);
            }
            break;
    }
}