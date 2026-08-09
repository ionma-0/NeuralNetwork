#pragma once
#include "Layer.h"
#define MAXEPOCHS 5000

struct TrainingResult{
    bool converged;
    float loss;
    unsigned int epochs;
};

struct SaturatedNeuronReport{
    bool anySaturated;
    unsigned int howManySaturated = 0;
    std::vector<unsigned int> saturatedPerLayer;
    std::vector<std::vector<bool>> saturatedNeuronMask; 
};

class NeuralNetwork{
    public:
    NeuralNetwork(std::vector<int> arch, float eta = 0.1f, Activation hiddenActivation = Activation::ReLU, Activation lastActivation = Activation::Identity, unsigned int seed = 0);

    void setLearningRate(float newLearningRate);
    float calculateLoss(const std::vector<float>& inData, const std::vector<float>& outData);
    void applyAllGradients();
    TrainingResult train(const std::vector<std::vector<float>>& inputDataset,const std::vector<std::vector<float>>& outputDataset, float tolerance = 0.05f, unsigned int maxEpoch = 20000);
    SaturatedNeuronReport checkForSaturatedNeurons(const std::vector<std::vector<float>>& dataset, float tolerance = 0.05f);
    std::vector<float> forwardPropagation(const std::vector<float>& in);
    std::vector<float> output();

    private:
    Activation hiddenLayerActivation;
    Activation lastLayerActivation;
    std::vector<float> input;
    std::vector<Layer> layers;
    std::vector<int> architecture;
    int networkDepth;
    int examples;
    float learningRate;
    unsigned int seed;

    void backwardPropagation(const std::vector<float>& data);
    

};
