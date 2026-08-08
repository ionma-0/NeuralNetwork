#pragma once
#include <vector>

enum class Activation{
    Identity,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU
};

class Layer{
    public:
    Layer(int input, int output,unsigned int seed=0);

    void testLayer(); // Prints out weigths and biases and passes sample vectors through using relu activation
    void forward(const std::vector<float>& previousValues, Activation activationType); // Passes a vector through the layer and stores pre and post activation
    std::vector<float> preActivationValues(); // Returns the pre activation of the layer
    std::vector<float> postActivationValues(); // Returns the post activation of the layer
    std::vector<float> backProp(const std::vector<float>& otherDelta,const std::vector<float>& aux, Activation activationType); //Takes in the previous postActivation vector and calculates the back propagation for the hidden layers according to the activation type
    std::vector<float> lastLayerBackProp(const std::vector<float>& y, Activation activationType); //Takes in the example as a vector of the same size as the output layer and calculates the back propagation of the last layer according to the activation type
    void updateGradient(const std::vector<float>& delta,const std::vector<float>& previousPostActivation); // Takes in a delta vector and updates the weights gradient and the biases gradient
    void applyGradient(float eta); // Takes in the learning rate, applies the gradient to the weights and biases and sets the gradient to zero



    private:
    int inputSize;
    int outputSize;

    std::vector<float> preActivation;
    std::vector<float> postActivation;
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<float> accumulatedWeightGrad;
    std::vector<float> accumulatedBiasGrad;
    

    void activation(Activation activation = Activation::ReLU);
    
};