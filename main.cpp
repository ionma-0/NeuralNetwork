    #include <iostream>
    #include <fstream>
#include "Headers/NeuralNetwork.h"

int main(){
    //DECLARATION OF VARIABLES AND OBJECTS
    short opt = 0;
    float xorLearningRates[8] = {0.005f, 0.015f, 0.044f, 0.130f, 0.385f, 1.140f, 3.376f, 10.0f};
    Activation xorHiddenLayerActivations[4] = {Activation::ReLU, Activation::LeakyReLU, Activation::Sigmoid, Activation::Tanh};
    std::string filenames[4] = {"XOR_Training_Data/ReLU.csv", "XOR_Training_Data/LeakyReLU.csv", "XOR_Training_Data/Sigmoid.csv", "XOR_Training_Data/Tanh.csv"};

    //MENU
    std::cout << "MENU" << std::endl << "0)Exit\n1)XOR sweep\n\n" << "Choose an option: ";
    std::cin >> opt;

    switch(opt){
        default:
            break;
        case 1:
            
            for(int i = 0; i < 4 ; i++){ // i -> Hidden layer activation index
                std::ofstream writeFile(filenames[i]);
                if(!writeFile){
                    return 9;
                }else{
                    writeFile << "Learning Rate,Convergence,Mean Convergence Epoch,Saturated neurons,Mean Convergence Loss,Mean Non Convergence Loss,Mean Loss,Least Loss\n";
                
                    for(int j = 0; j < 8; j++){ // j -> Learning rate index
                        TrainingResult r;
                        SaturatedNeuronReport s;
                        float percentageOfConvergence = 0.0f;
                        float averageConvergenceLoss = 0.0f;
                        float averageNonConvergenceLoss = 0.0f;
                        float averageConvergenceEpoch = 0.0f;
                        float finishedWithOneOrMoreSaturatedNeuron = 0.0f;
                        float leastLoss = 1000.0f;

                        for(unsigned int k = 0; k < 500; k++){ // k -> Seed index
                            NeuralNetwork nn({2,2,1},xorLearningRates[j],xorHiddenLayerActivations[i],Activation::Sigmoid, k);

                            r = nn.train({{-1.0f,-1.0f},{-1.0f,1.0f},{1.0f,-1.0f},{1.0f,1.0f}},{{0.0f},{1.0f},{1.0f},{0.0f}},0.05f,(unsigned int)(2000.0f/xorLearningRates[j]));
                            s = nn.checkForSaturatedNeurons({{-1.0f,-1.0f},{-1.0f,1.0f},{1.0f,-1.0f},{1.0f,1.0f}},0.05f);
                            if(r.converged){
                                percentageOfConvergence+=1.0f;
                                averageConvergenceLoss+=r.loss;
                                averageConvergenceEpoch+=(float)r.epochs;
                            }else{
                                averageNonConvergenceLoss+=r.loss;
                            }

                            if(r.loss < leastLoss){
                                leastLoss = r.loss;
                            }

                            if(s.anySaturated){
                                finishedWithOneOrMoreSaturatedNeuron += 1.0f;
                            }
                            if((k+1)%100 == 0){
                                std::cout << std::endl << "Semilla: " << k << std::endl;
                            }
                        }

                        percentageOfConvergence = percentageOfConvergence/500.0f;
                        averageConvergenceLoss = averageConvergenceLoss/(500.0f*percentageOfConvergence);
                        averageNonConvergenceLoss = averageNonConvergenceLoss/(500.0f*(1.0f-percentageOfConvergence));
                        averageConvergenceEpoch = averageConvergenceEpoch/(500.0f*percentageOfConvergence);
                        finishedWithOneOrMoreSaturatedNeuron = finishedWithOneOrMoreSaturatedNeuron/5.0f;

                        std::cout << std::endl << (25.0f*((float)(i))) + (25.0f*((1.0f + (float)(j))/8.0f)) << std::endl;

                        writeFile << xorLearningRates[j] << "," << percentageOfConvergence*100.0f << "," << averageConvergenceEpoch << ","
                        << finishedWithOneOrMoreSaturatedNeuron << "," << averageConvergenceLoss << "," << averageNonConvergenceLoss << "," <<
                        percentageOfConvergence*averageConvergenceLoss + (1.0f-percentageOfConvergence)*averageNonConvergenceLoss << "," << leastLoss << "\n";

                        //std::cout << "\nTRAINING RESULTS | Activation: " << i << " | Learning rate: " << xorLearningRates[j]
                        //<< "\nPercentage of convergence: " << 100.0f*percentageOfConvergence << "%\nAverage convergence loss: " << averageConvergenceLoss
                        //<< "\nAverage non convergence loss: " << averageNonConvergenceLoss << "\nAverage convergence epoch: " << averageConvergenceEpoch <<
                        //"\nDead neurons: " << finishedBecauseDeadNeuron << "\n----------------------\n";
                    }
                }
            }

            break;
    }
    NeuralNetwork n1({2,2,2,1},0.1f,Activation::ReLU,Activation::Sigmoid, 2);

    n1.train({{-1.0f,-1.0f},{-1.0f,1.0f},{1.0f,-1.0f},{1.0f,1.0f}},{{0.0f},{1.0f},{1.0f},{0.0f}});

    return 0;
}