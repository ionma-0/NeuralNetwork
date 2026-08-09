    #include <iostream>
    #include <fstream>
    #include <cmath>
    #include "Headers/NeuralNetwork.h"

    #define PI_FLOAT 3.14159f
    #define SQRT_3F 1.73205f

int main(){
    //DECLARATION OF VARIABLES AND OBJECTS
    //General
    short opt = 0;
    Activation hiddenLayerActivations[4] = {Activation::ReLU, Activation::LeakyReLU, Activation::Sigmoid, Activation::Tanh};
    std::ofstream writeFile;

    //XOR sweep variables
    float xorLearningRates[8] = {0.005f, 0.015f, 0.044f, 0.130f, 0.385f, 1.140f, 3.376f, 10.0f};
    std::string xorFileNames[4] = {"ExperimentData/XOR_Training_Data/ReLU.csv", "ExperimentData/XOR_Training_Data/LeakyReLU.csv", "ExperimentData/XOR_Training_Data/Sigmoid.csv", "ExperimentData/XOR_Training_Data/Tanh.csv"};

    //Sin(x) training variables
    char sinAdaptiveSampling;
    size_t sinNumberOfPoints;
    size_t sinNumberOfHiddenLayers;
    std::vector<int> sinTrainingArchitecture;
    std::vector<std::vector<float>> sinInputData;
    std::vector<std::vector<float>> sinOutputData;
    std::string sinFileNames[4] = {"ExperimentData/Sin_Training_Data/ReLU.csv", "ExperimentData/Sin_Training_Data/LeakyReLU.csv", "ExperimentData/Sin_Training_Data/Sigmoid.csv", "ExperimentData/Sin_Training_Data/Tanh.csv"};
    std::vector<float> xPartition;
    std::vector<float> neuralNetworkYPrediction;
    std::vector<float> yPartition;


    //MENU
    std::cout << "MENU" << std::endl << "0)Exit\n1)XOR sweep\n2)Train sin(x)\n\n" << "Choose an option: ";
    std::cin >> opt;

    switch(opt){
        default:
            break;
        case 1:
            
            for(int i = 0; i < 4 ; i++){ // i -> Hidden layer activation index
                writeFile.open(xorFileNames[i]);
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
                            NeuralNetwork nn({2,2,1},xorLearningRates[j],hiddenLayerActivations[i],Activation::Sigmoid, k);

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

                    writeFile.close();
                }
            }

            break;

            case 2:

                //Parameter selection for the training
                std::cout << "\nHow many points do you want to train the network with?\n";
                std::cin >> sinNumberOfPoints;

                std::cout << "\nWould you like to use curvature-adaptive sampling of the points instead of uniform? (y/n)\n";
                std::cin >> sinAdaptiveSampling;

                std::cout << "\nHow many hidden layers do you want in the Neural Network?\n";
                std::cin >> sinNumberOfHiddenLayers;

                sinTrainingArchitecture.resize(sinNumberOfHiddenLayers + 2, 0);
                sinTrainingArchitecture[0] = 1;
                sinTrainingArchitecture[sinNumberOfHiddenLayers+1] = 1;
                for(int i = 1; i < sinNumberOfHiddenLayers+1; i++){
                    std::cout << "\nChoose how many neurons you want in hidden layer " << i << ": ";
                    std::cin >> sinTrainingArchitecture[i];
                }

                //Training data generation
                if(sinAdaptiveSampling == 'y'){
                    sinNumberOfPoints += 1 - (sinNumberOfPoints%2);
                }

                sinInputData.resize(sinNumberOfPoints,{});
                sinOutputData.resize(sinNumberOfPoints,{});

                sinInputData[0].emplace_back(-PI_FLOAT);
                sinOutputData[0].emplace_back(std::sin(sinInputData[0][0]));
                sinInputData[sinNumberOfPoints-1].emplace_back(PI_FLOAT);
                sinOutputData[sinNumberOfPoints-1].emplace_back(std::sin(sinInputData[sinNumberOfPoints-1][0]));
                for(size_t i = 1; i < sinNumberOfPoints-1; i++){
                    if(sinAdaptiveSampling == 'y'){
                        if(i <= (sinNumberOfPoints-1)/2){
                            float arg1 = std::max(-1.0f, std::min(1.0f, 4.0f/((float)(sinNumberOfPoints-1)) + std::cos(sinInputData[i-1][0])));
                            sinInputData[i].emplace_back(-std::acos(arg1));
                        }else{
                            float arg2 = std::max(-1.0f, std::min(1.0f, std::cos(sinInputData[i-1][0]) - 4.0f/((float)(sinNumberOfPoints-1))));
                            sinInputData[i].emplace_back(std::acos(arg2));
                        }
                    }else{
                        sinInputData[i].emplace_back(sinInputData[i-1][0] + 2.0f*PI_FLOAT/((float)(sinNumberOfPoints-1)));
                    }

                    sinOutputData[i].emplace_back(std::sin(sinInputData[i][0]));
                    sinInputData[i-1][0] *= SQRT_3F/PI_FLOAT;
                }
                sinInputData[sinNumberOfPoints-2][0] *= SQRT_3F/PI_FLOAT;
                sinInputData[sinNumberOfPoints-1][0] *= SQRT_3F/PI_FLOAT;

                for(size_t i = 0; i < 4; i++){
                    int sinSeed = -1;
                    float lossTolerance = 0.05f;
                    TrainingResult res;
                    SaturatedNeuronReport rep;
                    res.loss = 0.5f;
                    xPartition.clear();
                    yPartition.clear();
                    neuralNetworkYPrediction.clear();

                    while(res.loss > lossTolerance){
                        sinSeed++;
                        std::cout << "\nTraining with activation " << i << " and seed " << sinSeed << "...\n";
                        NeuralNetwork nn(sinTrainingArchitecture, 0.1f, hiddenLayerActivations[i],Activation::Identity,sinSeed);
                        res = nn.train(sinInputData,sinOutputData,lossTolerance,50000);
                        std::cout << res.loss << std::endl;
                        if(res.loss <= lossTolerance){
                            rep = nn.checkForSaturatedNeurons(sinInputData);
                            for(int j = 0; j < 500; j++){
                                xPartition.emplace_back(PI_FLOAT*(2.0f*j/499.0f - 1.0f));
                                yPartition.emplace_back(std::sin(xPartition[j]));   
                                
                                nn.forwardPropagation({xPartition[j]*SQRT_3F/PI_FLOAT});
                                neuralNetworkYPrediction.emplace_back(nn.output()[0]);
                            }
                        }
                    }

                    writeFile.open(sinFileNames[i]);
                    if(!writeFile){
                        exit(10);
                    }else{
                        writeFile << "Loss,Epochs,How may saturated\n";
                        writeFile << res.loss << "," << res.epochs << "," << rep.howManySaturated << "\n";

                        writeFile << "\nx,Sin(x),NN prediction\n";
                        for(int i = 0; i < 500; i++){
                            writeFile << xPartition[i] << "," << yPartition[i] << "," << neuralNetworkYPrediction[i] << "\n";
                        }

                        writeFile.close();
                    }


                }

                //Test to see if the data generates correctly
                writeFile.open("ExperimentData/Sin_Training_Data/Sin_TrainingData.csv");

                if(!writeFile){
                    exit(10);
                }else{
                    writeFile << "x,y\n";

                    for(int i = 0; i < sinNumberOfPoints; i++){
                        writeFile << sinInputData[i][0] << "," << sinOutputData[i][0] << "\n";
                    }
                }



                break;
    }

    return 0;
}