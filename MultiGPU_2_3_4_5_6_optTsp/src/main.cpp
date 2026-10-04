#include <iostream>
#include "config/ConfigParamsCF.h"
#include "CalculateurKopt.h"
#include "NeuralNet.h"

#include "random_generator_cf.h"

using namespace std;


using namespace components;


int main(int argc, char *argv[])
{
    char* fileData;
    char* fileSolution;
    char* fileStats;
    char* fileConfig;

    /*
     * Lecture des fichiers d'entree
     */
    if (argc <= 1) {
        fileData = (char*) "input.svg";
        fileSolution = (char*) "output.svg";
        fileStats = (char*) "output.stats";
        fileConfig = (char*) "config.cfg";
    } else if (argc == 2) {
        fileData = argv[1];
        fileSolution = (char*) "output.svg";
        fileStats = (char*) "output.stats";
        fileConfig = (char*) "config.cfg";
    } else if (argc == 3) {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = (char*) "output.stats";
        fileConfig = (char*) "config.cfg";
    } else if (argc == 4) {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = argv[3];
        fileConfig = (char*) "config.cfg";
    } else {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = argv[3];
        fileConfig = argv[4];
    }
    cout << "RUN PARAMETERS: " << argv[0] << " " << fileData << " " << fileSolution << " " << fileStats << " " << fileConfig << endl;

    /*
     * Lecture des parametres
     */
    config::ConfigParamsCF* params = new config::ConfigParamsCF(fileConfig);
    params->readConfigParameters();


    CalculateurKopt::initialize(fileData, fileSolution, fileStats, params);
    CalculateurKopt::run();


    return 0;
}


