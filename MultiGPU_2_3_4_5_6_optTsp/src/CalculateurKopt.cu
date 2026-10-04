#include "CalculateurKopt.h"

#include "SolutionKOPT.h"
#include "random_generator_cf.h"

using namespace std;

typedef SolutionKOPT<2, 2> Solution;


Solution* sol = NULL;
Solution* sol1 = NULL;



void CalculateurKopt::initialize(char* fileData, char* fileSolution, char* fileStats, config::ConfigParamsCF* params)
{
    g_ConfigParameters = params;

    // Initialise le générateur de nombres aléatoires
    if (!g_ConfigParameters->useSeed) {
        g_ConfigParameters->seedValue = random_cf::aleat_get_time();
        cout << "SEED VALUE " << g_ConfigParameters->seedValue << endl;
    }
    random_cf::aleat_initialize(g_ConfigParameters->seedValue);

    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        cout << "INIT EVALUATE" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution();
        //        sol->initStatisticsFile();
        break;



    case RUN2OPT:
    {
        cout << "RUN2OPT INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution();
        sol->initStatisticsFile();

        sol->initStatisticsFile("2optimalTour");
        sol->evaluateInit();
        sol->writeStatisticsToFile("2optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1, cout);
    }
        break;

    case RUN3OPT:
    {
        cout << "RUN3OPT INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution();
        sol->initStatisticsFile();

        sol->initStatisticsFile("3optimalTour");
        sol->evaluateInit();
        sol->writeStatisticsToFile("3optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1, cout);
    }
        break;

    case RUN4OPT:
    {
        cout << "RUN4OPT INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution(g_ConfigParameters->functionModeChoice); //qiao add
        sol->initStatisticsFile();

        sol->initStatisticsFile("4optimalTour");
        sol->evaluateInit();
        sol->writeStatisticsToFile("4optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1, cout);
    }
        break;
    case RUN5OPT:
    {
        cout << "RUN5OPT INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution(g_ConfigParameters->functionModeChoice);
        sol->initStatisticsFile("5optimalTour");

        sol->evaluateInit();
        sol->writeStatisticsToFile("5optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1, cout);
    }
        break;

    case RUN6OPT:
    {
        cout << "RUN6OPT INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution(g_ConfigParameters->functionModeChoice);
        sol->initStatisticsFile("6optimalTour");

        sol->evaluateInit();
        sol->writeStatisticsToFile("6optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1, cout);
    }
        break;

    case RUN_KOPT:
    {
        cout << "Run Variable k-opt Searching INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution(g_ConfigParameters->functionModeChoice);
        sol->initStatisticsFile("koptimalTour");

        sol->evaluateInit();
        sol->writeStatisticsToFile("koptimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1, cout);
    }
        break;
    case RUNIterKOPT:
    {
        cout << "Run Iterative K-optimal INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution(g_ConfigParameters->functionModeChoice);
        sol->initStatisticsFile("iterKoptimalTour");

        sol->evaluateInit();
        sol->writeStatisticsToFile("iterKoptimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1, cout);
    }
        break;

    default:
        cout << "UNSUPPORTED FUNCTIONMODE=" << params->functionModeChoice << " !! " << endl;
        break;
    }

}//initialize

void CalculateurKopt::run()
{
    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        cout << "EVALUATE" << endl;

        sol->initEvaluate();
        sol->evaluate();

        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);

        sol->writeSolution();

        delete sol;
        break;

    case RUN2OPT:
        cout << "RUN2OPT RUN" << endl;

        sol->run2opt("2optimalTour"); //rocki 2-opt with massive moves
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile("2optimalTour");
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
        sol->writeSolution("2optimalTour");
        delete sol;
        break;

    case RUN3OPT:
        cout << "RUN3OPT RUN" << endl;

        sol->run3opt("3optimalTour"); //rocki 3-opt with massive moves
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile("3optimalTour");
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
        sol->writeSolution("3optimalTour");
        delete sol1;
        break;

    case RUN4OPT:
        cout << "RUN4OPT RUN" << endl;

        sol->run4opt("4optimalTour"); //rocki 2-opt with massive moves
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile("4optimalTour");
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
        sol->writeSolution("4optimalTour");
        delete sol;
        break;

    case RUN5OPT:
        cout << "RUN5OPT RUN" << endl;

//        sol->run5opt_2("5optimalTour"); //rocki 2-opt with massive moves

        sol->run5opt("5optimalTour"); //rocki 2-opt with massive moves
//        sol->run5opt_single("5optimalTour"); //rocki 2-opt with massive moves
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile("5optimalTour");
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
        sol->writeSolution("5optimalTour");
        delete sol;
        break;

    case RUN6OPT:
        cout << "RUN6OPT RUN" << endl;

        sol->run6opt("6optimalTour"); //rocki 2-opt with massive moves
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile("6optimalTour");
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
        sol->writeSolution("6optimalTour");
        delete sol;
        break;

    case RUN_KOPT:
        cout << "RUN variable kOPT RUN" << endl;

        sol->runVariableKopt("koptimalTour"); //qiao run gpu variable k-opt kernels
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile("koptimalTour");
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
        sol->writeSolution("koptimalTour");
        delete sol;
        break;

    case RUNIterKOPT:
        cout << "RUN iterative k-optimal RUN" << endl;

        sol->runGpuIterativeKoptimal("iterKoptimalTour"); //qiao run gpu iterative k-optimal from 2 to 6
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile("iterKoptimalTour");
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
        sol->writeSolution("iterKoptimalTour");
        delete sol;
        break;

    default:
        cout << "UNSUPPORTED FUNCTIONMODE=" << g_ConfigParameters->functionModeChoice << " !! " << endl;
        break;
    }

}//run

bool CalculateurKopt::activate()
{
    bool ret = true;
    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        cout << "EVALUATE" << endl;

        sol->initEvaluate();
        sol->evaluate();

        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);

        sol->writeSolution();

        delete sol;
        break;


    case RUN2OPT:
        cout << "CONSTRUCTION" << endl;

        sol->constructSolutionSeq();
        // Après construction
        sol->evaluate();
        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution();
        delete sol;
        break;

    case RUN3OPT:
        cout << "RUN" << endl;

        sol1->run();
        // Après run
        sol1->evaluate();
        //sol->writeStatisticsToFile(-1);
        //sol->setIdentical(sol1);
        sol1->writeStatisticsToFile();
        sol1->writeHeaderStatistics(cout);
        sol1->writeStatistics(cout);
        sol1->writeSolution();
        delete sol;
        break;

    case RUN4OPT:
        cout << "RUN" << endl;

        sol->run();
        // Après run
        sol->evaluate();
        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution();
        delete sol;
        break;

    case RUN5OPT:
        cout << "RUN" << endl;

        sol->run();
        // Après run
        sol->evaluate();
        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution();
        delete sol;
        break;

    case RUN6OPT:
        cout << "RUN" << endl;

        sol->run();
        // Après run
        sol->evaluate();
        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution();
        delete sol;
        break;

    default:
        cout << "UNSUPPORTED FUNCTIONMODE=" << g_ConfigParameters->functionModeChoice << " !! " << endl;
        break;
    }

    return ret;

}//activate

#ifndef SEPARATE_COMPILATION
#include "SolutionKOPT.cu"
#include "SolutionKOPTRW.cu"
#include "SolutionKOPTOperators.cu"
#endif
