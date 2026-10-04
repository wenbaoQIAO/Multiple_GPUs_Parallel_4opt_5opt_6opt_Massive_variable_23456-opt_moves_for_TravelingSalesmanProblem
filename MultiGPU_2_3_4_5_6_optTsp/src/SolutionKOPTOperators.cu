#include "config/ConfigParamsCF.h"
#include "random_generator_cf.h"
#include "SolutionKOPT.h"
#include "distance_functors.h"
#include "CalculateurKopt.h"

/** Operateurs de changement de SolutionKOPT courante.
 *
 */
#define FULL_GPU 1
#define FULL_GPU_FIND_MIN1 1
#define FULL_GPU_FIND_MIN2 1
#define FULL_GPU_CGU 1
#define FULL_GPU_FLATTENING 1
#define MAXNUMRUNS 2
#define CPUEXECUTE 1
#define SINGLGPU 0
#define MULTIGPUMODE2 0 //mode1 conflict with mode2, should use together with MULTIGPU
#define MULTIGPUMODE1 1 //mode1 conflict with mode2, should use together with MULTIGPU
#define MULTIGPU 1//should use together with GPUDEVICE0
#define GPUDEVICE0 0 //should use together with MULTIGPU
#define NUMRUNSLIMIT 2000 //2000
#define ONERUNTEST 1

#define EMST_DETECT_CYCLE  0
#define EMST_FIND_MIN_PAIR_LIST 1// Distributed broadcast or distributed linked list

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::initConstruct()
{
}//initConstruct

/** Construction Sequentielle
 */
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::constructSolutionSeq()
{
    cout << "CONSTRUCTION SEQUENTIELLE ..." << endl;
    int nNodes = mr_links_cpu.adaptiveMap.getWidth();

    int iteration = 0;// maximum iterations

    int radiusSearchCells = 0;
    g_ConfigParameters->readConfigParameter("test_2opt", "radiusSearchCells", radiusSearchCells);

    float gpuTimingKernels = 0;
    float mstTotalTimeFrequen = 0;

    cout << "CONSTRUCTION done" << endl;
}

/*!
 * \return vrai si l'operateur est applique selon choix aleatoire,
 *  faux si l'operateur n'est pas applique
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::operator_1() {
    bool ret = true;


    global_objectif = computeObjectif();

    return ret;
}//operator_1

/*!
 * \return vrai si l'operateur est applique selon choix aleatoire,
 *  faux si l'operateur n'est pas applique
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::operator_2() {
    bool noUsed = true;

    return noUsed;
}//operator_1


template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::applyOperator(int i)
{
    bool ret = false;
    switch (i)
    {
    case 0:
        break ;
    case 1:
        break ;
    case 2:
        break ;
    case 3:
        break ;
    case 4:
        break ;
    case 5:
        break ;
    case 6:
        break ;
    case 7:
        break ;
    }
    ret = operator_1();
    return ret;
}

template<std::size_t DimP, std::size_t DimCM>
int SolutionKOPT<DimP, DimCM>::nbrOperators() const
{
    return g_ConfigParameters->probaOperators.size();
}

//! \brief Run et activate
//!
//wb.Q 2024 void run
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run() {

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdbOptEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer2optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);


    unsigned long maxChecks = nCity*(nCity - 1) / 2; // total number of checks for 2-opt
    unsigned int iter = maxChecks / (BLOCKSIZE * GRIDSIZE);

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {

    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();


    //! count time gpu total
    timeGpuTotal += timeGpuH2D + timeGpuD2H + timeGpuKernel;


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();


}// end run



//! \brief Run et activate
//!
//wb.Q 202206 implement GPU parallel 2-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run2opt(string fileName) {

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdbOptEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer2optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer2optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer2optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;


    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);


    unsigned long maxChecks = nCity*(nCity - 1) / 2; // total number of checks for 2-opt
    unsigned int iter = maxChecks / (BLOCKSIZE * GRIDSIZE);

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < 2000  && percentageImprove > 0 )
        {
            activateRocki2opt(numRuns, nCity, maxChecks, iter, optimum,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun,outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                              linkCoordTourCpu,linkCoordTourGpu);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }
    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel 2-opt and massive variable 2-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki2opt(int& numRuns, int nCity, double maxChecks2opt, double iter,float optimum,
                                                  int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                  float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                  float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                  float& timeRefresh, float& timeSelect, float& timeExecute,
                                                  float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun,
                                                  ofstream & outfileSearchTimePerRunRun,
                                                  float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu)
{
    cout << endl << "Enter 2-opt iteration=: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    double timeTotallOneRun = 0;
    float elapsedTimeOpt = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(0, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean md_links_firstPara before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;

    cudaSetDevice(0);
    // time for GPU memcp HD
    float elapsedTimeOptHD = 0;
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);

    // copy tour ordering to gpu, clean gpu network links

    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);

    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTimeOptHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D tour order : " <<  elapsedTimeOptHD << endl;

    cudaSetDevice(0);

    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change


    //qiao only for test
    cout << "Warning: maxChecks2opt= " << maxChecks2opt << endl;

    // cuda timer
    double time = 46;
    double *d_time;


    //divide and conquer
    double maxChecksoptDivide = 1.27719e+11;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    double iterDivide = (double)maxChecksoptDivide /(double) ((double)BLOCKSIZE * (double) GRIDSIZE);
    if(maxChecks2opt < maxChecksoptDivide)
        iterDivide = 1;
    double maxStride = (double) maxChecks2opt /  (double)maxChecksoptDivide;
    if(maxStride < 1)
        maxStride = 0;
    for(double iStride = 0; iStride < maxStride+1; iStride++ )
    {

        cudaSetDevice(0);
        K_oneThreadOne2opt_Rocki_iterStride(md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecksoptDivide, iterDivide, iStride);

        //        cudaDeviceSynchronize();

        cout << "Inner one time " << iStride << endl << endl;
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeOpt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // find the maximum gpu time for a parallel 2-opt run
    cout << "Time:: GPU side one 2-opt runtime : " << elapsedTimeOpt << endl;

    if(maxtimeGpuOptSearch < elapsedTimeOpt){
        maxtimeGpuOptSearch = elapsedTimeOpt;
    }


    //! sequentially select non-interacted 2-exchanges
    float elapsedTimeOpt_DH2 = 0;
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);

    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);// node3

    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTimeOpt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H 2-opt candidates: " << elapsedTimeOpt_DH2 << endl;

    //! clean for mark non-interacted 2-opt
    md_links_firstPara.activeMap.resetValue(0); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(0); // for nodes in stackB

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 2-exchanges
    md_links_firstPara.selectNonIteracted2ExchangeRocki(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 2-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTimeOptHD2 = 0;
    float elapsedTimeOptHD3 = 0;
    float elapsedTimeOpt_DH = 0;
    float elapsedTimeOpt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    md_links_firstPara.executeNonInteract2optOnlyNode3(numOptimizedOneRun);

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 2-opt: " << timeCpuExecuteNonItera << endl;


#else

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks : " <<  elapsedTimeOptHD3 << endl;


    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTimeOptHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTimeOpt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTimeOpt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTimeOpt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTimeOpt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTimeOpt_DH << endl;

#endif


    timeGpuExecute =  elapsedTimeOptHD2 + elapsedTimeOpt_execute + elapsedTimeOpt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;

    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 2-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTimeOptHD + elapsedTimeOptHD2;
        timeGpuD2H += elapsedTimeOpt_DH + elapsedTimeOpt_DH2;
        timeGpuKernel += elapsedTimeOpt + elapsedTimeOpt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun;
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);

        //! count time gpu total
        timeGpuTotal = timeGpuH2D + timeGpuD2H + timeGpuKernel;

        //        timeTotallOneRun = elapsedTimeOptHD + elapsedTimeOptHD2 + elapsedTimeOpt_DH + elapsedTimeOpt_DH2 + elapsedTimeOpt + elapsedTimeOpt_execute
        //                + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;


        outfileTimePerRunRun << timeCpuKey + timeGpuTotal << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << elapsedTimeOpt << " " << endl;

        traceTSP.timeObtainKoptimal = timeCpuKey + timeGpuTotal;


        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour


        // outfile
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("2optimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey + timeGpuTotal << ", pdb: " << pdbOneRun << ", searchTime: " << elapsedTimeOpt << endl;

        outfileKoptimalTimePerRunRun.close();


    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif

    return ret;
}//end 2opt



//! \brief Run qiao 2024 run 4-opt
//!
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run4opt(string fileName) {

    cout << "Begin run 4-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdbOptEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer4optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer4optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer4optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;



    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);

    cudaSetDevice(0);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);

    //! prepare the pre-ordered link + coordinates on device 1
    cudaSetDevice(1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu_1;
    linkCoordTourGpu_1.gpuResize(nCity,1);

    //copy result from device1
    Grid<float> densityMap_1;//level 1 density map
    densityMap_1.resize(nCity,1);
    Grid<unsigned long long> optCandidateMap_1;
    optCandidateMap_1.resize(nCity,1);


#if GPUDEVICE0
    cudaSetDevice(0); //here can not add this line
#endif

    double temp = (double)nCity /(double) 2;
    double maxChecks2opt = temp *(nCity - 1) ; // N

    double temptemp =  (double)maxChecks2opt/ (double)2;
    double maxChecks4opt = temptemp*(maxChecks2opt - 1);
    double iter4opt = (double)maxChecks4opt /(double) ((double)BLOCKSIZE * (double)GRIDSIZE);
    if(iter4opt < 1)
        iter4opt = 1;

    cout << "Check maxChecks4opt = " << maxChecks4opt << ", iter4opt = " << iter4opt << endl;


    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 4-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateRocki4opt(numRuns, nCity, maxChecks2opt, maxChecks4opt, iter4opt, optimum,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun,outfilePdbPerRunRun,outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                              linkCoordTourCpu,linkCoordTourGpu,linkCoordTourGpu_1,densityMap_1,optCandidateMap_1);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }
    }// end activateRocki

    //! free gpu memory
    cudaSetDevice(0);
    linkCoordTourGpu.gpuFreeMem();
    cudaStreamDestroy(stream0);

    cudaSetDevice(1);
    linkCoordTourGpu_1.gpuFreeMem();
    cudaStreamDestroy(stream1);


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki4opt(int& numRuns, int nCity,double maxChecks2opt,
                                                  double  maxChecks4opt,
                                                  double iter,float optimum,
                                                  int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                  float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                  float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                  float& timeRefresh, float& timeSelect, float& timeExecute,
                                                  float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun, ofstream &outfilePdbPerRunRun,
                                                  ofstream &outfileSearchTimePerRunRun,
                                                  float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu_1,
                                                  Grid<float>& densityMap_1, Grid<unsigned long long>& optCandidateMap_1 )
{
    cout << endl << "****>>>>Enter 4-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    double timeTotallOneRun = 0;
    float elapsedTimeOpt = 0;
    float elapsedTimeOptCPUCount = 0;

    float elapsedTimeOpt_d1 = 0;
    double gpuSearchingTime0 = 0;
    double gpuSearchingTime1 = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;


    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(0, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt


    densityMap_1.resetValue(initialPrepareValue);
    optCandidateMap_1.resetValue(initialPrepareValue);

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    // time for GPU memcp HD
    float elapsedTimeOptHD = 0;
#if GPUDEVICE0
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);
#endif

    cudaSetDevice(0);

    // copy tour ordering to gpu, clean gpu network links
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order


#if MULTIGPU

    cudaSetDevice(1);
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu_1);// refresh doubly linked tour order
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu_1.grayValueMap);// refresh tsp order gpu side
#endif

#if GPUDEVICE0
    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTimeOptHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D grayValueMap : " <<  elapsedTimeOptHD << endl;
#endif

    cudaSetDevice(0);
    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt

    //    //qiao only for test
    //    cout <<"device 0 " << endl;
    //    Grid<GLfloat> minRadiusMap_0;
    //    minRadiusMap_0.resize(nCity,1);
    //    minRadiusMap_0.gpuCopyDeviceToHost(md_links_gpu.minRadiusMap);
    //    for(int i  = 0; i <10; i++)
    //        cout << minRadiusMap_0[0][i] ;
    //    cout << endl;

#if MULTIGPU
    cudaSetDevice(1);
    md_links_gpu_1.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu_1.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu_1.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt

    //    //qiao only for test
    //    cout <<"device 1 " << endl;
    //    Grid<GLfloat> minRadiusMap_1;
    //    minRadiusMap_1.resize(nCity,1);
    //    minRadiusMap_1.gpuCopyDeviceToHost(md_links_gpu_1.minRadiusMap);
    //    for(int i  = 0; i <512; i++)
    //        cout << minRadiusMap_1[0][i] ;
    //    cout << endl;

#endif



    // cuda timer
    double time = 46;
    double *d_time;


    //qiao only for test
    cout << "Warning: maxChecks4opt= " << maxChecks4opt << ", Warning: maxChecks2opt= " << maxChecks2opt << endl;


    //divide and conquer
    double maxChecks4optDivide = 1.27719e+12; //1073741824; //is too slow
    double packSize = (double)BLOCKSIZE * (double)GRIDSIZE;


    cudaDeviceSynchronize();
    //    cudaEventSynchronize();
    cudaStreamSynchronize(0);


    //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
    double iterDivide = (double)maxChecks4optDivide /(double) (packSize);
    if(maxChecks4opt < packSize)
        iterDivide = 1;
    double maxStride = (double) maxChecks4opt /  (double)maxChecks4optDivide;
    if(maxStride < 1)
        maxStride = 0;

    cout << "Changed maxChecks4optDivide = " << maxChecks4optDivide << ", iterDivide = " << iterDivide << ", maxStride= " << maxStride << endl;

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //qiao here does not run correctly
    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //                                 maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);



    if(maxStride == 0)
    {
        //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
        cudaSetDevice(0);
        //#if GPUDEVICE0
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        //#endif
        K_oneThreadOne4opt_qiao_iterStride(stream0, md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, 0);
        //#if GPUDEVICE0
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeOpt, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // find the maximum gpu time for a parallel 5-opt run
        cout << "Time:: GPU side one 4-opt run : " << elapsedTimeOpt << endl;
        gpuSearchingTime0 += elapsedTimeOpt;

        //#endif
        cout << "Enter for stride 0 " << endl;

    }
    else
    {

#if SINGLGPU

        for(int iStride = 0; iStride < maxStride; iStride = iStride +1 )
        {

            cudaSetDevice(0);

            K_oneThreadOne4opt_qiao_iterStride(stream0, md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);
            //            K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream0 >>> (md_links_gpu, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);// global
        }

#endif

#if MULTIGPUMODE1
        for(int iStride = 0; iStride < maxStride; iStride = iStride + 2 )
        {
            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
            cout << "Enter for stride d0 "<< iStride << endl;
            cudaSetDevice(0);

            //faster by using the same one stream
            K_oneThreadOne4opt_qiao_iterStride(stream0, md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);
            //            K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream0 >>> (md_links_gpu, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);// global

            cout << "Enter for stride d1 "<< iStride+1 << endl;
            cudaSetDevice(1);
            //faster by using the same one stream
            K_oneThreadOne4opt_qiao_iterStride(stream1, md_links_gpu_1, linkCoordTourGpu_1,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride+1);
            //            K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream1 >>> (md_links_gpu_1, linkCoordTourGpu_1,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride+1);// global


        }

#endif

#if MULTIGPUMODE2
        int taskDivision =  (int)((double)maxStride / (double)4); //5 for 7146 only 1.8s to finish one iteration

        for(int iStride = 0; iStride < taskDivision; iStride = iStride + 1 )
        {
            cout << "Enter for stride d1 "<< iStride << endl;
            cudaSetDevice(1);

            //faster by using the same one stream
            K_oneThreadOne4opt_qiao_iterStride(stream1, md_links_gpu_1, linkCoordTourGpu_1,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);

            //            K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream1 >>> (md_links_gpu_1, linkCoordTourGpu_1,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);// global
        }

        for(int iStride = taskDivision ; iStride < maxStride +1 ; iStride = iStride + 1 )
        {
            cout << "Enter for stride d0 "<< iStride << endl;

            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
            cudaSetDevice(0);

            //faster by using the same one stream
            K_oneThreadOne4opt_qiao_iterStride(stream0, md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);

            //            K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream0 >>> (md_links_gpu, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);// global
        }

#endif
    }

    cudaStreamSynchronize(0);
    cudaStreamSynchronize(nullptr);
    cudaStreamSynchronize(stream0);

    cudaStreamSynchronize(stream1);



    cudaDeviceSynchronize();
    cout << "End GPU parallel 4-opt search qiao: " << endl;


    // end time cpu
    elapsedTimeOptCPUCount = GetCounter(pcFreq, CounterStart);
    cout << "Time:: multi GPU search time on CPU side single or two qiao : " << elapsedTimeOptCPUCount << endl;


    if(maxtimeGpuOptSearch < elapsedTimeOptCPUCount){
        maxtimeGpuOptSearch = elapsedTimeOptCPUCount;
    }




    //! sequentially select non-interacted 4-exchanges
    float elapsedTimeOpt_DH2 = 0;
#if GPUDEVICE0
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);
#endif

    cudaDeviceSynchronize();
    cout << "Here ending 4-opt search and begin result transfer " << endl;


    cudaSetDevice(0);
    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);//
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates


#if MULTIGPU
    cudaSetDevice(1);
    densityMap_1.gpuCopyDeviceToHost(md_links_gpu_1.densityMap);
    optCandidateMap_1.gpuCopyDeviceToHost(md_links_gpu_1.optCandidateMap);//opt candidates
#endif
    cudaDeviceSynchronize();

#if GPUDEVICE0

    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTimeOpt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H " << elapsedTimeOpt_DH2 << endl;
#endif



    //! clean for mark non-interacted 23456-opt
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB



    //    // qiao only for test
    //    int numCandidate = 0;
    //    int numCandidate_1 = 0;
    //    int numSameCandidate = 0;
    //    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    //    {
    //        if(md_links_firstPara.optCandidateMap[0][i] > 0 )
    //        {
    //            numCandidate += 1;
    //            //            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
    //        }

    //        if(optCandidateMap_1[0][i] != initialPrepareValueLL && optCandidateMap_1[0][i] == md_links_firstPara.optCandidateMap[0][i])
    //        {
    //            numSameCandidate += 1;
    //            //            cout << " find same candidate " << md_links_firstPara.grayValueMap[0][i] << endl;

    //        }
    //        else if(optCandidateMap_1[0][i] > 0)
    //        {
    //            numCandidate_1 += 1;
    //            //            cout << " candidate order device 1 " << md_links_firstPara.grayValueMap[0][i] << endl;

    //        }
    //    }
    //    cout << "After one: device1 search num of candidates: " << numCandidate << ", device1 found candidates: " << numCandidate_1 << endl;



    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

#if MULTIGPU
    //merge results from different GPU cards  merge results from device1 to md_links_firstPara.densityMap and md_links_firstPara.optCandidateMap
    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    {
        if(optCandidateMap_1[0][i] > 0 && md_links_firstPara.optCandidateMap[0][i] == initialPrepareValueLL)
        {
            md_links_firstPara.optCandidateMap[0][i] = optCandidateMap_1[0][i];
            md_links_firstPara.densityMap[0][i] = densityMap_1[0][i];
        }

    }
#endif

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 4-opt: " << timeCpuSelectNonItera << endl;




    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTimeOptHD2 = 0;
    float elapsedTimeOptHD3 = 0;
    float elapsedTimeOpt_DH = 0;
    float elapsedTimeOpt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);
    // md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 4-opt: " << timeCpuExecuteNonItera << endl;


#else

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks: " <<  elapsedTimeOptHD3 << endl;


    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTimeOptHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTimeOpt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTimeOpt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTimeOpt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTimeOpt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTimeOpt_DH << endl;

#endif


    timeGpuExecute =  elapsedTimeOptHD2 + elapsedTimeOpt_execute + elapsedTimeOpt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    //    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 4-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTimeOptHD + elapsedTimeOptHD2;
        timeGpuD2H += elapsedTimeOpt_DH + elapsedTimeOpt_DH2;
        timeGpuKernel += elapsedTimeOptCPUCount + elapsedTimeOpt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);

        //! count time gpu total
        timeGpuTotal = timeGpuH2D + timeGpuD2H + timeGpuKernel;

        outfileTimePerRunRun << timeCpuKey + timeGpuTotal << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << elapsedTimeOptCPUCount << endl;// << ", d1: " << elapsedTimeOpt_d1 << ", cpu: " << elapsedTimeOptCPUCount << endl;

        traceTSP.timeObtainKoptimal =  timeCpuKey + timeGpuTotal;


        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);

    }
    else{
        numRuns -= 1; // the last run does not optimized the tour

        // outfile
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("4optimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey + timeGpuTotal << ", pdb: " << pdbOneRun << ", searchTime: " << elapsedTimeOptCPUCount << endl;

        outfileKoptimalTimePerRunRun.close();

    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif


    return ret;
}//end 4opt



// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM> // former version than activateRocki4opt
bool SolutionKOPT<DimP, DimCM>::activateRocki4optMultiGPU(int& numRuns, int nCity,double maxChecks2opt,
                                                          double  maxChecks4opt,
                                                          double iter,float optimum,
                                                          int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                          float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                          float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                          float& timeRefresh, float& timeSelect, float& timeExecute,
                                                          float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun, ofstream &outfilePdbPerRunRun, ofstream & outfileSearchTimePerRunRun,
                                                          float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                          Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu_1,
                                                          Grid<float>& densityMap_1, Grid<unsigned long long>& optCandidateMap_1 )
{
    cout << endl << "****>>>>Enter 4-opt activate function: " << numRuns << endl;


    bool ret = true;
    numRuns ++;

    double timeTotallOneRun = 0;
    float elapsedTimeOpt = 0;


    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;


    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt


    densityMap_1.resetValue(initialPrepareValue);
    optCandidateMap_1.resetValue(initialPrepareValue);

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);


    // time for GPU memcp HD
    float elapsedTimeOptHD = 0;

#if GPUDEVICE0
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);
#endif

    cudaSetDevice(0);
    // copy tour ordering to gpu, clean gpu network links
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order

#if MULTIGPU

    cudaSetDevice(1);
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu_1);// refresh doubly linked tour order
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu_1.grayValueMap);// refresh tsp order gpu side
#endif

#if GPUDEVICE0
    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTimeOptHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D grayValueMap : " <<  elapsedTimeOptHD << endl;
#endif

    cudaSetDevice(0);
    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt

#if MULTIGPU
    cudaSetDevice(1);
    md_links_gpu_1.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu_1.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu_1.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt

    //    //qiao only for test
    //    cout <<"device 1 " << endl;
    //    Grid<GLfloat> minRadiusMap_1;
    //    minRadiusMap_1.resize(nCity,1);
    //    minRadiusMap_1.gpuCopyDeviceToHost(md_links_gpu_1.minRadiusMap);
    //    for(int i  = 0; i <512; i++)
    //        cout << minRadiusMap_1[0][i] ;
    //    cout << endl;

#endif

    // cuda timer
    double time = 46;
    double *d_time;


    //qiao only for test
    cout << "Warning: maxChecks4opt= " << maxChecks4opt << ", Warning: maxChecks2opt= " << maxChecks2opt << endl;


    //divide and conquer
    double maxChecks4optDivide = 1.27719e+11;
    double packSize = (double)BLOCKSIZE * (double)GRIDSIZE;
    //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
    double iterDivide = (double)maxChecks4optDivide /(double) (packSize);
    if(maxChecks4opt < packSize)
        iterDivide = 1;
    double maxStride = (double) maxChecks4opt /  (double)maxChecks4optDivide;
    if(maxStride < 1)
        maxStride = 0;

    cout << "Changed maxChecks4optDivide = " << maxChecks4optDivide << ", iterDivide = " << iterDivide << ", maxStride= " << maxStride << endl;

#if GPUDEVICE0
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif

    cudaDeviceSynchronize();
    //    cudaEventSynchronize();
    cudaStreamSynchronize(0);

    if(maxStride == 0)
    {
        cudaSetDevice(0);
        K_oneThreadOne4opt_qiao_iterStride(stream0, md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, 0);
    }
    else {
#if MULTIGPU
        for(int iStride = 0; iStride < maxStride; iStride = iStride + 2 )
#else

        for(int iStride = 0; iStride < maxStride+1; iStride++ )
#endif

        {

            cout << "Enter for stride " << endl;

            cudaSetDevice(0);
            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
            K_oneThreadOne4opt_qiao_iterStride(stream0, md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);

#if MULTIGPU
            cudaSetDevice(1);
            K_oneThreadOne4opt_qiao_iterStride(stream1, md_links_gpu_1, linkCoordTourGpu_1, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride+1);

#endif
            //            cout << "Inner one time " << iStride << endl << endl;

            //            cout << "Counting loop iStride = " << iStride << endl;

            //            if(iStride % 4 == 0){
            //                cudaDeviceSynchronize();
            //                cout << "Done 4 kernels " << endl << endl;
            //            }
        }
    }

    cudaDeviceSynchronize();

#if GPUDEVICE0
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeOpt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // find the maximum gpu time for a parallel 2-opt run
    cout << "Time:: GPU side one 4-opt runtime : " << elapsedTimeOpt << endl;

    if(maxtimeGpuOptSearch < elapsedTimeOpt){
        maxtimeGpuOptSearch = elapsedTimeOpt;
    }
#endif

    //! sequentially select non-interacted 4-exchanges
    float elapsedTimeOpt_DH2 = 0;
#if GPUDEVICE0
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);
#endif

    cudaDeviceSynchronize();
    cout << "Here ending 4-opt search and begin result transfer " << endl;

    cudaSetDevice(0);
    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);//
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates

#if MULTIGPU
    cudaSetDevice(1);
    densityMap_1.gpuCopyDeviceToHost(md_links_gpu_1.densityMap);
    optCandidateMap_1.gpuCopyDeviceToHost(md_links_gpu_1.optCandidateMap);//opt candidates
#endif
    cudaDeviceSynchronize();

#if GPUDEVICE0
    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTimeOpt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H " << elapsedTimeOpt_DH2 << endl;
#endif

    // end time cpu
    elapsedTimeOpt = GetCounter(pcFreq, CounterStart);
    cout << "Time:: multi GPU search time on CPU side: " << elapsedTimeOpt << endl;




    //    //qiao only for test
    //    int numCandidate = 0;
    //    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    //    {
    //        if(md_links_firstPara.optCandidateMap[0][i] > 0)
    //        {
    //            numCandidate += 1;
    //            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
    //        }

    //    }
    //    cout << "After one GPU search num of candidates: " << numCandidate << endl;



    //! clean for mark non-interacted 23456-opt
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

#if MULTIGPU
    //merge results from different GPU cards  merge results from device1 to md_links_firstPara.densityMap and md_links_firstPara.optCandidateMap
    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    {
        if(optCandidateMap_1[0][i] > 0 && md_links_firstPara.optCandidateMap[0][i] == initialPrepareValueLL)
        {
            md_links_firstPara.optCandidateMap[0][i] = optCandidateMap_1[0][i];
            md_links_firstPara.densityMap[0][i] = densityMap_1[0][i];
        }

    }
#endif


    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 4-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTimeOptHD2 = 0;
    float elapsedTimeOptHD3 = 0;
    float elapsedTimeOpt_DH = 0;
    float elapsedTimeOpt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);
    // md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 4-opt: " << timeCpuExecuteNonItera << endl;


#else

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks: " <<  elapsedTimeOptHD3 << endl;


    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTimeOptHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTimeOpt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTimeOpt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTimeOpt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTimeOpt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTimeOpt_DH << endl;

#endif


    timeGpuExecute =  elapsedTimeOptHD2 + elapsedTimeOpt_execute + elapsedTimeOpt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 4-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTimeOptHD + elapsedTimeOptHD2;
        timeGpuD2H += elapsedTimeOpt_DH + elapsedTimeOpt_DH2;
        timeGpuKernel += elapsedTimeOpt + elapsedTimeOpt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);

        //! count time gpu total
        timeGpuTotal = timeGpuH2D + timeGpuD2H + timeGpuKernel;

        //        timeTotallOneRun = elapsedTimeOptHD + elapsedTimeOptHD2 + elapsedTimeOpt_DH +
        //                                     elapsedTimeOpt_DH2 + elapsedTimeOpt + elapsedTimeOpt_execute
        //                + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;

        outfileTimePerRunRun << timeCpuKey + timeGpuTotal << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << elapsedTimeOpt << endl;

        traceTSP.timeObtainKoptimal =  timeCpuKey + timeGpuTotal;


        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);

    }
    else{
        numRuns -= 1; // the last run does not optimized the tour
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif


    return ret;
}//end 4opt





//! \brief Run et activate
//!
//wb.Q 202408 implement 5-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run5opt(string fileName) {

    cout << "Begin run 5-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdbOptEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer5optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);


    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer5optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer5optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;


    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);

    cudaSetDevice(0);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);

    //! prepare the pre-ordered link + coordinates on device 1
    cudaSetDevice(1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu_1;
    linkCoordTourGpu_1.gpuResize(nCity,1);

    //copy result from device1
    Grid<float> densityMap_1;//level 1 density map
    densityMap_1.resize(nCity,1);
    Grid<unsigned long long> optCandidateMap_1;
    optCandidateMap_1.resize(nCity,1);

#if GPUDEVICE0
    cudaSetDevice(0); //here can not add this line
#endif

    double maxChecks2opt = nCity*(nCity - 1) / 2; // total number of checks for 2-opt
    unsigned int iter = maxChecks2opt / ((double)BLOCKSIZE * (double)GRIDSIZE);

    double maxChecks4opt = maxChecks2opt*(maxChecks2opt - 1) / 2;
    unsigned int iter4opt = maxChecks4opt / ((double)BLOCKSIZE * (double)GRIDSIZE);

    cout << " maxChecks4opt= " << maxChecks4opt << " maxChecks2opt= " << maxChecks2opt << endl;

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateRocki5opt(numRuns, nCity, maxChecks4opt, maxChecks2opt, iter4opt, optimum,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun,outfilePdbPerRunRun,outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                              linkCoordTourCpu,linkCoordTourGpu,linkCoordTourGpu_1,densityMap_1,optCandidateMap_1 );

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }
    }// end activateRocki

    //! free gpu memory
    cudaSetDevice(0);
    linkCoordTourGpu.gpuFreeMem();
    cudaStreamDestroy(stream0);

    cudaSetDevice(1);
    linkCoordTourGpu_1.gpuFreeMem();
    cudaStreamDestroy(stream1);


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();

}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki5opt(int& numRuns, int nCity, double maxChecks4opt,
                                                  double maxChecks2opt,
                                                  unsigned int iter,float optimum,
                                                  int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                  float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                  float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                  float& timeRefresh, float& timeSelect, float& timeExecute,
                                                  float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun,
                                                  ofstream & outfileSearchTimePerRunRun,
                                                  float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu_1,
                                                  Grid<float>& densityMap_1, Grid<unsigned long long>& optCandidateMap_1)
{
    cout << endl << "****>>>>Enter 5-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    double timeTotallOneRun = 0;
    float elapsedTimeOpt = 0;

    float elapsedTimeOpt_d1 = 0;
    double gpuSearchingTime0 = 0;
    double gpuSearchingTime1 = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(0, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt

    densityMap_1.resetValue(initialPrepareValue);
    optCandidateMap_1.resetValue(initialPrepareValue);

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps
    //! reserver, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;



    float elapsedTimeOptHD = 0;
#if GPUDEVICE0
    //    // time for GPU memcp HD
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);
#endif
    // copy tour ordering to gpu, clean gpu network links

    cudaSetDevice(0);
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order

#if MULTIGPU

    cudaSetDevice(1);
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu_1);// refresh doubly linked tour order
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu_1.grayValueMap);// refresh tsp order gpu side
#endif


#if GPUDEVICE0
    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTimeOptHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D grayValueMap : " <<  elapsedTimeOptHD << endl;
#endif

    cudaSetDevice(0);
    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt


    //    //qiao only for test
    //    cout <<"device 0 " << endl;
    //    Grid<GLfloat> minRadiusMap_0;
    //    minRadiusMap_0.resize(nCity,1);
    //    minRadiusMap_0.gpuCopyDeviceToHost(md_links_gpu.minRadiusMap);
    //    for(int i  = 0; i <10; i++)
    //        cout << minRadiusMap_0[0][i] ;
    //    cout << endl;

#if MULTIGPU
    cudaSetDevice(1);
    md_links_gpu_1.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu_1.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu_1.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt

    //    //qiao only for test
    //    cout <<"device 1 " << endl;
    //    Grid<GLfloat> minRadiusMap_1;
    //    minRadiusMap_1.resize(nCity,1);
    //    minRadiusMap_1.gpuCopyDeviceToHost(md_links_gpu_1.minRadiusMap);
    //    for(int i  = 0; i <512; i++)
    //        cout << minRadiusMap_1[0][i] ;
    //    cout << endl;

#endif

    //    cudaSetDevice(0);

    // cuda timer
    double time = 46;
    double *d_time;

    //qiao only for test
    //divide and conquer
    double maxChecks4optDivide = 1.27719e+12; //1073741824; //is too slow
    double packSize = (double)BLOCKSIZE * (double)GRIDSIZE;


    cudaDeviceSynchronize();
    //    cudaEventSynchronize();
    cudaStreamSynchronize(0);

    //! WB.Q one 4 edges loop n-j time in the kernel
    double iterDivide = (double)maxChecks4optDivide /(double) (packSize);
    if(maxChecks4opt < packSize)
        iterDivide = 1;
    double maxStride = (double) maxChecks4opt /  (double)maxChecks4optDivide;
    if(maxStride < 1)
        maxStride = 0;

    cout << "maxChecks4opt= " << maxChecks4opt << ", packSize= " << packSize << ", Changed maxChecks4optDivide = " << maxChecks4optDivide << ", iterDivide = "
         << iterDivide << ", maxStride= " << maxStride << endl;

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);


    if(maxStride == 0)
    {
        //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
        cudaSetDevice(0);
        //#if GPUDEVICE0
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        //#endif
        K_oneThreadOne5opt_qiao_StrideIterInner5(md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, 0);
        //#if GPUDEVICE0
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeOpt, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // find the maximum gpu time for a parallel 5-opt run
        cout << "Time:: GPU side one 5-opt run : " << elapsedTimeOpt << endl;
        gpuSearchingTime0 += elapsedTimeOpt;

        //#endif
        cout << "Enter for stride 0 " << endl;

    }

    else
    {

#if SINGLGPU

        for(int iStride = 0; iStride < maxStride; iStride = iStride +1 )
        {

            cudaSetDevice(0);

            K_oneThreadOne5opt_qiao_StrideIterInner5(md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);
            //            K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream0 >>> (md_links_gpu, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);// global
        }

#endif

#if MULTIGPUMODE1
        for(int iStride = 0; iStride < maxStride; iStride = iStride + 2 )
        {
            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
            cout << "Enter for stride d0 "<< iStride << endl;
            cudaSetDevice(0);

            //faster by using the same one stream
            K_oneThreadOne5opt_qiao_StrideIterInner5(md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);
            //             K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream0 >>> (md_links_gpu, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);// global

            cout << "Enter for stride d1 "<< iStride+1 << endl;
            cudaSetDevice(1);
            //faster by using the same one stream
            K_oneThreadOne5opt_qiao_StrideIterInner5_D1(md_links_gpu_1, linkCoordTourGpu_1,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride+50);
            //             K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream1 >>> (md_links_gpu_1, linkCoordTourGpu_1,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride+1);// global


        }

#endif

#if MULTIGPUMODE2
        int taskDivision =  (int)((double)maxStride / (double)4); //5 for 7146

        for(int iStride = 0; iStride < taskDivision; iStride = iStride + 1 )
        {
            cout << "Enter for stride d1 "<< iStride << endl;
            cudaSetDevice(1);

            //faster by using the same one stream
            K_oneThreadOne5opt_qiao_StrideIterInner5_D1(md_links_gpu_1, linkCoordTourGpu_1,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);

            //            K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream1 >>> (md_links_gpu_1, linkCoordTourGpu_1,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);// global
        }

        for(int iStride = taskDivision ; iStride < maxStride +1 ; iStride = iStride + 1 )
        {
            cout << "Enter for stride d0 "<< iStride << endl;

            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
            cudaSetDevice(0);

            //faster by using the same one stream
            K_oneThreadOne5opt_qiao_StrideIterInner5(md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);

            //  K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream0 >>> (md_links_gpu, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);// global

        }

#endif
    }

    cudaStreamSynchronize(0);
    cudaStreamSynchronize(stream0);

#if MULTIGPU
    cudaStreamSynchronize(stream1);
#endif



    cudaDeviceSynchronize();
    cout << "End GPU parallel 5-opt search: " << endl;

    //    //        K_oneThreadOne5opt_RockiSmall(md_links_gpu, linkCoordTourGpu, n, maxChecks4opt, maxChecks2opt, iter);//qiao here should be iter4opt

    // end time cpu
    elapsedTimeOpt = GetCounter(pcFreq, CounterStart);
    cout << "Time:: multi GPU search time on CPU side: " << elapsedTimeOpt << endl;



    if(maxtimeGpuOptSearch < elapsedTimeOpt){
        maxtimeGpuOptSearch = elapsedTimeOpt;
    }



    float elapsedTimeOpt_DH2 = 0;
#if GPUDEVICE0
    //! sequentially select non-interacted 5-exchanges
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);

#endif

    cudaDeviceSynchronize();
    cout << "Here ending 5-opt search and begin result transfer " << endl;


    cudaSetDevice(0);
    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates

    //    //qiao only for test
    //    cout <<"device 0 density" << endl;
    //    Grid<GLfloat> densityMap_cp;
    //    densityMap_cp.resize(nCity,1);
    //    for(int i  = 0; i <512; i++)
    //        densityMap_cp[0][i] = 2 ;
    //    densityMap_cp.gpuCopyHostToDevice(md_links_gpu.densityMap);
    //    densityMap_cp.gpuCopyDeviceToHost(md_links_gpu.densityMap);
    //    for(int i  = 0; i <512; i++)
    //        cout << densityMap_cp[0][i] ;
    //    cout << endl;

#if MULTIGPU
    cudaSetDevice(1);
    densityMap_1.gpuCopyDeviceToHost(md_links_gpu_1.densityMap);
    optCandidateMap_1.gpuCopyDeviceToHost(md_links_gpu_1.optCandidateMap);//opt candidates
#endif
    cudaDeviceSynchronize();

    //    //qiao only for test
    //    cout <<"device 1 density" << endl;
    //    densityMap_cp.resize(nCity,1);
    //    for(int i  = 0; i <512; i++)
    //        densityMap_cp[0][i] = 3 ;
    //    densityMap_cp.gpuCopyHostToDevice(md_links_gpu_1.densityMap);
    //    densityMap_cp.gpuCopyDeviceToHost(md_links_gpu_1.densityMap);
    //    for(int i  = 0; i <512; i++)
    //        cout << densityMap_cp[0][i] ;
    //    cout << endl;

#if GPUDEVICE0
    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTimeOpt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H " << elapsedTimeOpt_DH2 << endl;
#endif



    //! clean for mark non-interacted 23456-opt
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB


    //    // qiao only for test
    //    int numCandidate = 0;
    //    int numCandidate_1 = 0;
    //    int numSameCandidate = 0;
    //    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    //    {
    //        if(md_links_firstPara.optCandidateMap[0][i] > 0 )
    //        {
    //            numCandidate += 1;
    //            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
    //        }

    //        if(optCandidateMap_1[0][i] != initialPrepareValueLL && optCandidateMap_1[0][i] == md_links_firstPara.optCandidateMap[0][i])
    //        {
    //            numSameCandidate += 1;
    //            cout << " find same candidate " << md_links_firstPara.grayValueMap[0][i] << endl;

    //        }
    //        else if(optCandidateMap_1[0][i] > 0)
    //        {
    //            numCandidate_1 += 1;
    //            cout << " candidate order device 1 " << md_links_firstPara.grayValueMap[0][i] << endl;

    //        }
    //    }
    //    cout << "After one GPU search num of candidates: " << numCandidate << ", device1 found candidates: " << numCandidate_1 << endl;



    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

#if MULTIGPU
    //merge results from different GPU cards  merge results from device1 to md_links_firstPara.densityMap and md_links_firstPara.optCandidateMap
    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    {
        if(optCandidateMap_1[0][i] > 0 && md_links_firstPara.optCandidateMap[0][i] == initialPrepareValueLL)
        {
            md_links_firstPara.optCandidateMap[0][i] = optCandidateMap_1[0][i];
            md_links_firstPara.densityMap[0][i] = densityMap_1[0][i];
        }

    }
#endif

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 5-opt: " << timeCpuSelectNonItera << endl;



    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTimeOptHD2 = 0;
    float elapsedTimeOptHD3 = 0;
    float elapsedTimeOpt_DH = 0;
    float elapsedTimeOpt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //qiao need to change for device1
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 5-opt: " << timeCpuExecuteNonItera << endl;

#else

    //    cudaEvent_t startHD2, stopHD2;
    //    cudaEventCreate(&startHD2);
    //    cudaEventCreate(&stopHD2);
    //    cudaEventRecord(startHD2, 0);

    cudaSetDevice(0);
    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    cudaSetDevice(1);
    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu_1.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaDeviceSynchronize();

    //    cudaEventRecord(stopHD2, 0);
    //    cudaEventSynchronize(stopHD2);
    //    cudaEventElapsedTime(&elapsedTimeOptHD3, startHD2, stopHD2);
    //    cudaEventDestroy(startHD2);
    //    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks: " <<  elapsedTimeOptHD3 << endl;

    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTimeOptHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTimeOpt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTimeOpt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTimeOpt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTimeOpt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTimeOpt_DH << endl;

#endif

    timeGpuExecute =  elapsedTimeOptHD2 + elapsedTimeOpt_execute + elapsedTimeOpt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }

    double gpuSearchingTime = (gpuSearchingTime0 > gpuSearchingTime1) ? gpuSearchingTime0 : gpuSearchingTime1;
    cout << "In one run gpu searching 5-opt max time : " << gpuSearchingTime << ", compare 0 " << gpuSearchingTime0 << ", with 1: " <<gpuSearchingTime1 << endl;

    if(percentageImprove > 0){
        timeGpuH2D += elapsedTimeOptHD + elapsedTimeOptHD2;
        timeGpuD2H += elapsedTimeOpt_DH + elapsedTimeOpt_DH2;
        timeGpuKernel += elapsedTimeOpt;// (gpuSearchingTime0 > gpuSearchingTime1) ? gpuSearchingTime0 : gpuSearchingTime1; //elapsedTimeOpt + elapsedTimeOpt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);


        //! count time gpu total
        timeGpuTotal = timeGpuH2D + timeGpuD2H + timeGpuKernel;

        //        timeTotallOneRun = elapsedTimeOptHD + elapsedTimeOptHD2 + elapsedTimeOpt_DH + elapsedTimeOpt_DH2 + elapsedTimeOpt + elapsedTimeOpt_execute
        //                + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;


        outfileTimePerRunRun << timeCpuKey + timeGpuTotal << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;

        outfileSearchTimePerRunRun << elapsedTimeOpt << endl;


        traceTSP.timeObtainKoptimal =  timeCpuKey + timeGpuTotal;

        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour

        // outfile
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("5optimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey + timeGpuTotal << ", pdb: " << pdbOneRun << ", searchTime: " << elapsedTimeOpt << endl;

        outfileKoptimalTimePerRunRun.close();

    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif


    return ret;
}//end 5opt



//! \brief Run et activate
//!
//wb.Q 202408 implement 5-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run5opt_single(string fileName) {

    cout << "Begin run 5-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdbOptEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePerRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);


    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer5optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer5optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);

    cudaSetDevice(0);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);


    //copy result from device1
    Grid<float> densityMap_1;//level 1 density map
    densityMap_1.resize(nCity,1);
    Grid<unsigned long long> optCandidateMap_1;
    optCandidateMap_1.resize(nCity,1);


    cudaSetDevice(0); //here can not add this line


    double maxChecks2opt = nCity*(nCity - 1) / 2; // total number of checks for 2-opt
    unsigned int iter = maxChecks2opt / ((double)BLOCKSIZE * (double)GRIDSIZE);

    double maxChecks4opt = maxChecks2opt*(maxChecks2opt - 1) / 2;
    unsigned int iter4opt = maxChecks4opt / ((double)BLOCKSIZE * (double)GRIDSIZE);

    cout << " maxChecks4opt= " << maxChecks4opt << " maxChecks2opt= " << maxChecks2opt << endl;

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateRocki5opt_singleGPU(numRuns, nCity, maxChecks4opt, maxChecks2opt, iter4opt, optimum,
                                        maxOptExecuPerRun, numOptimizedTotal,
                                        timeGpuKernel, timeGpuH2D, timeGpuD2H,
                                        timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                        timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                                        outfileTimePerRunRun,outfilePdbPerRunRun,outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                                        linkCoordTourCpu,linkCoordTourGpu,densityMap_1,optCandidateMap_1 );

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }
    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();

    outfilePdbPerRunRun.close();

}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki5opt_singleGPU(int& numRuns, int nCity, double maxChecks4opt,
                                                            double maxChecks2opt,
                                                            unsigned int iter,float optimum,
                                                            int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                            float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                            float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                            float& timeRefresh, float& timeSelect, float& timeExecute,
                                                            float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun,
                                                            ofstream &outfileSearchTimePerRunRun,
                                                            float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                            Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                            Grid<float>& densityMap_1, Grid<unsigned long long>& optCandidateMap_1)
{
    cout << endl << "****>>>>Enter 5-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    double timeTotallOneRun = 0;
    float elapsedTimeOpt = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt

    densityMap_1.resetValue(initialPrepareValue);
    optCandidateMap_1.resetValue(initialPrepareValue);

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps
    //! reserver, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    float elapsedTimeOptHD = 0;

    //    // time for GPU memcp HD
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);

    // copy tour ordering to gpu, clean gpu network links

    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order


    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTimeOptHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D grayValueMap : " <<  elapsedTimeOptHD << endl;


    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt


    //    //qiao only for test
    //    cout <<"device 0 " << endl;
    //    Grid<GLfloat> minRadiusMap_0;
    //    minRadiusMap_0.resize(nCity,1);
    //    minRadiusMap_0.gpuCopyDeviceToHost(md_links_gpu.minRadiusMap);
    //    for(int i  = 0; i <10; i++)
    //        cout << minRadiusMap_0[0][i] ;
    //    cout << endl;


    // cuda timer
    double time = 46;
    double *d_time;

    //qiao only for test
    cout << "Warning: maxChecks4opt= " << maxChecks4opt << " maxChecks2opt= " << maxChecks2opt << endl;

    double maxChecks4optDivide = 1.27719e+11;
    double packSize = (double)BLOCKSIZE * (double)GRIDSIZE;




    //    //! WB.Q parallel check exhaustive 5-opt along the tour for each edge
    //    for(int n = 9; n < md_links_firstPara.adaptiveMap.getWidth(); n++)
    //        //    for(int n = 9; n < 11; n++)
    //    {
    //        cout << "5-opt n-row = " << n << endl;

    //        double maxChecks2opt = n*(n - 1) / 2; // total number of checks for 2-opt
    //        double maxChecks4opt = maxChecks2opt*(maxChecks2opt - 1) / 2;

    //        double iterDivide = (double)maxChecks4optDivide /(double) (packSize);
    //        if(maxChecks4opt < packSize)
    //            iterDivide = 1;
    //        double maxStride = (double) maxChecks4opt /  (double)maxChecks4optDivide;
    //        if(maxStride <= 1)
    //            maxStride = 0;

    //        cout << "maxChecks4opt= " << maxChecks4opt << ", packSize= " << packSize << ", Changed maxChecks4optDivide = " << maxChecks4optDivide << ", iterDivide = "
    //             << iterDivide << ", maxStride= " << maxStride << endl;

    //        if(maxStride == 0)
    //        {
    //            cout << "Enter only one stride " << endl;
    //            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
    //            cudaSetDevice(0);
    //            K_oneThreadOne5opt_qiao_StrideIter(md_links_gpu, linkCoordTourGpu, n, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, 0);

    //        }

    //        else
    //        {
    //            for(double iStride = 0; iStride < maxStride; iStride = iStride +2 )
    //            {

    //                cout << "Enter loop stride " << endl;

    //                //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
    ////                cout << "Enter device 0 , CPU start id = " <<  maxChecks4optDivide * (iStride) << endl;
    //                cudaSetDevice(0);
    //                K_oneThreadOne5opt_qiao_StrideIter(md_links_gpu, linkCoordTourGpu, n, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);


    //                //                if(iStride+1 <= maxStride)
    //                //                {
    ////                cout << "Enter device 1 , CPU start id = " <<  maxChecks4optDivide * (iStride+1) << endl;
    //                cudaSetDevice(1);
    //                K_oneThreadOne5opt_qiao_StrideIter(md_links_gpu_1, linkCoordTourGpu_1, n, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride+1);
    //                //                }
    //                cudaDeviceSynchronize();

    //                cout << "Counting loop iStride = " << iStride << endl;
    //            }

    //        }
    //        cout << " End out one row " << n << endl << endl;
    //        //        K_oneThreadOne5opt_RockiSmall(md_links_gpu, linkCoordTourGpu, n, maxChecks4opt, maxChecks2opt, iter);//qiao here should be iter4opt
    //    }

    //! WB.Q one 4 edges loop n-j time in the kernel
    double iterDivide = (double)maxChecks4optDivide /(double) (packSize);
    if(maxChecks4opt < packSize)
        iterDivide = 1;
    double maxStride = (double) maxChecks4opt /  (double)maxChecks4optDivide;
    if(maxStride <= 1)
        maxStride = 0;

    cout << "maxChecks4opt= " << maxChecks4opt << ", packSize= " << packSize << ", Changed maxChecks4optDivide = " << maxChecks4optDivide << ", iterDivide = "
         << iterDivide << ", maxStride= " << maxStride << endl;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    if(maxStride == 0)
    {
        //! WB.Q parallel check exhaustive 4-opt along the tour for each edge

        K_oneThreadOne5opt_qiao_StrideIterInner5(md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, 0);

        cout << "Enter for stride 0 " << endl;

    }

    else
    {

        for(int iStride = 0; iStride < 14.7; iStride = iStride +1 )

        {

            cout << "Enter for stride " << endl;

            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge

            K_oneThreadOne5opt_qiao_StrideIterInner5(md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);


            cout << "Counting loop iStride = " << iStride << endl;

            cudaDeviceSynchronize();

        }

    }

    cudaDeviceSynchronize();
    cout << "End GPU parallel 5-opt search: " << endl;

    //    //        K_oneThreadOne5opt_RockiSmall(md_links_gpu, linkCoordTourGpu, n, maxChecks4opt, maxChecks2opt, iter);//qiao here should be iter4opt


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeOpt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // find the maximum gpu time for a parallel 5-opt run
    cout << "Time:: GPU side one 5-opt run : " << elapsedTimeOpt << endl;

    if(maxtimeGpuOptSearch < elapsedTimeOpt){
        maxtimeGpuOptSearch = elapsedTimeOpt;
    }


    float elapsedTimeOpt_DH2 = 0;

    //! sequentially select non-interacted 5-exchanges
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);


    cudaDeviceSynchronize();
    cout << "Here ending 5-opt search and begin result transfer " << endl;


    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates

    //    //qiao only for test
    //    cout <<"device 0 density" << endl;
    //    Grid<GLfloat> densityMap_cp;
    //    densityMap_cp.resize(nCity,1);
    //    for(int i  = 0; i <512; i++)
    //        densityMap_cp[0][i] = 2 ;
    //    densityMap_cp.gpuCopyHostToDevice(md_links_gpu.densityMap);
    //    densityMap_cp.gpuCopyDeviceToHost(md_links_gpu.densityMap);
    //    for(int i  = 0; i <512; i++)
    //        cout << densityMap_cp[0][i] ;
    //    cout << endl;

    cudaDeviceSynchronize();

    //    //qiao only for test
    //    cout <<"device 1 density" << endl;
    //    densityMap_cp.resize(nCity,1);
    //    for(int i  = 0; i <512; i++)
    //        densityMap_cp[0][i] = 3 ;
    //    densityMap_cp.gpuCopyHostToDevice(md_links_gpu_1.densityMap);
    //    densityMap_cp.gpuCopyDeviceToHost(md_links_gpu_1.densityMap);
    //    for(int i  = 0; i <512; i++)
    //        cout << densityMap_cp[0][i] ;
    //    cout << endl;

    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTimeOpt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H " << elapsedTimeOpt_DH2 << endl;




    // end time cpu
    elapsedTimeOpt = GetCounter(pcFreq, CounterStart);
    cout << "Time:: multi GPU search time on CPU side: " << elapsedTimeOpt << endl;


    //qiao need to change for device1
    //! clean for mark non-interacted 23456-opt
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);


    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 5-opt: " << timeCpuSelectNonItera << endl;



    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTimeOptHD2 = 0;
    float elapsedTimeOptHD3 = 0;
    float elapsedTimeOpt_DH = 0;
    float elapsedTimeOpt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //qiao need to change for device1
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 5-opt: " << timeCpuExecuteNonItera << endl;

#else

    //    cudaEvent_t startHD2, stopHD2;
    //    cudaEventCreate(&startHD2);
    //    cudaEventCreate(&stopHD2);
    //    cudaEventRecord(startHD2, 0);

    cudaSetDevice(0);
    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    cudaSetDevice(1);
    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu_1.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaDeviceSynchronize();

    //    cudaEventRecord(stopHD2, 0);
    //    cudaEventSynchronize(stopHD2);
    //    cudaEventElapsedTime(&elapsedTimeOptHD3, startHD2, stopHD2);
    //    cudaEventDestroy(startHD2);
    //    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks: " <<  elapsedTimeOptHD3 << endl;

    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTimeOptHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTimeOpt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTimeOpt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTimeOpt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTimeOpt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTimeOpt_DH << endl;

#endif

    timeGpuExecute =  elapsedTimeOptHD2 + elapsedTimeOpt_execute + elapsedTimeOpt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTimeOptHD + elapsedTimeOptHD2;
        timeGpuD2H += elapsedTimeOpt_DH + elapsedTimeOpt_DH2;
        timeGpuKernel += elapsedTimeOpt + elapsedTimeOpt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);


        //! count time gpu total
        timeGpuTotal = timeGpuH2D + timeGpuD2H + timeGpuKernel;

        //        timeTotallOneRun = elapsedTimeOptHD + elapsedTimeOptHD2 + elapsedTimeOpt_DH + elapsedTimeOpt_DH2 + elapsedTimeOpt + elapsedTimeOpt_execute
        //                + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;


        outfileTimePerRunRun << timeCpuKey + timeGpuTotal << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << elapsedTimeOpt << " " << endl;


        traceTSP.timeObtainKoptimal =  timeCpuKey + timeGpuTotal;

        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif


    return ret;
}//end 5opt



//! \brief Run et activate
//!
//wb.Q 202407 implement GPU 3-opt following rocki's method
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run3opt(string fileName) {

    cout << "Begin run 3-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdbOptEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer3optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer3optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer3optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);


    cudaSetDevice(0);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);

    //! prepare the pre-ordered link + coordinates on device 1
    cudaSetDevice(1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu_1;
    linkCoordTourGpu_1.gpuResize(nCity,1);

    //copy result from device1
    Grid<float> densityMap_1;//level 1 density map
    densityMap_1.resize(nCity,1);
    Grid<unsigned long long> optCandidateMap_1;
    optCandidateMap_1.resize(nCity,1);


#if GPUDEVICE0
    cudaSetDevice(0); //here can not add this line
#endif


    networkLinksCP.resize(nCity,1);

    double temp = (double) nCity / (double)6;
    double maxChecks3opt = temp * (nCity - 1) * (nCity - 2) ; // total number of checks for 3-opt
    double iter = (double)maxChecks3opt / ((double)BLOCKSIZE * (double)GRIDSIZE);//+1 ;//need to +1 to get maximum
    if(iter < 1)
        iter = 1;

    cout << "Check maxChecks3opt = " << maxChecks3opt << ", iter = " << iter << endl;

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 3-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateRocki3opt(numRuns, nCity, maxChecks3opt, iter, optimum,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun, outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                              linkCoordTourCpu,linkCoordTourGpu, networkLinksCP,linkCoordTourGpu_1,densityMap_1,optCandidateMap_1);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }
    }// end activateRocki

    //! free gpu memory
    cudaSetDevice(0);
    linkCoordTourGpu.gpuFreeMem();
    cudaStreamDestroy(stream0);

    cudaSetDevice(1);
    linkCoordTourGpu_1.gpuFreeMem();
    cudaStreamDestroy(stream1);


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki3opt(int& numRuns, int nCity,double maxChecks3opt, double iter,float optimum,
                                                  int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                  float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                  float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                  float& timeRefresh, float& timeSelect, float& timeExecute,
                                                  float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun,
                                                  ofstream & outfileSearchTimePerRunRun,
                                                  float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu, Grid<BufferLinkPointCoord>& networkLinksCP,  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu_1,
                                                  Grid<float>& densityMap_1, Grid<unsigned long long>& optCandidateMap_1  )
{
    cout << endl << "****>>>>Enter 3-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    double timeTotallOneRun = 0;
    float elapsedTimeOpt = 0;
    float elapsedTimeOptCPUCount = 0;


    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(0, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean md_links_firstPara before mark tour ordering
    md_links_firstPara.activeMap.resetValue(initialPrepareValue);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores k value of k-opt
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt
    md_links_firstPara.grayValueMap.resetValue(initialPrepareValue);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position


    densityMap_1.resetValue(initialPrepareValue);
    optCandidateMap_1.resetValue(initialPrepareValue);


    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;



    // time for GPU memcp HD
    float elapsedTimeOptHD = 0;
#if GPUDEVICE0
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);
#endif

    cudaSetDevice(0);

    // copy tour ordering to gpu, clean gpu network links
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order

#if MULTIGPU

    cudaSetDevice(1);
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu_1);// refresh doubly linked tour order
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu_1.grayValueMap);// refresh tsp order gpu side
#endif

#if GPUDEVICE0
    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTimeOptHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D tour order : " <<  elapsedTimeOptHD << endl;
#endif

    cudaSetDevice(0);
    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for mark k of k-opt
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change

    //    //qiao only for test
    //    cout <<"device 0 " << endl;
    //    Grid<GLfloat> minRadiusMap_0;
    //    minRadiusMap_0.resize(nCity,1);
    //    minRadiusMap_0.gpuCopyDeviceToHost(md_links_gpu.minRadiusMap);
    //    for(int i  = 0; i <10; i++)
    //        cout << minRadiusMap_0[0][i] ;
    //    cout << endl;

#if MULTIGPU
    cudaSetDevice(1);
    md_links_gpu_1.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu_1.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu_1.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt

    //    //qiao only for test
    //    cout <<"device 1 " << endl;
    //    Grid<GLfloat> minRadiusMap_1;
    //    minRadiusMap_1.resize(nCity,1);
    //    minRadiusMap_1.gpuCopyDeviceToHost(md_links_gpu_1.minRadiusMap);
    //    for(int i  = 0; i <512; i++)
    //        cout << minRadiusMap_1[0][i] ;
    //    cout << endl;

#endif

    // cuda timer
    double time = 46;
    double *d_time;


    double maxChecksoptDivide = 1.27719e+10;//1073741824;
    double packSize = (double)BLOCKSIZE * (double)GRIDSIZE;


    cudaDeviceSynchronize();
    //    cudaEventSynchronize();
    cudaStreamSynchronize(0);


    //qiao only for test
    cout << "Warning: maxChecks3opt= " << maxChecks3opt << ", width: " << md_links_firstPara.adaptiveMap.width << ", gpu.width="
         << md_links_gpu.adaptiveMap.width << endl;


    double iterDivide = (double)maxChecksoptDivide /(double) (packSize);
    if(maxChecks3opt < packSize)
        iterDivide = 1;
    double maxStride = (double) maxChecks3opt /  (double)maxChecksoptDivide;
    if(maxStride < 1)
        maxStride = 0;
    cout << "Changed maxChecks3optDivide = " << maxChecksoptDivide << ", iterDivide = " << iterDivide << ", maxStride= " << maxStride << endl;


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //qiao here does not run correctly
    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //                                 maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);


    if(maxStride == 0)
    {
        //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
        cudaSetDevice(0);

        K_oneThreadOne3opt_qiao_stride(md_links_gpu, linkCoordTourGpu, maxChecks3opt, maxChecksoptDivide, iterDivide, 0);

        //#endif
        cout << "Enter for stride 0 " << endl;

    }

    else
    {

#if SINGLGPU

        for(int iStride = 0; iStride < maxStride; iStride = iStride +1 )
        {

            cudaSetDevice(0);
            K_oneThreadOne3opt_qiao_stride(md_links_gpu, linkCoordTourGpu, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);


        }

#endif

#if MULTIGPUMODE1
        for(double iStride = 0; iStride < maxStride+1; iStride = iStride + 2 )
        {

            cudaSetDevice(0);
            //faster by using the same one stream
            K_oneThreadOne3opt_qiao_stride(md_links_gpu, linkCoordTourGpu, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);

            //                        K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest <<< b, t, 24576, stream0 >>> (md_links_gpu, linkCoordTourGpu, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);// global


            cudaSetDevice(1);
            K_oneThreadOne3opt_qiao_stride(md_links_gpu_1, linkCoordTourGpu_1, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride+1);
            //                        K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest <<< b, t, 24576, stream1 >>> (md_links_gpu_1, linkCoordTourGpu_1, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride+1);// global

            cout << "Inner one time " << iStride << endl << endl;

        }
#endif

#if MULTIGPUMODE2
        int taskDivision =  (int)((double)maxStride / (double)3);

        for(int iStride = 0; iStride < taskDivision; iStride = iStride + 1 )
        {
            cout << "Enter for stride d1 "<< iStride << endl;
            cudaSetDevice(1);

            K_oneThreadOne3opt_qiao_stride(md_links_gpu_1, linkCoordTourGpu_1, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);

            //            K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest <<< b, t, 24576, stream1 >>> (md_links_gpu_1, linkCoordTourGpu_1, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);// global
        }

        for(int iStride = taskDivision ; iStride < maxStride +1 ; iStride = iStride + 1 )
        {
            cout << "Enter for stride d0 "<< iStride << endl;

            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
            cudaSetDevice(0);

            K_oneThreadOne3opt_qiao_stride(md_links_gpu, linkCoordTourGpu, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);

            //            K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest <<< b, t, 24576, stream0 >>> (md_links_gpu, linkCoordTourGpu, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);// global

        }

#endif
    }


    cudaStreamSynchronize(0);
    cudaStreamSynchronize(nullptr);
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);


    cudaDeviceSynchronize();

    cout << "End GPU parallel 3-opt search qiao: " << endl;
    // end time cpu
    elapsedTimeOptCPUCount = GetCounter(pcFreq, CounterStart);
    cout << "Time:: multi GPU search time on CPU side single or two qiao : " << elapsedTimeOptCPUCount << endl;

    if(maxtimeGpuOptSearch < elapsedTimeOptCPUCount){
        maxtimeGpuOptSearch = elapsedTimeOptCPUCount;
    }


    //! sequentially select non-interacted 4-exchanges
    float elapsedTimeOpt_DH2 = 0;
#if GPUDEVICE0
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);
#endif

    cudaSetDevice(0);
    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);// node3
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates

#if MULTIGPU
    cudaSetDevice(1);
    densityMap_1.gpuCopyDeviceToHost(md_links_gpu_1.densityMap);
    optCandidateMap_1.gpuCopyDeviceToHost(md_links_gpu_1.optCandidateMap);//opt candidates
#endif
    cudaDeviceSynchronize();

#if GPUDEVICE0
    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTimeOpt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H 3-opt candidates: " << elapsedTimeOpt_DH2 << endl;
#endif


    //    //qiao only for test
    //    int numCandidate = 0;
    //    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    //    {
    //        if(md_links_firstPara.optCandidateMap[0][i] > 0)
    //        {
    //            numCandidate += 1;
    //            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
    //        }

    //    }
    //    cout << "After one GPU search num of candidates: " << numCandidate << endl;


    //! clean for mark non-interacted 23456-opt
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB

    // qiao only for test
    int numCandidate = 0;
    int numCandidate_1 = 0;
    int numSameCandidate = 0;
    //    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    //    {
    //        if(md_links_firstPara.optCandidateMap[0][i] > 0 )
    //        {
    //            numCandidate += 1;
    //            //            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
    //        }

    //        if(optCandidateMap_1[0][i] != initialPrepareValueLL && optCandidateMap_1[0][i] == md_links_firstPara.optCandidateMap[0][i])
    //        {
    //            numSameCandidate += 1;
    //            //            cout << " find same candidate " << md_links_firstPara.grayValueMap[0][i] << endl;

    //        }
    //        else if(optCandidateMap_1[0][i] > 0)
    //        {
    //            numCandidate_1 += 1;
    //            //            cout << " candidate order device 1 " << md_links_firstPara.grayValueMap[0][i] << endl;

    //        }
    //    }
    //    cout << "After one: device1 search num of candidates: " << numCandidate << ", device1 found candidates: " << numCandidate_1 << endl;



    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

#if MULTIGPU
    //merge results from different GPU cards  merge results from device1 to md_links_firstPara.densityMap and md_links_firstPara.optCandidateMap
    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    {
        if(optCandidateMap_1[0][i] > 0 && md_links_firstPara.optCandidateMap[0][i] == initialPrepareValueLL)
        {
            md_links_firstPara.optCandidateMap[0][i] = optCandidateMap_1[0][i];
            md_links_firstPara.densityMap[0][i] = densityMap_1[0][i];
        }

    }
#endif

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 3-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTimeOptHD2 = 0;
    float elapsedTimeOptHD3 = 0;
    float elapsedTimeOpt_DH = 0;
    float elapsedTimeOpt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //GPU version final version correct
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera intera 3-opt: " << timeCpuExecuteNonItera << endl;


#else

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks: " <<  elapsedTimeOptHD3 << endl;


    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTimeOptHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTimeOpt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTimeOpt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTimeOpt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTimeOpt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTimeOpt_DH << endl;

#endif


    timeGpuExecute =  elapsedTimeOptHD2 + elapsedTimeOpt_execute + elapsedTimeOpt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    //    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 3-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;


    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTimeOptHD + elapsedTimeOptHD2;
        timeGpuD2H += elapsedTimeOpt_DH + elapsedTimeOpt_DH2;
        timeGpuKernel += elapsedTimeOptCPUCount + elapsedTimeOpt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);

        //! count time gpu total
        timeGpuTotal = timeGpuH2D + timeGpuD2H + timeGpuKernel;

        //        timeTotallOneRun = elapsedTimeOptHD + elapsedTimeOptHD2 + elapsedTimeOpt_DH +
        //                                     elapsedTimeOpt_DH2 + elapsedTimeOpt + elapsedTimeOpt_execute
        //                + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;



        outfileTimePerRunRun << timeCpuKey + timeGpuTotal << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << elapsedTimeOptCPUCount << endl;

        traceTSP.timeObtainKoptimal = timeCpuKey + timeGpuTotal;

        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour
        // outfile
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("3optimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey + timeGpuTotal << ", pdb: " << pdbOneRun << ", searchTime: " << elapsedTimeOptCPUCount << endl;

        outfileKoptimalTimePerRunRun.close();

    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif

    return ret;
}



//! \brief Run et activate
//!
//wb.Q 202408 implement 6-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run6opt(string fileName) {

    cout << "Begin run 6-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdbOptEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePerRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);


    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer6optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer6optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);


    double temp = (double) nCity / (double)6;
    double maxChecks3opt = (double)temp * ((double)nCity - 1) * ((double)nCity - 2) ; // total number of checks for 3-opt
    double temp2 = (double) maxChecks3opt /(double)2;
    double maxCheck6opt = (double)temp2*((double)maxChecks3opt - 1);  // total number of checks for 6-opt
    unsigned int iter = (double)maxCheck6opt / ((double)BLOCKSIZE * (double)GRIDSIZE);

    cout << "Check maxChecks6opt = " << maxCheck6opt << ", iter = " << iter << endl;


    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateRocki6opt(numRuns, nCity, maxCheck6opt, maxChecks3opt,iter, optimum,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun, outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                              linkCoordTourCpu,linkCoordTourGpu);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }
    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);



    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki6opt(int& numRuns, int nCity, double maxChecks6opt,
                                                  double maxChecks3opt, unsigned int iter,float optimum,
                                                  int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                  float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                  float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                  float& timeRefresh, float& timeSelect, float& timeExecute,
                                                  float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun,ofstream & outfileSearchTimePerRunRun,
                                                  float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu )
{
    cout << endl << "****>>>>Enter 6-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    double timeTotallOneRun = 0;
    float elapsedTimeOpt = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;


    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(0, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps
    //! reserver, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    // time for GPU memcp HD
    float elapsedTimeOptHD = 0;
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);

    // copy tour ordering to gpu, clean gpu network links
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order

    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTimeOptHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D grayValueMap : " <<  elapsedTimeOptHD << endl;


    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt


    // cuda timer
    double time = 46;
    double *d_time;

    double maxChecksoptDivide = 1.27719e+11;
    double packSize = (double)BLOCKSIZE * (double)GRIDSIZE;


    //qiao only for test
    cout << "Warning: maxChecks6opt= " << maxChecks6opt << endl;

    double iterDivide = (double)maxChecksoptDivide /(double) (packSize);
    if(maxChecks6opt < packSize)
        iterDivide = 1;
    double maxStride = (double) maxChecks6opt /  (double)maxChecksoptDivide;
    //    double maxStride = (double) maxChecks3opt /  (double)maxChecksoptDivide; //max 6-opt edge's order 132
    if(maxStride < 1)
        maxStride = 0;
    cout << "Changed maxChecks6optDivide = " << maxChecksoptDivide << ", iterDivide = " << iterDivide << ", maxStride= " << maxStride << endl;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //    for(double iStride = 0; iStride < maxStride+1; iStride++ )
    for(double iStride = 0; iStride < 1; iStride++ )
    {

        K_oneThreadOne6opt_qiao_iterStride(md_links_gpu, linkCoordTourGpu,maxChecks6opt, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);

        //        cout << "Inner one time " << iStride << endl << endl;

    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeOpt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // find the maximum gpu time for a parallel 2-opt run
    cout << "Time:: GPU side one 6-opt run : " << elapsedTimeOpt << endl;

    if(maxtimeGpuOptSearch < elapsedTimeOpt){
        maxtimeGpuOptSearch = elapsedTimeOpt;
    }


    //! sequentially select non-interacted 2-exchanges
    float elapsedTimeOpt_DH2 = 0;
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);

    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates


    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTimeOpt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H " << elapsedTimeOpt_DH2 << endl;


    //    //qiao only for test
    //    int numCandidate = 0;
    //    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    //    {
    //        if(md_links_firstPara.optCandidateMap[0][i] > 0)
    //        {
    //            numCandidate += 1;
    //            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
    //        }

    //    }
    //    cout << "After one GPU search num of candidates: " << numCandidate << endl;


    //! clean for mark non-interacted 6-opt
    md_links_firstPara.activeMap.resetValue(0); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(0); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 6-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTimeOptHD2 = 0;
    float elapsedTimeOptHD3 = 0;
    float elapsedTimeOpt_DH = 0;
    float elapsedTimeOpt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);
    //    //    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify only work for reading possibilites to nodeParent
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);

    //    networkLinksCP.resize(nCity, 1);
    //    networkLinksCP.assign(md_links_firstPara.networkLinks);
    //    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap, networkLinksCP);//qiao 2024 need modify


    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 6-opt: " << timeCpuExecuteNonItera << endl;


#else

    //    cudaEvent_t startHD2, stopHD2;
    //    cudaEventCreate(&startHD2);
    //    cudaEventCreate(&stopHD2);
    //    cudaEventRecord(startHD2, 0);

    //    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    //    errorCheckCudaThreadSynchronize();

    //    cudaEventRecord(stopHD2, 0);
    //    cudaEventSynchronize(stopHD2);
    //    cudaEventElapsedTime(&elapsedTimeOptHD3, startHD2, stopHD2);
    //    cudaEventDestroy(startHD2);
    //    cudaEventDestroy(stopHD2);
    //    cout << "Time:: memcp H to D networkLinks:  " <<  elapsedTimeOptHD3 << endl;


    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTimeOptHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTimeOpt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTimeOpt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTimeOpt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTimeOpt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTimeOpt_DH << endl;

#endif


    timeGpuExecute =  elapsedTimeOptHD2 + elapsedTimeOpt_execute + elapsedTimeOpt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 6-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTimeOptHD + elapsedTimeOptHD2;
        timeGpuD2H += elapsedTimeOpt_DH + elapsedTimeOpt_DH2;
        timeGpuKernel += elapsedTimeOpt + elapsedTimeOpt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);

        //! count time gpu total
        timeGpuTotal = timeGpuH2D + timeGpuD2H + timeGpuKernel;
        //        timeTotallOneRun = elapsedTimeOptHD + elapsedTimeOptHD2 + elapsedTimeOpt_DH + elapsedTimeOpt_DH2 + elapsedTimeOpt + elapsedTimeOpt_execute
        //                + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;


        outfileTimePerRunRun << timeCpuKey + timeGpuTotal << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << elapsedTimeOpt << endl;


        traceTSP.timeObtainKoptimal =  timeCpuKey + timeGpuTotal;

        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour

        // outfile
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("6optimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey + timeGpuTotal << ", pdb: " << pdbOneRun << ", searchTime: " << elapsedTimeOpt << endl;

        outfileKoptimalTimePerRunRun.close();

    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif

    return ret;
}// end 6opt



//! \brief Run et activate
//!
//wb.Q 202408 implement 5-opt from 6-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run5opt_2(string fileName) {

    cout << "Begin run 5-opt_2 >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdbOptEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer5opt2Run.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile timeline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer5opt2Run.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer3optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;



    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);


    cudaSetDevice(0);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);

    //! prepare the pre-ordered link + coordinates on device 1
    cudaSetDevice(1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu_1;
    linkCoordTourGpu_1.gpuResize(nCity,1);


    //copy result from device1
    Grid<float> densityMap_1;//level 1 density map
    densityMap_1.resize(nCity,1);
    Grid<unsigned long long> optCandidateMap_1;
    optCandidateMap_1.resize(nCity,1);


    double temp = (double) nCity / (double)6;
    double maxChecks3opt = temp * (nCity - 1) * (nCity - 2) ; // total number of checks for 3-opt
    double temp2 = (double) maxChecks3opt /(double)2;
    double maxCheck6opt = temp2*(maxChecks3opt - 1);  // total number of checks for 6-opt
    double iter = (double)maxCheck6opt / (double) (BLOCKSIZE * GRIDSIZE);

    cout << "Check maxChecks6opt = " << maxCheck6opt << ", iter = " << iter << endl;


    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateRocki5opt_2(numRuns, nCity, maxCheck6opt, maxChecks3opt,iter, optimum,
                                maxOptExecuPerRun, numOptimizedTotal,
                                timeGpuKernel, timeGpuH2D, timeGpuD2H,
                                timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                                outfileTimePerRunRun, outfilePdbPerRunRun,outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                                linkCoordTourCpu,linkCoordTourGpu,linkCoordTourGpu_1,densityMap_1,optCandidateMap_1);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }
    }// end activateRocki

    //! free gpu memory
    cudaSetDevice(0);
    linkCoordTourGpu.gpuFreeMem();

    cudaSetDevice(1);
    linkCoordTourGpu_1.gpuFreeMem();


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

// qiao 2024 add run 5-opt from 6-opt
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki5opt_2(int& numRuns, int nCity, double maxChecks6opt,
                                                    double maxChecks3opt, unsigned int iter,float optimum,
                                                    int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                    float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                    float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                    float& timeRefresh, float& timeSelect, float& timeExecute,
                                                    float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun, ofstream &outfileSearchTimePerRunRun,
                                                    float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                    Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu_1,
                                                    Grid<float>& densityMap_1, Grid<unsigned long long>& optCandidateMap_1 )
{
    cout << endl << "****>>>>Enter 5-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    double timeTotallOneRun = 0;
    float elapsedTimeOpt = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;


    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    densityMap_1.resetValue(initialPrepareValue);
    optCandidateMap_1.resetValue(initialPrepareValue);


    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps
    //! reserver, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    // time for GPU memcp HD
    float elapsedTimeOptHD = 0;
    //    cudaEvent_t startHD, stopHD;
    //    cudaEventCreate(&startHD);
    //    cudaEventCreate(&stopHD);
    //    cudaEventRecord(startHD, 0);

    // copy tour ordering to gpu, clean gpu network links
    cudaSetDevice(0);
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order
    cudaSetDevice(1);
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu_1);// refresh doubly linked tour order
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu_1.grayValueMap);// refresh tsp order gpu side

    //    cudaEventRecord(stopHD, 0);
    //    cudaEventSynchronize(stopHD);
    //    cudaEventElapsedTime(&elapsedTimeOptHD, startHD, stopHD);
    //    cudaEventDestroy(startHD);
    //    cudaEventDestroy(stopHD);
    //    cout << "Time:: memcp H to D grayValueMap : " <<  elapsedTimeOptHD << endl;


    cudaSetDevice(0);
    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt


    //qiao only for test
    cout <<"device 0 " << endl;
    Grid<GLfloat> minRadiusMap_0;
    minRadiusMap_0.resize(nCity,1);
    minRadiusMap_0.gpuCopyDeviceToHost(md_links_gpu.minRadiusMap);
    for(int i  = 0; i <10; i++)
        cout << minRadiusMap_0[0][i] ;
    cout << endl;


    cudaSetDevice(1);
    md_links_gpu_1.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu_1.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu_1.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt

    //qiao only for test
    cout <<"device 1 " << endl;
    Grid<GLfloat> minRadiusMap_1;
    minRadiusMap_1.resize(nCity,1);
    minRadiusMap_1.gpuCopyDeviceToHost(md_links_gpu_1.minRadiusMap);
    for(int i  = 0; i <512; i++)
        cout << minRadiusMap_1[0][i] ;
    cout << endl;


    // cuda timer
    double time = 46;
    double *d_time;

    double maxChecksoptDivide = 21990232545280; //1.27719e+11; //2147483647
    double packSize = (double)BLOCKSIZE * (double)GRIDSIZE;


    //qiao only for test
    cout << "Warning: maxChecks6opt= " << maxChecks6opt << endl;

    cudaDeviceSynchronize();
    //    cudaEventSynchronize();
    cudaStreamSynchronize(0);


    //    cudaEvent_t start, stop;
    //    cudaEventCreate(&start);
    //    cudaEventCreate(&stop);
    //    cudaEventRecord(start, 0);

    double iterDivide = (double)maxChecksoptDivide /(double) (packSize);
    if(maxChecks6opt < packSize)
        iterDivide = 1;
    double maxStride = (double) maxChecks6opt /  (double)maxChecksoptDivide;

    if(maxStride < 1)
        maxStride = 0;
    cout << "Changed maxChecks6optDivide = " << maxChecksoptDivide
         << ", packsize= " << packSize<<  ", iterDivide = " << iterDivide << ", maxStride= " << maxStride << endl;


    if(maxStride == 0)
    {
        //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
        cudaSetDevice(0);
        K_5opt_qiao_iterStride(md_links_gpu, linkCoordTourGpu,maxChecks6opt, maxChecks3opt, maxChecksoptDivide, iterDivide, 0);

        cout << "Enter for stride 0 " << endl;

    }
    else
    {
        //    for(double iStride = 0; iStride < maxStride; iStride = iStride + 2 )
        for(double iStride = 0; iStride < 21; iStride = iStride +2 )
        {

            cout << "Enter for stride " << endl;

            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
            cudaSetDevice(0);
            K_5opt_qiao_iterStride(md_links_gpu, linkCoordTourGpu,maxChecks6opt, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);

            cudaSetDevice(1);
            K_5opt_qiao_iterStride(md_links_gpu_1, linkCoordTourGpu_1,maxChecks6opt, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);

            cout << "Counting loop iStride = " << iStride << endl;

        }

    }
    cudaDeviceSynchronize();


    //    //    for(double iStride = 0; iStride < maxStride+1; iStride++ )
    //    for(double iStride = 0; iStride < 1; iStride++ )
    //    {

    //        K_5opt_qiao_iterStride(md_links_gpu, linkCoordTourGpu,maxChecks6opt, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);

    //        cout << "Inner one time " << iStride << endl << endl;

    //    }

    //    cudaDeviceSynchronize();

    //! WB.Q parallel check exhaustive 6-opt along the tour for each edge
    //    K_oneThreadOne6opt_RockiSmall(md_links_gpu, linkCoordTourGpu, maxChecks6opt,  maxChecks3opt, iter);

    cout << "End GPU parallel 5-opt search: " << endl;

    //    cudaEventRecord(stop, 0);
    //    cudaEventSynchronize(stop);
    //    cudaEventElapsedTime(&elapsedTimeOpt, start, stop);
    //    cudaEventDestroy(start);
    //    cudaEventDestroy(stop);
    //    // find the maximum gpu time for a parallel 2-opt run
    //    cout << "Time:: GPU side one 6-opt run : " << elapsedTimeOpt << endl;

    //    if(maxtimeGpuOptSearch < elapsedTimeOpt){
    //        maxtimeGpuOptSearch = elapsedTimeOpt;
    //    }


    //! sequentially select non-interacted 2-exchanges
    float elapsedTimeOpt_DH2 = 0;
    //    cudaEvent_t startDH2, stopDH2;
    //    cudaEventCreate(&startDH2);
    //    cudaEventCreate(&stopDH2);
    //    cudaEventRecord(startDH2, 0);

    cudaSetDevice(0);
    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates


    //    cudaEventRecord(stopDH2, 0);
    //    cudaEventSynchronize(stopDH2);
    //    cudaEventElapsedTime(&elapsedTimeOpt_DH2, startDH2, stopDH2);
    //    cudaEventDestroy(startDH2);
    //    cudaEventDestroy(stopDH2);
    //    cout << "Time:: memcp D to H " << elapsedTimeOpt_DH2 << endl;


    //    //qiao only for test
    //    int numCandidate = 0;
    //    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    //    {
    //        if(md_links_firstPara.optCandidateMap[0][i] > 0)
    //        {
    //            numCandidate += 1;
    //            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
    //        }

    //    }
    //    cout << "After one GPU search num of candidates: " << numCandidate << endl;

    cudaSetDevice(1);
    densityMap_1.gpuCopyDeviceToHost(md_links_gpu_1.densityMap);
    optCandidateMap_1.gpuCopyDeviceToHost(md_links_gpu_1.optCandidateMap);//opt candidates

    cudaDeviceSynchronize();

    //! clean for mark non-interacted 6-opt
    md_links_firstPara.activeMap.resetValue(0); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(0); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! qiao merge results from device1 to md_links_firstPara.densityMap and md_links_firstPara.optCandidateMap
    int numCandidate = 0;
    int numCandidate_1 = 0;
    int numSameCandidate = 0;
    //    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    //    {
    //        if(md_links_firstPara.optCandidateMap[0][i] > 0 )
    //        {
    //            numCandidate += 1;
    //            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
    //        }

    //        if(optCandidateMap_1[0][i] != initialPrepareValueLL && optCandidateMap_1[0][i] == md_links_firstPara.optCandidateMap[0][i])
    //        {
    //            numSameCandidate += 1;
    //            cout << " find same candidate " << md_links_firstPara.grayValueMap[0][i] << endl;

    //        }

    //        //! QIAO work code
    //        else if(optCandidateMap_1[0][i] > 0)
    //        {
    //            numCandidate_1 += 1;
    //            cout << " candidate order device 1 " << md_links_firstPara.grayValueMap[0][i] << endl;

    //            if(md_links_firstPara.optCandidateMap[0][i] == initialPrepareValueLL)
    //            {
    //                md_links_firstPara.optCandidateMap[0][i] = optCandidateMap_1[0][i];
    //                md_links_firstPara.densityMap[0][i] = densityMap_1[0][i];

    //            }

    //        }

    //    }
    //    cout << "After one GPU search num of candidates: " << numCandidate << ", device1 found candidates: " << numCandidate_1 << endl;

#if MULTIGPU
    //merge results from different GPU cards  merge results from device1 to md_links_firstPara.densityMap and md_links_firstPara.optCandidateMap
    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    {
        if(optCandidateMap_1[0][i] > 0 && md_links_firstPara.optCandidateMap[0][i] == initialPrepareValueLL)
        {
            md_links_firstPara.optCandidateMap[0][i] = optCandidateMap_1[0][i];
            md_links_firstPara.densityMap[0][i] = densityMap_1[0][i];
        }

    }
#endif

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 6-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTimeOptHD2 = 0;
    float elapsedTimeOptHD3 = 0;
    float elapsedTimeOpt_DH = 0;
    float elapsedTimeOpt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //qiao need to change for device1
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 5-opt: " << timeCpuExecuteNonItera << endl;


#else

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks:  " <<  elapsedTimeOptHD3 << endl;


    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTimeOptHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTimeOpt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTimeOpt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTimeOpt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTimeOpt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTimeOpt_DH << endl;

#endif


    timeGpuExecute =  elapsedTimeOptHD2 + elapsedTimeOpt_execute + elapsedTimeOpt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 5-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTimeOptHD + elapsedTimeOptHD2;
        timeGpuD2H += elapsedTimeOpt_DH + elapsedTimeOpt_DH2;
        timeGpuKernel += elapsedTimeOpt + elapsedTimeOpt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);

        //! count time gpu total
        timeGpuTotal = timeGpuH2D + timeGpuD2H + timeGpuKernel;
        //        timeTotallOneRun = timeCpuKey + timeGpuTotal;
        //        timeTotallOneRun = elapsedTimeOptHD + elapsedTimeOptHD2 + elapsedTimeOpt_DH + elapsedTimeOpt_DH2 + elapsedTimeOpt + elapsedTimeOpt_execute
        //                + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;


        outfileTimePerRunRun << timeCpuKey + timeGpuTotal << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << elapsedTimeOpt << endl;

        traceTSP.timeObtainKoptimal =  timeCpuKey + timeGpuTotal;

        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif

    return ret;
}// end 6opt



//! \brief Run et activate
//!
//wb.Q 202408 implement 6-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::runVariableKopt(string fileName) {

    cout << "Begin GPU variable k-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdbOptEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePerVariablKoptRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);


    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPerVariablKoptRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePerVkoptRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;


    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);


    double temp = (double) nCity / (double)6;
    double maxChecks3opt = temp * (nCity - 1) * (nCity - 2) ; // total number of checks for 3-opt
    double temp2 = (double) maxChecks3opt /(double)2;
    double maxCheck6opt = temp2*(maxChecks3opt - 1);  // total number of checks for 6-opt
    unsigned int iter = maxCheck6opt / (BLOCKSIZE * GRIDSIZE);

    cout << "Check maxChecks6opt = " << maxCheck6opt << ", iter = " << iter << endl;


    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 variable k-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateVariableKopt(numRuns, nCity, maxCheck6opt, maxChecks3opt,iter, optimum,
                                 maxOptExecuPerRun, numOptimizedTotal,
                                 timeGpuKernel, timeGpuH2D, timeGpuD2H,
                                 timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                 timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                                 outfileTimePerRunRun, outfilePdbPerRunRun,outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                                 linkCoordTourCpu,linkCoordTourGpu);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }
    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel variable 23456-opt and massive variable 23456-opt moves on global tour
// qiao each id can be checed with 2-/3-/4-/6-opt increasingly, until find one
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateVariableKopt(int& numRuns, int nCity, double maxChecks6opt,
                                                     double maxChecks3opt, unsigned int iter,float optimum,
                                                     int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                     float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                     float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                     float& timeRefresh, float& timeSelect, float& timeExecute,
                                                     float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun,
                                                     ofstream & outfileSearchTimePerRunRun,
                                                     float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                     Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu)
{
    cout << endl << "****>>>>Enter GPU variable k-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    double timeTotallOneRun = 0;
    float elapsedTimeOpt = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(0, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps
    //! reserver, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    // time for GPU memcp HD
    float elapsedTimeOptHD = 0;
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);

    // copy tour ordering to gpu, clean gpu network links
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order

    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTimeOptHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D grayValueMap : " <<  elapsedTimeOptHD << endl;


    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt


    // cuda timer
    double time = 46;
    double *d_time;

    double maxChecksoptDivide = 1.27719e+11;
    double packSize = (double)BLOCKSIZE * (double)GRIDSIZE;


    double maxChecks2opt = nCity*(nCity - 1) / 2; // total number of checks for 2-opt
    double temp4opt = (double)maxChecks2opt/ (double)2;
    double maxChecks4opt = temp4opt*(maxChecks2opt - 1);


    //qiao only for test
    cout << "Warning: maxChecks6opt= " << maxChecks6opt << endl;
    double iterDivide = (double)maxChecksoptDivide /(double) (packSize);
    if(maxChecks6opt < packSize)
        iterDivide = 1;
    double maxStride = (double) maxChecks6opt /  (double)maxChecksoptDivide;
    double maxStride3opt = (double) maxChecks3opt /  (double)maxChecksoptDivide; //max 6-opt edge's order 132
    double maxStride4opt = (double) maxChecks4opt /  (double)maxChecksoptDivide; //max 6-opt edge's order 132

    if(maxStride < 1)
        maxStride = 0;
    cout << "Changed maxChecks6optDivide = " << maxChecksoptDivide << ", iterDivide = " << iterDivide << ", maxStride= " << maxStride << endl;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //    for(double iStride = 0; iStride < maxStride+1; iStride++ )
    for(double iStride = 0; iStride < maxStride4opt; iStride++ )
    {

        K_VariableKopt_qiao_iterStride(md_links_gpu, linkCoordTourGpu, maxChecks2opt,maxChecks3opt, maxChecks4opt, maxChecks6opt,  maxChecksoptDivide, iterDivide, iStride);
        cudaDeviceSynchronize();
        cout << "Inner one time " << iStride << endl << endl;

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeOpt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // find the maximum gpu time for a parallel 2-opt run
    cout << "Time:: GPU side one variable-opt run : " << elapsedTimeOpt << endl;

    if(maxtimeGpuOptSearch < elapsedTimeOpt){
        maxtimeGpuOptSearch = elapsedTimeOpt;
    }


    //! sequentially select non-interacted 2-exchanges
    float elapsedTimeOpt_DH2 = 0;
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);

    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates


    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTimeOpt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H " << elapsedTimeOpt_DH2 << endl;


    //    //qiao only for test
    //    int numCandidate = 0;
    //    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    //    {
    //        if(md_links_firstPara.optCandidateMap[0][i] > 0)
    //        {
    //            numCandidate += 1;
    //            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
    //        }

    //    }
    //    cout << "After one GPU search num of candidates: " << numCandidate << endl;


    //! clean for mark non-interacted 6-opt
    md_links_firstPara.activeMap.resetValue(0); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(0); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera v-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTimeOptHD2 = 0;
    float elapsedTimeOptHD3 = 0;
    float elapsedTimeOpt_DH = 0;
    float elapsedTimeOpt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);
    //    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify

    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera v-opt: " << timeCpuExecuteNonItera << endl;


#else

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks:  " <<  elapsedTimeOptHD3 << endl;


    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTimeOptHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTimeOptHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTimeOpt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTimeOpt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTimeOpt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTimeOpt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTimeOpt_DH << endl;

#endif


    timeGpuExecute =  elapsedTimeOptHD2 + elapsedTimeOpt_execute + elapsedTimeOpt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 6-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTimeOptHD + elapsedTimeOptHD2;
        timeGpuD2H += elapsedTimeOpt_DH + elapsedTimeOpt_DH2;
        timeGpuKernel += elapsedTimeOpt + elapsedTimeOpt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);


        //! count time gpu total
        timeGpuTotal = timeGpuH2D + timeGpuD2H + timeGpuKernel;
        //        timeTotallOneRun = elapsedTimeOptHD + elapsedTimeOptHD2 + elapsedTimeOpt_DH + elapsedTimeOpt_DH2 + elapsedTimeOpt + elapsedTimeOpt_execute
        //                + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;



        outfileTimePerRunRun << timeCpuKey + timeGpuTotal << " " << endl;

        outfilePdbPerRunRun << pdbOneRun << " " << endl;

        traceTSP.timeObtainKoptimal =  timeCpuKey + timeGpuTotal;
        outfileSearchTimePerRunRun << elapsedTimeOpt << endl;

        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);

    }
    else{
        numRuns -= 1; // the last run does not optimized the tour

        // outfile
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("variableKoptimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey + timeGpuTotal << ", pdb: " << pdbOneRun << ", searchTime: " << elapsedTimeOpt << endl;

        outfileKoptimalTimePerRunRun.close();

    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif

    return ret;
}// end variable kopt



//! \brief Run et activate
//!
//wb.Q 202408 implement iterative k-optimal
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::runGpuIterativeKoptimal(string fileName) {

    cout << "Begin run runGpuIterativeKoptimal >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdbOptEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePerIterKoptimalRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPerIterKoptimalRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePerIterKoptRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;

    float evaLastRun = 0;
    float percentageImprove2opt = 999999;
    float percentageImprove3opt = 999999;
    float percentageImprove4opt = 999999;
    float percentageImprove5opt = 999999;
    float percentageImprove6opt = 999999;

    networkLinksCP.resize(nCity,0);

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);


    cudaSetDevice(0);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);

    //! prepare the pre-ordered link + coordinates on device 1
    cudaSetDevice(1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu_1;
    linkCoordTourGpu_1.gpuResize(nCity,1);


    //copy result from device1
    Grid<float> densityMap_1;//level 1 density map
    densityMap_1.resize(nCity,1);
    Grid<unsigned long long> optCandidateMap_1;
    optCandidateMap_1.resize(nCity,1);

    //    cudaSetDevice(0); // necessary for running the following run2opt


    unsigned long maxChecks2opt = nCity*(nCity - 1) / 2; // total number of checks for 2-opt
    unsigned int iter2opt = maxChecks2opt / (BLOCKSIZE * GRIDSIZE);
    cout << "Check maxChecks2opt = " << maxChecks2opt << ", iter = " << iter2opt << endl;

    double temp = (double) nCity / (double)6;
    double maxChecks3opt = temp * (nCity - 1) * (nCity - 2) ; // total number of checks for 3-opt
    double iter3opt = ( maxChecks3opt / (BLOCKSIZE * GRIDSIZE) ) ;//+1 ;//need to +1 to get maximum
    if(iter3opt < 1)
        iter3opt = 1;

    double temptemp =  (double)maxChecks2opt/ (double)2;
    double maxChecks4opt = temptemp*(maxChecks2opt - 1);
    double iter4opt = (double)maxChecks4opt /(double) (BLOCKSIZE * GRIDSIZE);
    if(iter4opt < 1)
        iter4opt = 1;


    double temp2 = (double) maxChecks3opt /(double)2;
    double maxCheck6opt = temp2*(maxChecks3opt - 1);  // total number of checks for 6-opt
    unsigned int iter6opt = maxCheck6opt / (BLOCKSIZE * GRIDSIZE);


    int numCityTraversed = 0;

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {

        //GPU 2-opt
        while (numRuns < NUMRUNSLIMIT  && percentageImprove2opt > 0 )
        {
            activateRocki2opt(numRuns, nCity, maxChecks2opt, iter2opt, optimum,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun,outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove2opt,
                              linkCoordTourCpu,linkCoordTourGpu);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }

        //cout TSP length of 2-optimal
        float eva2optimal = md_links_firstPara.evaluateWeightOfTSP(distEuclidean,numCityTraversed);
        cout << "Evaluate:: eva2optimal =  " << eva2optimal << endl;


#if ONERUNTEST
        int oneRunTest = numRuns;
        while (numRuns < oneRunTest+1  && percentageImprove3opt > 0 )
#else
        //GPU 3-opt
        while (numRuns < NUMRUNSLIMIT  && percentageImprove3opt > 0 )
#endif
        {
            activateRocki3opt(numRuns, nCity, maxChecks3opt, iter3opt, optimum,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun, outfilePdbPerRunRun,outfileSearchTimePerRunRun, evaLastRun, percentageImprove3opt,
                              linkCoordTourCpu,linkCoordTourGpu,networkLinksCP,linkCoordTourGpu_1,densityMap_1,optCandidateMap_1);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }


        //cout TSP length of 3-optimal
        float eva3optimal = md_links_firstPara.evaluateWeightOfTSP(distEuclidean,numCityTraversed);
        cout << "Evaluate:: eva3optimal =  " << eva3optimal << endl;


#if ONERUNTEST
        oneRunTest = numRuns;
        while (numRuns < oneRunTest+1  && percentageImprove4opt > 0 )
#else
        //GPU 4-opt
        while (numRuns < NUMRUNSLIMIT  && percentageImprove4opt > 0 )
#endif
        {
            activateRocki4opt(numRuns, nCity, maxChecks2opt, maxChecks4opt, iter4opt, optimum,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun,outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove4opt,
                              linkCoordTourCpu,linkCoordTourGpu,linkCoordTourGpu_1,densityMap_1,optCandidateMap_1);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }

        //cout TSP length of 4-optimal
        float eva4optimal = md_links_firstPara.evaluateWeightOfTSP(distEuclidean,numCityTraversed);
        cout << "Evaluate:: eva4optimal =  " << eva4optimal << endl;


        int runs = 0;

        //GPU 5-opt
#if ONERUNTEST
        oneRunTest = numRuns;
        while (numRuns < oneRunTest+1  && percentageImprove5opt > 0 )
#else
        while (numRuns < NUMRUNSLIMIT  && percentageImprove5opt > 0 && runs < 6 )
#endif
        {
            runs ++;
            activateRocki5opt(numRuns, nCity, maxChecks4opt, maxChecks2opt, iter4opt, optimum,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun,outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove5opt,
                              linkCoordTourCpu,linkCoordTourGpu,linkCoordTourGpu_1,densityMap_1,optCandidateMap_1 );

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }

        //        cudaSetDevice(0);
        //        //GPU 5-opt
        //        while (numRuns < NUMRUNSLIMIT  && percentageImprove5opt > 0 )
        //        {
        //            activateRocki5opt_2(numRuns, nCity, maxChecks4opt, maxChecks2opt, iter4opt, optimum,
        //                              maxOptExecuPerRun, numOptimizedTotal,
        //                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
        //                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
        //                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
        //                              outfileTimePerRunRun, evaLastRun, percentageImprove5opt,
        //                              linkCoordTourCpu,linkCoordTourGpu,linkCoordTourGpu_1,densityMap_1,optCandidateMap_1 );

        //            if (g_ConfigParameters->traceActive) {
        //                evaluate();
        //                writeStatisticsToFile(numRuns, fileName);
        //            }
        //        }

        //cout TSP length of 5-optimal
        float eva5optimal = md_links_firstPara.evaluateWeightOfTSP(distEuclidean,numCityTraversed);
        cout << "Evaluate:: eva5opt milstone =  " << eva5optimal << endl;



        //wb.Q 2024 6-opt
        runs = 0;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove6opt > 0 && runs < 6)
        {
            runs ++;
            activateRocki6opt(numRuns, nCity, maxCheck6opt, maxChecks3opt,iter6opt, optimum,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun,outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove6opt,
                              linkCoordTourCpu,linkCoordTourGpu);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, fileName);
            }
        }

        float eva6optimal = md_links_firstPara.evaluateWeightOfTSP(distEuclidean,numCityTraversed);
        cout << "Evaluate:: eva6opt milstone =  " << eva6optimal << endl;


    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();
    linkCoordTourGpu_1.gpuFreeMem();
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run


