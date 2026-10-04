#include <cmath>
#include "SolutionKOPT.h"
//#include "Multiout.h"

#define TEST_CODE   0


#define SFX_STATICS_TSP ".statics"

#define round(x) ((fabs(ceil(x) - (x)) < fabs(floor(x) - (x))) ? ceil(x) : floor(x))

//template <typename T> std::string tostr(const T& t)
//{
//    std::ostringstream os; os << t; return os.str();
//}

template<std::size_t DimP, std::size_t DimCM>
std::ofstream* SolutionKOPT<DimP, DimCM>::OutputStream = NULL;

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::readSolution()
{
    readSolution(fileData);
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::readSolution(int functionModeChoice)
{
    readSolution(fileData, functionModeChoice);
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::readSolution(const char* fileC)
{
    cout << "BEGIN READ " << fileC << std::endl;

    string file = fileC;
    // Initialize NN netwoks CPU/GPU
    // read the tsp file into a NN (1D) and get the margin values.
    EMST_RW irw;
    irw.readFile(file, md_links_cpu, pMin, pMax);

    traceTSP.benchMark = file;
    cout << "TraceTSP.benchmark: " << traceTSP.benchMark << endl;


    initialize(md_links_cpu, pMin, pMax);
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::readSolution(string file,  int functionModeChoice)
{
    cout << "BEGIN READ " << file << std::endl;
    traceTSP.benchMark = file;

    // Initialize NN netwoks CPU/GPU
    // read the tsp file into a NN (1D) and get the margin values.
    EMST_RW irw;
    irw.readFile(file, md_links_cpu, pMin, pMax);

    traceTSP.benchMark = file;
    cout << "TraceTSP.benchmark: " << traceTSP.benchMark << endl;

    //    if(functionModeChoice < 7)
    //        irw.readOptPossibilites(functionModeChoice, md_links_cpu.nodeParentMap);//qiao add to read opt possibilities into nodeParentMap, size is not N
    //    else if(functionModeChoice >= 7)
    irw.readOptPossibilites(functionModeChoice, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);//qiao add to read opt possibilities into nodeParentMap, size is not N

    initialize(md_links_cpu, pMin, pMax);
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::writeSolution()
{
    writeSolution(fileSolution);
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::writeSolution(const char* file)
{
    cout << "BEGIN WRITE " << file << std::endl;

    // output write md_links_cpu.networklinks, its best tsp tour was written to md_links_cpu at the evaluation step
    md_links_cpu.writeLinks(file, traceTSP.benchMark);

    cout << "END WRITE" << std::endl;
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::openStatisticsFile()
{
    // Ouverture du fichier traitement en mode append
    OutputStream->open(fileStats, ios::app);
    if (!OutputStream->rdbuf()->is_open())
    {
        cerr << "Unable to open file " << fileStats << "CRITICAL ERROR" << endl;
        exit(-1);
    }
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::openStatisticsFile(string file)
{
    // Ouverture du fichier traitement en mode append

    file.append("_");

    file.append(traceTSP.benchMark);
    file.append(SFX_STATICS_TSP);

    OutputStream->open(file, ios::app);
    cout << "Write statics open file: " << file << endl;
    if (!OutputStream->rdbuf()->is_open())
    {
        cerr << "Unable to open file " << file << "CRITICAL ERROR" << endl;
        exit(-1);
    }
}


template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::closeStatisticsFile()
{
    OutputStream->close();
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::initStatisticsFile()
{
    time(&t0);
    x0 = clock();
#ifdef CUDA_CODE
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif
    openStatisticsFile();
    writeHeaderStatistics(*OutputStream);
    closeStatisticsFile();
}

//qiao add
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::initStatisticsFile(string file)
{
    cout << "Enter initialize static " << file << endl;

    openStatisticsFile(file);
    writeHeaderStatistics(*OutputStream, file);
    closeStatisticsFile();
}


template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::writeHeaderStatistics(std::ostream& o)
{
    ifstream fin(fileStats);
    string s;
    fin >> s;
    if(s.length() == 0)
    {
        o << "iteration" << setw(30)
          << "benchMark" << setw(30)
          << "length" << setw(30)
          << "timeObtainKoptimal" << setw(30)
          << "pdb" << setw(30)<< endl;

        //        o << endl;
    }
}

//qiao add
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::writeHeaderStatistics(std::ostream& o, string file)
{
    cout << "Enter write header " << endl;

    ifstream fin(file);
    string s;
    fin >> s;
    if(s.length() == 0)
    {
        o << "iteration" << setw(30)
          << "benchMark" << setw(30)
          << "length" << setw(30)
          << "timeObtainKoptimal" << setw(30)
          << "pdb" << setw(30)<< endl;

        //        o << endl;
    }
}


template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::writeStatistics(int iteration, std::ostream& o)
{

    cout << "Write statics iteration= " << iteration << endl;

    o << iteration << setw(30)
      << traceTSP.benchMark << setw(30)
      << traceTSP.length << setw(30)
      << traceTSP.timeObtainKoptimal << setw(30)
      << traceTSP.pdb << setw(30) << endl;

    o << endl << endl;
}
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::writeStatisticsToFile(int iteration)
{
    openStatisticsFile();
    writeStatistics(iteration, *OutputStream);
    closeStatisticsFile();
}

//qiao add to special filename
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::writeStatisticsToFile(int iteration, string file)
{
    openStatisticsFile(file);
    writeStatistics(iteration, *OutputStream);
    closeStatisticsFile();
}



template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::writeStatisticsToFile(string file)
{
    openStatisticsFile(file);
    writeStatistics(iteration, *OutputStream);
    closeStatisticsFile();
}


template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::writeStatisticsToFile()
{
    openStatisticsFile();
    writeStatistics(iteration, *OutputStream);
    closeStatisticsFile();
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::writeStatistics(std::ostream& o)
{
    openStatisticsFile();
    writeStatistics(iteration, o);
    closeStatisticsFile();
}



