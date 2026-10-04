#ifndef KOPT_OPERATORS_H
#define KOPT_OPERATORS_H
/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao, J.C. Créput
 * Creation date : Sep. 2016
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

//#include <cuda_runtime.h>
//#include <cuda.h>
//#include <helper_functions.h>
//#include <device_launch_parameters.h>
//#include <curand_kernel.h>
#include <device_atomic_functions.h>
#include <sm_60_atomic_functions.h>
#include <sm_61_intrinsics.h>

#include <vector>
#include <iterator>


#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <sm_60_atomic_functions.h>

#include "macros_cuda.h"
#include "ConfigParams.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "basic_operations.h"

#include "distance_functors.h"


//! reference EMST components
#include "NeuralNetKOPT.h"


#define EMST_BLOCK_SIZE 128

#define BLOCKSIZE 1024
#define GRIDSIZE 1048576 //2097152// 2147483646 // 1048576//(4opt) //16777216 //10485760 //2147483640 // 512000 (3opt) //20480 // 1024 for sw24978
#define SHAREDMAXCITIES 1979//734
#define OPTPOSSIBILITES4OPT 200
#define OPTPOSSIBILITES5OPT 2080
#define OPTPOSSIBILITES6OPT 23220

using namespace std;
using namespace components;

struct TspResultInfo
{
    GLfloat length = 0;
    GLint size = 0;
    GLfloat pdb = 0;
    GLfloat timeFlatten = 0;
    GLfloat timeTestTermination = 0;
    GLfloat timeFindNextClosest = 0;
    GLfloat timeFindMinPair = 0;
    GLfloat timeConnectGraphUnion = 0;
    GLfloat timeCumulativeFindNextClosest = 0;
    GLfloat timeCumulativeFindMinPair = 0;
    // wb.Q add
    GLfloat timeCumulativeConnetUnion = 0;
    GLfloat timeCumulativeFlatten = 0;
    GLfloat timeCumulativeTermination = 0;
    GLfloat timeConstructCellular = 0;
    GLfloat timeObtainKoptimal = 0;
    string benchMark = "";
    GLfloat optimumLength = 0;
    GLfloat maxtimeGpuOptSearch = 0;
};

//#include "MstOperator.h"

namespace operators
{


//! WB.Q add to return optimum value for PDB
float returnOptimal(string str){

    if(str == "qa194.tsp")
        return 9352;
    if(str == "ar9152.tsp")
        return 837479;
    if(str == "dj38.tsp")
        return 6656;
    if(str == "uy734.tsp")
        return 79114;
    else if(str == "zi929.tsp")
        return 95345;
    else if(str == "lu980.tsp")
        return 11340;
    else if(str == "rw1621.tsp")
        return 26051;
    if(str == "mu1979.tsp")
        return 86891;
    else if(str == "UY734.tsp")
        return 79114;
    else if(str == "ZI929.tsp")
        return 95345;
    else if(str == "nu3496.tsp")
        return 96132;
    else if(str == "ca4663.tsp")
        return 1290319;
    else if(str == "tz6117.tsp")
        return 394718;
    else if(str == "eg7146.tsp")
        return 172387;
    else if(str == "ym7663.tsp")
        return 238314;
    else if(str == "ei8246.tsp")
        return 206171;
    else if(str == "ja9847.tsp")
        return 491924;
    else if(str == "gr9882.tsp")
        return 300899;
    else if(str == "kz9976.tsp")
        return 1061882;
    else if(str == "fi10639.tsp")
        return 520527;
    else if(str == "mo14185.tsp")
        return 427377;
    else if(str == "ho14473.tsp")
        return 177105;
    else if(str == "it16862.tsp")
        return 557315;
    else if(str == "vm22775.tsp")
        return 569288;
    else if(str == "sw24978.tsp")
        return 855597;
    else if(str == "bm33708.tsp")
        return 959304;
    else if(str == "ch71009.tsp")
        return 4566563;
    else
        return -1;
}


//!QWB add to test changement in grid
template<class type>
int testGridNum(Grid<type> testGrid){

    int numTest = 0;
    for (int j = 0; j < testGrid.height; j++ )
        for (int i = 0; i < testGrid.width; i++)
        {
            if (testGrid[j][i])
                numTest += 1;// testGrid[j][i];
        }
    return numTest;

}

//! WB.Q add to count time
//! copy source code from https://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
void StartCounter(double& PCFreq, __int64& CounterStart)
{

    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
        cout << "QueryPerformanceFrequency failed!\n";

    PCFreq = double(li.QuadPart)/1000.0;// millisecond
    //        PCFreq = double(li.QuadPart); // s
    //    PCFreq = double(li.QuadPart)/1000000.0; // microsecond

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}
double GetCounter(double PCFreq, __int64 CounterStart)
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-CounterStart)/PCFreq;
}


//! WB.Q error check cuda synchronozation
bool errorCheckCudaThreadSynchronize(){
    cudaError_t err = cudaThreadSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit(1);
    }
    else
        return 0;
}

//! WB.Q error check cuda synchronozation
bool cudaChk(cudaError_t err){
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit(1);
    }
    else
        return 0;
}

/*!
 * \brief 191116 QWB: add compute distance with ordered array
 */
__device__ float dist(int i, int j, doubleLinkedEdgeForTSP* coords){
    //    float dx, dy;
    //    dx = coords[i].currentCoord[0] - coords[j].currentCoord[0];
    //    dy = coords[i].currentCoord[1] - coords[j].currentCoord[1];
    //    return (dx*dx + dy*dy);
    //     double dist = (dx*dx + dy*dy);
    //     return sqrt(dist);

    GLfloat dist = (coords[i].currentCoord - coords[j].currentCoord) * (coords[i].currentCoord - coords[j].currentCoord);
    return dist;

}


/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
KERNEL void K_2opt_oneThreadOne2opt_rockiSmall_shared(Grid<float> densityMap,
                                                      Grid<float> minRadiusMap,
                                                      Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                      unsigned long maxChecks,
                                                      unsigned int iter)
{

    int local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  densityMap.width; // each thread has this register


    if(local_id < maxChecks && threadIdx.x < width)
    {

        int i, j, id;

        __shared__ doubleLinkedEdgeForTSP sharedPoints[BLOCKSIZE]; // each sub-tour has the same length of blockDim.x

        for(int k = threadIdx.x; k < width; k+= blockDim.x)// step through all the points in list, but in blocks
        {
            sharedPoints[threadIdx.x] = arrayTSP[0][threadIdx.x];

            __syncthreads();

        }// end for copy shared

        for(unsigned int nu = 0; nu < iter; nu++)
        {
            id = local_id + nu * BLOCKSIZE * GRIDSIZE;

            if(id < maxChecks)
            {
                //WB.Q this way will produce i = j
                i = int(3 + sqrtf(8.0f * (float)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;
                if(j > 0  && i < width-1 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 )
                {

                    float optimization =  dist(j-1, j, sharedPoints) + dist(i-1, i, sharedPoints) - dist(j-1, i-1, sharedPoints) - dist(j, i, sharedPoints);

                    if(optimization > 0)
                    {
                        // here automic operation is necessary
                        int node1 = (int)sharedPoints[j-1].current;
                        int node3 = (int)sharedPoints[i-1].current;

                        float localMinChange = minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            atomicExch(&(densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation
                            atomicExch(&(minRadiusMap[0][node1]), optimization);
                        }

                    }

                }

            }

        }// for iter

    }
}// end K_2optOneThreadOne2opt



/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_rockiSmall(NeuralNetLinks<BufferDimension, Point> nn_source,
                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                               unsigned long maxChecks,
                                               unsigned int iter)
{

    int local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks){

        //        int packSize = blockDim.x * gridDim.x;
        int i, j, id;

        //        for(int nu = 0; nu <= iter; nu++)
        {

            id = local_id;// + nu * packSize;

            //            //qiao only for test
            //            if(id == maxChecks - 2)
            //                printf("Check inner GPU id == maxCheck2opt %d ", id); //correct

            //            if(id < maxChecks)
            {
                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0f * (float)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;

                //qiao only for test
                if(id == maxChecks-2)
                    printf("maximum 2-opt id id %d,  i, j: %d %d \n", id, i,j);


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                    float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                    if(newLength < oldLength)
                    {
                        float optimization = oldLength - newLength;
                        // here automic operation is necessary
                        int node1 = (int)arrayTSP[0][j-1].current;
                        int node3 = (int)arrayTSP[0][i-1].current;

                        float localMinChange = nn_source.minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation
                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadOne2opt




/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method work correctly
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_qiaoIterStride_best(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                        Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                        double max2optChecks, double maxChecksOptDivide,
                                                        double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < max2optChecks){

        double startId = maxChecksOptDivide * istride;

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < max2optChecks)
            {

                int i, j;
                id = trunc(id);

                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0 * (double)id + 1.0)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                    float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                    if(newLength < oldLength)
                    {
                        float optimization = oldLength - newLength;
                        // here automic operation is necessary
                        int node1 = (int)arrayTSP[0][j-1].current;
                        int node3 = (int)arrayTSP[0][i-1].current;

                        float localMinChange = nn_source.minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation
                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadIter2opt




/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method work correctly
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_qiaoIterStride_first(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                         Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                         double max2optChecks, double maxChecksOptDivide,
                                                         double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < max2optChecks){

        double startId = maxChecksOptDivide * istride;

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < max2optChecks)
            {

                int i, j;
                id = trunc(id);

                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0 * (double)id + 1.0)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    bool existingCandidate = 0;
                    if(nn_source.minRadiusMap[0][j-1] == 1 || nn_source.minRadiusMap[0][i-1] == 1)
                        existingCandidate = 1;

                    //                            if(j> 9145)//350631671)
                    //                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);

                    if(existingCandidate == 0)
                    {

                        float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                        float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                        if(newLength < oldLength)
                        {
                            // here automic operation is necessary
                            int node1 = (int)arrayTSP[0][j-1].current;
                            int node3 = (int)arrayTSP[0][i-1].current;


                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation

                            atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);

                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadIter2opt



/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method work correctly
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_qiaoIterStride(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                   Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                   double max2optChecks, double maxChecksOptDivide,
                                                   double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < max2optChecks){

        double startId = maxChecksOptDivide * istride;

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;



            if(id > 0 && id < max2optChecks)
            {

                int i, j;
                id = trunc(id);

                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0f * (double)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;

                //qiao only for test
                if(id == max2optChecks-2)
                    printf("maximum 2-opt id id %f,  i, j: %d %d \n", id, i,j);


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    bool existingCandidate = 0;
                    if(nn_source.minRadiusMap[0][j-1] == 1 || nn_source.minRadiusMap[0][i-1] == 1 )
                        existingCandidate = 1;

                    if(existingCandidate == 0)
                    {

                        float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                        float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);

                        if(newLength < oldLength)
                        {
                            // here automic operation is necessary
                            int node1 = (int)arrayTSP[0][j-1].current;
                            int node3 = (int)arrayTSP[0][i-1].current;

                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation

                        }

                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadIter2opt


/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method work correctly
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_qiaoIterStride_best_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                               double max2optChecks, double maxChecksOptDivide,
                                                               double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < max2optChecks){

        double startId = maxChecksOptDivide * istride;

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < max2optChecks)
            {

                int i, j;
                id = trunc(id);

                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0f * (double)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;

                //qiao only for test
                if(id == max2optChecks-2)
                    printf("maximum 2-opt id id %f,  i, j: %d %d \n", id, i,j);


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                    float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                    if(newLength < oldLength)
                    {
                        float optimization = oldLength - newLength;
                        // here automic operation is necessary
                        int node1 = (int)arrayTSP[0][j-1].current;
                        int node3 = (int)arrayTSP[0][i-1].current;

                        float localMinChange = nn_source.minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation
                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadIter2opt


/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_rockiSmall_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                    Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                    unsigned long maxChecks,
                                                    unsigned int iter)
{

    int local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks){

        int packSize = blockDim.x * gridDim.x;
        int i, j, id;

        for(int nu = 0; nu <= iter; nu++)
        {

            id = local_id + nu * packSize;

            //            //qiao only for test
            //            if(id == maxChecks - 2)
            //                printf("Check inner GPU id == maxCheck2opt %d ", id); //correct

            //            if(id < maxChecks)
            {
                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0f * (float)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;

                //qiao only for test
                if(id == maxChecks-2)
                    printf("maximum 2-opt id id %d,  i, j: %d %d \n", id, i,j);


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                    float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                    if(newLength < oldLength)
                    {
                        float optimization = oldLength - newLength;
                        // here automic operation is necessary
                        int node1 = (int)arrayTSP[0][j-1].current;
                        int node3 = (int)arrayTSP[0][i-1].current;

                        float localMinChange = nn_source.minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation
                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadOne2opt





/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_rockiSmall(NeuralNetLinks<BufferDimension, Point> nn_source,
                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                               double maxChecks2opt, double maxChecks4opt,
                                               unsigned int iter)
{

    double  id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(id < maxChecks4opt)
    {

        //        int packSize = blockDim.x * gridDim.x;
        double  outi, outj;

        //        id = trunc(id);

        if(id > maxChecks4opt-5)//350631671)
            printf("largeID id %f \n", id);
        else if (id <5)
            printf("SmallID id %f \n", id);

        {

            //            id = local_id;// + nu * packSize;

            //            if(id < maxChecks4opt)
            {
                //WB.Q this way will produce i = j
                outi = (3 + sqrt(8.0f * (double )id + 1.0f)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    int k = int(3 + sqrt(8.0f * (double )outi + 1.0f)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    int j = int(3 + sqrt(8.0f * (double )outj + 1.0f)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(k > width - 2)
                        //                            printf(" maximum 4-opt id = %d, outi= %d, outj=%d, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1 ||nn_source.minRadiusMap[0][k-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);
                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;


                                int optCandi = opt / 8;
                                //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength= dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0]) + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }
    }
    __syncthreads();
}// end K_4optOneThreadOne4opt


/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_qiaoIterStride(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                   Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                   double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                   double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        //        if(local_id == 0 )
        //            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;
            id = trunc(id);

            if(id > 0 && id < maxChecks4opt)
            {

                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                //                double test = startId + 0.99*maxChecks4optDivide ;
                //                if(startId > 0&& id > test)//350631671)
                //                    printf("largeID %f, local_id %f \n", id, local_id);
                //                else if (id <5)
                //                    printf("SmallID %f, local_id %f, outi %f, outj %f \n", id, local_id, outi, outj);
                //                else if (id <0)
                //                    printf("SmallID < 0 %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);
                //                if(id == (startId + maxChecks4optDivide-1))
                //                    printf("bound id  %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(w > 9144)
                        //                            printf(" maximum 4-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1 ||nn_source.minRadiusMap[0][k-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);
                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;


                                int optCandi = opt / 8;
                                //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength= dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0]) + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                //                                if(opti > 0 && opti > optimiz)
                                if(opti > 0)
                                {
                                    finalSelect = optCandi;
                                    //                                    optimiz = opti; //select the best of 200 re-connection scheme

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);

                                    break;

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt




/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_qiaoIterStride_Best(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                        Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                        double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                        double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        //        if(local_id == 0 )
        //            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;
            id = trunc(id);

            if(id > 0 && id < maxChecks4opt)
            {

                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(w > 9144)
                        //                            printf(" maximum 4-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);

                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);
                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;


                                int optCandi = opt / 8;
                                //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength= dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0]) + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;


                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;

                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {

                                    unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                    unsigned int node7 = (int)arrayTSP[0][k-1].current;


                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt




/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_qiaoIterStride_Best_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                               double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                               double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    //        __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES4OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    if(threadIdx.x < OPTPOSSIBILITES4OPT)
        optPossibilities[threadIdx.x] = nn_source.nodeParentMap[0][threadIdx.x ];

    __syncthreads();

    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        //        if(local_id == 0 )
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;


            if(id > 0 && id < maxChecks4opt)
            {

                id = trunc(id);
                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {


                        {

                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);

                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;


                                int optCandi = opt / 8;
                                newLength= dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP) + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;
                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;

                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {
                                    unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                    unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                    unsigned int node7 = (int)sharedArrayTSP[k-1].current;


                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt




/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_4opt_qiaoIterStride_Best_sharePossible(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                          Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                          double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                          double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ QWChar optPossibilities[OPTPOSSIBILITES4OPT];

    if(threadIdx.x < OPTPOSSIBILITES4OPT)
        optPossibilities[threadIdx.x] = nn_source.nodeParentMap[0][threadIdx.x ];

    __syncthreads();

    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        //        if(local_id == 0 )
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;


            if(id > 0 && id < maxChecks4opt)
            {

                id = trunc(id);
                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {


                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);

                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;


                                int optCandi = opt / 8;
                                newLength= dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0]) + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;
                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;

                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {
                                    unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                    unsigned int node7 = (int)arrayTSP[0][k-1].current;


                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt



/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_qiaoIterStride_sharedwithOccup(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                   Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                   double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                   double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register



    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];// if do not use this occupy, it takes longger time than using it on global mem
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES4OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    if(threadIdx.x < OPTPOSSIBILITES4OPT)
        optPossibilities[threadIdx.x] = nn_source.nodeParentMap[0][threadIdx.x ];

    __syncthreads();


    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        //        if(local_id == 0 )
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;
            id = trunc(id);

            if(id > 0 && id < maxChecks4opt)
            {

                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                //                double test = startId + 0.99*maxChecks4optDivide ;
                //                if(startId > 0&& id > test)//350631671)
                //                    printf("largeID %f, local_id %f \n", id, local_id);
                //                else if (id <5)
                //                    printf("SmallID %f, local_id %f, outi %f, outj %f \n", id, local_id, outi, outj);
                //                else if (id <0)
                //                    printf("SmallID < 0 %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);
                //                if(id == (startId + maxChecks4optDivide-1))
                //                    printf("bound id  %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(w > 1971)
                        //                            printf(" maximum 4-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);


                        bool existingCandidate = 0;
                        if(sharedArrayOccupied[w-1] == 1 || sharedArrayOccupied[j-1] == 1 ||sharedArrayOccupied[p-1] == 1 ||sharedArrayOccupied[k-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);
                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            //                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;



                                int optCandi = opt / 8;
                                //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength= dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP) + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                //                                if(opti > 0 && opti > optimiz)
                                if(opti > 0 )
                                {
                                    finalSelect = optCandi;


                                    sharedArrayOccupied[w-1]= 1;
                                    sharedArrayOccupied[j-1]= 1;
                                    sharedArrayOccupied[p-1]= 1;
                                    sharedArrayOccupied[k-1]=1;

                                    break;

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;

                                unsigned long long result = 0;
                                result = result | node3;
                                result = result << 16;
                                result = result | node5;
                                result = result << 16;
                                result = result | node7;

                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                codekopt = finalSelect * 100 + 4;

                                atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt



/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_qiaoIterStride_shared_noShareOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                        Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                        double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                        double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES4OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    if(threadIdx.x < OPTPOSSIBILITES4OPT)
        optPossibilities[threadIdx.x] = nn_source.nodeParentMap[0][threadIdx.x ];

    __syncthreads();


    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        //        if(local_id == 0 )
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;
            id = trunc(id);

            if(id > 0 && id < maxChecks4opt)
            {

                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                //                double test = startId + 0.99*maxChecks4optDivide ;
                //                if(startId > 0&& id > test)//350631671)
                //                    printf("largeID %f, local_id %f \n", id, local_id);
                //                else if (id <5)
                //                    printf("SmallID %f, local_id %f, outi %f, outj %f \n", id, local_id, outi, outj);
                //                else if (id <0)
                //                    printf("SmallID < 0 %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);
                //                if(id == (startId + maxChecks4optDivide-1))
                //                    printf("bound id  %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(w > 1971)
                        //                            printf(" maximum 4-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1 ||nn_source.minRadiusMap[0][k-1] == 1)
                            existingCandidate = 1;
                        //                        if(sharedArrayOccupied[w-1] == 1 || sharedArrayOccupied[j-1] == 1 ||sharedArrayOccupied[p-1] == 1 ||sharedArrayOccupied[k-1] == 1)
                        //                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);
                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            //                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;

                                int nd1 = (int)optPossibilities[opt] -1;
                                int nd2 = (int) optPossibilities[opt+1] -1;
                                int nd3 = (int)optPossibilities[opt+2] -1;
                                int nd4 = (int)optPossibilities[opt+3] -1;
                                int nd5 = (int)optPossibilities[opt+4] -1;
                                int nd6 = (int)optPossibilities[opt+5] -1;
                                int nd7 = (int)optPossibilities[opt+6] -1;
                                int nd8 = (int)optPossibilities[opt+7] -1;



                                int optCandi = opt / 8;
                                //                                printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength= dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP) + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                //                                if(opti > 0 && opti > optimiz)
                                if(opti > 0)
                                {
                                    finalSelect = optCandi;
                                    //                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);


                                    break;
                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;




                                unsigned long long result = 0;
                                result = result | node3;
                                result = result << 16;
                                result = result | node5;
                                result = result << 16;
                                result = result | node7;

                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                codekopt = finalSelect * 100 + 4;

                                //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange


                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt






/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_rockiSmall(NeuralNetLinks<BufferDimension, Point> nn_source,
                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                               int idRow5th, double maxChecks4opt, double maxChecks2opt,
                                               unsigned int iter)
{

    double id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(id < maxChecks4opt)
    {

        double outi, outj;

        //WB.Q this way will produce i = j
        outi = int(3 + sqrt(8.0f * (double)id + 1.0f)) / 2 ;
        outj = id - (outi-2)*(outi-1)/2 + 1;

        if(outi < maxChecks2opt && outj < maxChecks2opt)
        {
            int k = int(3 + sqrt(8.0f * (double)outi + 1.0f)) / 2 ;
            int p = outi - (k-2)*(k-1)/2 + 1;

            int j = int(3 + sqrt(8.0f * (double)outj + 1.0f)) / 2 ;
            int w = outj - (j-2)*(j-1)/2 + 1;

            if(id == maxChecks4opt-2)
                printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

            if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
            {

                //                        if(idRow5th > width -2 && k > width - 2)
                //                            printf(" maximum 4-opt id = %d, outi= %d, outj=%d, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);

                bool existingCandidate = 0;
                if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                        ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                    existingCandidate = 1;

                if(existingCandidate == 0)
                {

                    float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]) + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                    float newLength;
                    int array[10];
                    array[0] = w-1;
                    array[1] = w;
                    array[2] = j-1;
                    array[3] = j;
                    array[4] = p-1;
                    array[5] = p;
                    array[6] = k-1;
                    array[7] = k;
                    array[8] = idRow5th-1;
                    array[9] = idRow5th;

                    int finalSelect = -1;
                    float optimiz = -INFINITY;

                    for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                    {
                        int nd1 = nn_source.nodeParentMap[0][opt] -1;
                        int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                        int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                        int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                        int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                        int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                        int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                        int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                        int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                        int nd10 = nn_source.nodeParentMap[0][opt+9] -1;


                        int optCandi = opt / 10;
                        // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                        newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                        float opti = oldLength - newLength;
                        if(opti > 0 && opti > optimiz)
                        {
                            finalSelect = optCandi;
                            optimiz = opti;

                            atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                        }
                    }

                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (int)arrayTSP[0][w-1].current;
                        unsigned int node3 = (int)arrayTSP[0][j-1].current;
                        unsigned int node5 = (int)arrayTSP[0][p-1].current;
                        unsigned int node7 = (int)arrayTSP[0][k-1].current;
                        unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                        //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                        //                            if(optimiz > localMinChange)
                        {

                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;
                            result = result << 16;
                            result = result | node9;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 5;

                            //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                            //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                        }
                    }
                }

            }
        }



    }
    __syncthreads();
}// end K_2optOneThreadOne2opt




/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_rockiSmall_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                    Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                    int idRow5th, double maxChecks4opt,
                                                    double maxChecks2opt,
                                                    unsigned int iter)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {

        int packSize = blockDim.x * gridDim.x;

        for(int nu = 0; nu <= iter; nu++)
        {

            double id = local_id + nu * packSize;

            if(id < maxChecks4opt)
            {

                double outi, outj;

                //WB.Q this way will produce i = j
                outi = int(3 + sqrt(8.0f * (double)id + 1.0f)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    int k = int(3 + sqrt(8.0f * (double)outi + 1.0f)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    int j = int(3 + sqrt(8.0f * (double)outj + 1.0f)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id == maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(idRow5th > width -2 && k > width - 2)
                        //                            printf(" maximum 4-opt id = %d, outi= %d, outj=%d, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);

                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]) + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;
                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;
                                unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }
                }


            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadOne2opt



/*!
 * \brief 2409 QWB: add parallel 5opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                     Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                     int idRow5th, double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                     double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;


            //            printf("id %f, local_id %f \n", id, local_id);



            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id > maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        if(w > 1965)
                            printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]) + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;
                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;
                                unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }
                }


            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt





/*!
 * \brief 2409 QWB: add parallel 5opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                 Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                 int idRow5th, double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                 double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;


            //            printf("id %f, local_id %f \n", id, local_id);

            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id > maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        if(w > 1965)
                            printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                        float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]) + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                        float newLength;
                        int array[10];
                        array[0] = w-1;
                        array[1] = w;
                        array[2] = j-1;
                        array[3] = j;
                        array[4] = p-1;
                        array[5] = p;
                        array[6] = k-1;
                        array[7] = k;
                        array[8] = idRow5th-1;
                        array[9] = idRow5th;

                        int finalSelect = -1;

                        for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                        {
                            int nd1 = nn_source.nodeParentMap[0][opt] -1;
                            int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                            int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                            int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                            int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                            int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                            int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                            int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                            int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                            int nd10 = nn_source.nodeParentMap[0][opt+9] -1;


                            int optCandi = opt / 10;
                            // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                            newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                    + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                            float opti = oldLength - newLength;
                            if(opti > 0)
                            {
                                finalSelect = optCandi;
                                break; // stop search when meet the first 5-opt of these 5 edges

                            }
                        }

                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (int)arrayTSP[0][w-1].current;
                            unsigned int node3 = (int)arrayTSP[0][j-1].current;
                            unsigned int node5 = (int)arrayTSP[0][p-1].current;
                            unsigned int node7 = (int)arrayTSP[0][k-1].current;
                            unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;



                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;
                            result = result << 16;
                            result = result | node9;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 5;

                            //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                            //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange


                        }

                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt



/*!
 * \brief 2409 QWB: add parallel 5opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                        Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                        int idRow5th, double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                        double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register


    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];

    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nodeParentMap[0][m];

        __syncthreads();

    }

    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;


            //            printf("id %f, local_id %f \n", id, local_id);

            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id > maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        if(w > 1965)
                            printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                        float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP) + dist(idRow5th-1, idRow5th, sharedArrayTSP);

                        float newLength;
                        int array[10];
                        array[0] = w-1;
                        array[1] = w;
                        array[2] = j-1;
                        array[3] = j;
                        array[4] = p-1;
                        array[5] = p;
                        array[6] = k-1;
                        array[7] = k;
                        array[8] = idRow5th-1;
                        array[9] = idRow5th;

                        int finalSelect = -1;

                        for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                        {
                            //                            int nd1 = nn_source.nodeParentMap[0][opt] -1;
                            //                            int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                            //                            int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                            //                            int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                            //                            int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                            //                            int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                            //                            int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                            //                            int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                            //                            int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                            //                            int nd10 = nn_source.nodeParentMap[0][opt+9] -1;

                            int nd1 = optPossibilities[opt] -1;
                            int nd2 = optPossibilities[opt+1] -1;
                            int nd3 = optPossibilities[opt+2] -1;
                            int nd4 = optPossibilities[opt+3] -1;
                            int nd5 = optPossibilities[opt+4] -1;
                            int nd6 = optPossibilities[opt+5] -1;
                            int nd7 = optPossibilities[opt+6] -1;
                            int nd8 = optPossibilities[opt+7] -1;
                            int nd9 = optPossibilities[opt+8] -1;
                            int nd10 = optPossibilities[opt+9] -1;


                            int optCandi = opt / 10;
                            // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);

                            newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                    + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);


                            float opti = oldLength - newLength;
                            if(opti > 0)
                            {
                                finalSelect = optCandi;
                                break; // stop search when meet the first 5-opt of these 5 edges

                            }
                        }

                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                            unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                            unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                            unsigned int node7 = (int)sharedArrayTSP[k-1].current;
                            unsigned int node9 = (int)sharedArrayTSP[idRow5th-1].current;



                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;
                            result = result << 16;
                            result = result | node9;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 5;

                            //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                            //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange


                        }

                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt



/*!
 * \brief 2409 QWB: add parallel 5opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_qiao_stride_iter_SelectBest_iterN(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                     Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                     double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                     double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                id = trunc(id);
                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(k > p && p> j&& j>w && k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        float oldLength_4 = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);// + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                        float newLength;
                        int array[10];
                        array[0] = w-1;
                        array[1] = w;
                        array[2] = j-1;
                        array[3] = j;
                        array[4] = p-1;
                        array[5] = p;
                        array[6] = k-1;
                        array[7] = k;

                        for (int idRow5th = k+2; idRow5th < width; idRow5th ++)
                        {

                            float oldLength = oldLength_4;

                            oldLength += dist(idRow5th-1, idRow5th, arrayTSP[0]);

                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {

                                int nd1 = nn_source.nVisitedMap[0][opt] -1;
                                int nd2 = nn_source.nVisitedMap[0][opt+1] -1;
                                int nd3 = nn_source.nVisitedMap[0][opt+2] -1;
                                int nd4 = nn_source.nVisitedMap[0][opt+3] -1;
                                int nd5 = nn_source.nVisitedMap[0][opt+4] -1;
                                int nd6 = nn_source.nVisitedMap[0][opt+5] -1;
                                int nd7 = nn_source.nVisitedMap[0][opt+6] -1;
                                int nd8 = nn_source.nVisitedMap[0][opt+7] -1;
                                int nd9 = nn_source.nVisitedMap[0][opt+8] -1;
                                int nd10 = nn_source.nVisitedMap[0][opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;
                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {
                                    unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                    unsigned int node7 = (int)arrayTSP[0][k-1].current;
                                    unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //   printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //        node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);

                                }
                            }

                        }

                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt




/*!
 * \brief 2409 QWB: add parallel 5opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_qiao_stride_iter_SelectBest_iterN_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                            Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                            double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                            double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register

    //    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nVisitedMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                id = trunc(id);
                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if(k > p && p> j&& j>w && k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        float oldLength_4 = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);// + dist(idRow5th-1, idRow5th, sharedArrayTSP);

                        float newLength;
                        int array[10];
                        array[0] = w-1;
                        array[1] = w;
                        array[2] = j-1;
                        array[3] = j;
                        array[4] = p-1;
                        array[5] = p;
                        array[6] = k-1;
                        array[7] = k;

                        for (int idRow5th = k+2; idRow5th < width; idRow5th ++)
                        {

                            float oldLength = oldLength_4;

                            oldLength += dist(idRow5th-1, idRow5th, sharedArrayTSP);

                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -1;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;
                                int nd9 = optPossibilities[opt+8] -1;
                                int nd10 = optPossibilities[opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                        + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;
                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {
                                    unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                    unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                    unsigned int node7 = (int)sharedArrayTSP[k-1].current;
                                    unsigned int node9 = (int)sharedArrayTSP[idRow5th-1].current;

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //   printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //        node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);

                                }
                            }

                        }

                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt



/*!
 * \brief 2409 QWB: add parallel 5opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_qiao_stride_iter_SelectBest_sharePossible(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                             Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                             double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                             double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register

    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nVisitedMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                id = trunc(id);
                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if(k > p && p> j&& j>w && k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        float oldLength_4 = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);// + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                        float newLength;
                        int array[10];
                        array[0] = w-1;
                        array[1] = w;
                        array[2] = j-1;
                        array[3] = j;
                        array[4] = p-1;
                        array[5] = p;
                        array[6] = k-1;
                        array[7] = k;

                        for (int idRow5th = k+2; idRow5th < width; idRow5th ++)
                        {

                            float oldLength = oldLength_4;

                            oldLength += dist(idRow5th-1, idRow5th, arrayTSP[0]);

                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -1;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;
                                int nd9 = optPossibilities[opt+8] -1;
                                int nd10 = optPossibilities[opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;
                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {
                                    unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                    unsigned int node7 = (int)arrayTSP[0][k-1].current;
                                    unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //   printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //        node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);

                                }
                            }

                        }

                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt



/*!
 * \brief 2409 QWB: add parallel 5opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterN(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                       Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                       double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                       double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            //            printf("id %f, local_id %f \n", id, local_id);

            if(id > 0 && id < maxChecks4opt)
            {

                id = trunc(id);
                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    // if(id > maxChecks4opt-2)
                    //     printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if(k > p && p> j&& j>w && k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        bool existingCandidate = 0;

                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1)  // ||nn_source.minRadiusMap[0][idRow5th-1] == 1
                            existingCandidate = 1;


                        if(existingCandidate == 0)
                        {

                            float oldLength_4 = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);// + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            for (int idRow5th = k+2; idRow5th < width; idRow5th ++)
                            {

                                //                                if(idRow5th ==  width -1)
                                //                                    printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                                if(nn_source.minRadiusMap[0][idRow5th-1] == 1)
                                    continue;


                                float oldLength = oldLength_4;

                                oldLength += dist(idRow5th-1, idRow5th, arrayTSP[0]);

                                array[8] = idRow5th-1;
                                array[9] = idRow5th;

                                int finalSelect = -1;

                                for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                                {
                                    //                                    int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                    //                                    int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                    //                                    int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                    //                                    int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                    //                                    int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                    //                                    int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                    //                                    int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                    //                                    int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                    //                                    int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                    //                                    int nd10 = nn_source.nodeParentMap[0][opt+9] -1;

                                    int nd1 = nn_source.nVisitedMap[0][opt] -1;
                                    int nd2 = nn_source.nVisitedMap[0][opt+1] -1;
                                    int nd3 = nn_source.nVisitedMap[0][opt+2] -1;
                                    int nd4 = nn_source.nVisitedMap[0][opt+3] -1;
                                    int nd5 = nn_source.nVisitedMap[0][opt+4] -1;
                                    int nd6 = nn_source.nVisitedMap[0][opt+5] -1;
                                    int nd7 = nn_source.nVisitedMap[0][opt+6] -1;
                                    int nd8 = nn_source.nVisitedMap[0][opt+7] -1;
                                    int nd9 = nn_source.nVisitedMap[0][opt+8] -1;
                                    int nd10 = nn_source.nVisitedMap[0][opt+9] -1;


                                    int optCandi = opt / 10;
                                    // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                    newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                            + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                                    float opti = oldLength - newLength;
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;

                                        atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                        break; // stop optpossibilities search when meet the first 5-opt of these 5 edges
                                    }
                                }

                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                    unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                    unsigned int node7 = (int)arrayTSP[0][k-1].current;
                                    unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //   printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //        node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    break; // stop row5th loop
                                }

                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt



/*!
 * \brief 2409 QWB: add parallel 5opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterN_shared_noShareOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                                            Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                                            double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                                            double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register

    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            //            optPossibilities[m] = nn_source.nodeParentMap[0][m];
            optPossibilities[m] = nn_source.nVisitedMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            // printf("id %f, local_id %f \n", id, local_id);

            if(id > 0 && id < maxChecks4opt)
            {

                id = trunc(id);
                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    //                    if(id > maxChecks4opt-2)
                    //                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if(k > p && p> j&& j>w && k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        bool existingCandidate = 0;

                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1)  // ||nn_source.minRadiusMap[0][idRow5th-1] == 1
                            existingCandidate = 1;


                        if(existingCandidate == 0)
                        {

                            float oldLength_4 = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);// + dist(idRow5th-1, idRow5th, sharedArrayTSP);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            for (int idRow5th = k+2; idRow5th < width; idRow5th ++)
                            {

                                //                                if(idRow5th ==  width -1)
                                //                                    printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                                if(nn_source.minRadiusMap[0][idRow5th-1] == 1)
                                    continue;

                                float oldLength = oldLength_4;

                                oldLength += dist(idRow5th-1, idRow5th, sharedArrayTSP);

                                array[8] = idRow5th-1;
                                array[9] = idRow5th;

                                int finalSelect = -1;

                                for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                                {
                                    //                                    int nd1 =  nn_source.nodeParentMap[0][opt] -1;
                                    //                                    int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                    //                                    int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                    //                                    int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                    //                                    int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                    //                                    int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                    //                                    int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                    //                                    int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                    //                                    int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                    //                                    int nd10 = nn_source.nodeParentMap[0][opt+9] -1;

                                    //                                    int nd1 =  nn_source.nVisitedMap[0][opt] -1;
                                    //                                    int nd2 = nn_source.nVisitedMap[0][opt+1] -1;
                                    //                                    int nd3 = nn_source.nVisitedMap[0][opt+2] -1;
                                    //                                    int nd4 = nn_source.nVisitedMap[0][opt+3] -1;
                                    //                                    int nd5 = nn_source.nVisitedMap[0][opt+4] -1;
                                    //                                    int nd6 = nn_source.nVisitedMap[0][opt+5] -1;
                                    //                                    int nd7 = nn_source.nVisitedMap[0][opt+6] -1;
                                    //                                    int nd8 = nn_source.nVisitedMap[0][opt+7] -1;
                                    //                                    int nd9 = nn_source.nVisitedMap[0][opt+8] -1;
                                    //                                    int nd10 = nn_source.nVisitedMap[0][opt+9] -1;

                                    int nd1 = optPossibilities[opt] -1;
                                    int nd2 = optPossibilities[opt+1] -1;
                                    int nd3 = optPossibilities[opt+2] -1;
                                    int nd4 = optPossibilities[opt+3] -1;
                                    int nd5 = optPossibilities[opt+4] -1;
                                    int nd6 = optPossibilities[opt+5] -1;
                                    int nd7 = optPossibilities[opt+6] -1;
                                    int nd8 = optPossibilities[opt+7] -1;
                                    int nd9 = optPossibilities[opt+8] -1;
                                    int nd10 = optPossibilities[opt+9] -1;


                                    int optCandi = opt / 10;
                                    //                                     printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                    newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                            + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                    float opti = oldLength - newLength;
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;

                                        atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                        break; // stop optpossibilities search when meet the first 5-opt of these 5 edges

                                    }
                                }

                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (unsigned int)sharedArrayTSP[w-1].current;
                                    unsigned int node3 = (unsigned int)sharedArrayTSP[j-1].current;
                                    unsigned int node5 = (unsigned int)sharedArrayTSP[p-1].current;
                                    unsigned int node7 = (unsigned int)sharedArrayTSP[k-1].current;
                                    unsigned int node9 = (unsigned int)sharedArrayTSP[idRow5th-1].current;

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    break; // stop row5th loop
                                }

                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

}// end K_5optOneThreadOne5opt




/*!
 * \brief 2409 QWB: add parallel 5opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterNShared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                             Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                             double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                             double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register

    //    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            //            optPossibilities[m] = nn_source.nodeParentMap[0][m];
            optPossibilities[m] = nn_source.nVisitedMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;


            //  printf("id %f, local_id %f \n", id, local_id);

            if(id > 0 && id < maxChecks4opt)
            {

                id = trunc(id);
                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    //                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    if(k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        bool existingCandidate = 0;

                        //                        if(sharedArrayOccupied[w-1] == 1 || sharedArrayOccupied[j-1] == 1 ||sharedArrayOccupied[p-1] == 1
                        //                                ||sharedArrayOccupied[k-1] == 1 ) //||sharedArrayOccupied[idRow5th-1] == 1)
                        //                            existingCandidate = 1;
                        //                        if(existingCandidate == 0)
                        {


                            float oldLength_4 = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            for (int idRow5th = k+2; idRow5th < width; idRow5th ++)
                            {


                                //                                if(sharedArrayOccupied[idRow5th-1] == 1)
                                //                                    continue;


                                //                            if(idRow5th == width - 1)
                                //                                printf(" maximum row 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                                //                            if(id > maxChecks4opt-2)
                                //                                printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                                float oldLength =  oldLength_4;
                                oldLength += dist(idRow5th-1, idRow5th, sharedArrayTSP);

                                array[8] = idRow5th-1;
                                array[9] = idRow5th;

                                int finalSelect = -1;

                                for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                                {
                                    //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                    //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                    //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                    //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                    //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                    //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                    //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                    //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                    //                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                    //                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;

                                    int nd1 = optPossibilities[opt] -1;
                                    int nd2 = optPossibilities[opt+1] -1;
                                    int nd3 = optPossibilities[opt+2] -1;
                                    int nd4 = optPossibilities[opt+3] -1;
                                    int nd5 = optPossibilities[opt+4] -1;
                                    int nd6 = optPossibilities[opt+5] -1;
                                    int nd7 = optPossibilities[opt+6] -1;
                                    int nd8 = optPossibilities[opt+7] -1;
                                    int nd9 = optPossibilities[opt+8] -1;
                                    int nd10 = optPossibilities[opt+9] -1;


                                    int optCandi = opt / 10;
                                    // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                    newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                            + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                    float opti = oldLength - newLength;
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;

                                        //                                        atomicExch(&(sharedArrayOccupied[w-1]), 1);
                                        //                                        atomicExch(&(sharedArrayOccupied[j-1]), 1);
                                        //                                        atomicExch(&(sharedArrayOccupied[p-1]), 1);
                                        //                                        atomicExch(&(sharedArrayOccupied[k-1]), 1);
                                        //                                        atomicExch(&(sharedArrayOccupied[idRow5th-1]), 1);

                                        //                                        sharedArrayOccupied[w-1]= 1;
                                        //                                        sharedArrayOccupied[j-1]= 1;
                                        //                                        sharedArrayOccupied[p-1]= 1;
                                        //                                        sharedArrayOccupied[k-1]= 1;
                                        //                                        sharedArrayOccupied[idRow5th-1]= 1;


                                        break; // stop optpossibilities search when meet the first 5-opt of these 5 edges

                                    }
                                }

                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (unsigned int)sharedArrayTSP[w-1].current;
                                    unsigned int node3 = (unsigned int)sharedArrayTSP[j-1].current;
                                    unsigned int node5 = (unsigned int)sharedArrayTSP[p-1].current;
                                    unsigned int node7 = (unsigned int)sharedArrayTSP[k-1].current;
                                    unsigned int node9 = (unsigned int)sharedArrayTSP[idRow5th-1].current;


                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    break; // stop row5th loop
                                }

                            }


                        }


                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt



/*!
 * \brief 2409 QWB: add parallel 5opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterNSharedOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                                   Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                                   double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                                   double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register


    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nodeParentMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;


            //            printf("id %f, local_id %f \n", id, local_id);

            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    //                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    if(k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {
                        float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);

                        float newLength;
                        int array[10];
                        array[0] = w-1;
                        array[1] = w;
                        array[2] = j-1;
                        array[3] = j;
                        array[4] = p-1;
                        array[5] = p;
                        array[6] = k-1;
                        array[7] = k;

                        for (int idRow5th = k+2; idRow5th < width; idRow5th ++)
                        {

                            //                            if(idRow5th == width - 1)
                            //                                printf(" maximum row 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                            //                            if(id > maxChecks4opt-2)
                            //                                printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                            oldLength += dist(idRow5th-1, idRow5th, sharedArrayTSP);

                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                //                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                //                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;
                                int nd9 = optPossibilities[opt+8] -1;
                                int nd10 = optPossibilities[opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                        + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0)
                                {
                                    finalSelect = optCandi;
                                    break; // stop optpossibilities search when meet the first 5-opt of these 5 edges

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;
                                unsigned int node9 = (int)sharedArrayTSP[idRow5th-1].current;


                                unsigned long long result = 0;
                                result = result | node3;
                                result = result << 16;
                                result = result | node5;
                                result = result << 16;
                                result = result | node7;
                                result = result << 16;
                                result = result | node9;

                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                codekopt = finalSelect * 100 + 5;

                                //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                break; // stop row5th loop
                            }

                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt



/*!
 * \brief 2409 QWB: add parallel 5opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterNSharedOutPossibleLoop(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                                            Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                                            double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                                            double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width = nn_source.adaptiveMap.width; // each thread has this register


    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            //            optPossibilities[m] = nn_source.nodeParentMap[0][m];
            optPossibilities[m] = nn_source.nVisitedMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            //            printf("id %f, local_id %f \n", id, local_id);

            if(id > 0 && id < maxChecks4opt)
            {

                id = trunc(id);
                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {
                        bool existingCandidate = 0;

                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1)  // ||nn_source.minRadiusMap[0][idRow5th-1] == 1
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9, nd10;
                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);//+ dist(idRow5th-1, idRow5th, sharedArrayTSP);

                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                //                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                //                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;

                                nd1 = optPossibilities[opt] -1;
                                nd2 = optPossibilities[opt+1] -1;
                                nd3 = optPossibilities[opt+2] -1;
                                nd4 = optPossibilities[opt+3] -1;
                                nd5 = optPossibilities[opt+4] -1;
                                nd6 = optPossibilities[opt+5] -1;
                                nd7 = optPossibilities[opt+6] -1;
                                nd8 = optPossibilities[opt+7] -1;
                                nd9 = optPossibilities[opt+8] -1;
                                nd10 = optPossibilities[opt+9] -1;

                                unsigned int node9;
                                int optCandi = opt / 10;

                                for (int idRow5th = k+2; idRow5th < width; idRow5th ++)
                                {

                                    if(nn_source.minRadiusMap[0][idRow5th-1] == 1)
                                        continue;

                                    oldLength += dist(idRow5th-1, idRow5th, sharedArrayTSP);
                                    array[8] = idRow5th-1;
                                    array[9] = idRow5th;

                                    // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                    float newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                            + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                    float opti = oldLength - newLength;
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;
                                        //
                                        node9 = (int)sharedArrayTSP[idRow5th-1].current;

                                        atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);


                                        break; // stop optpossibilities search when meet the first 5-opt of these 5 edges

                                    }

                                }

                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (unsigned int)sharedArrayTSP[w-1].current;
                                    unsigned int node3 = (unsigned int)sharedArrayTSP[j-1].current;
                                    unsigned int node5 = (unsigned int)sharedArrayTSP[p-1].current;
                                    unsigned int node7 = (unsigned int)sharedArrayTSP[k-1].current;
                                    //                               unsigned int node9 = (int)sharedArrayTSP[idRow5th-1].current;


                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    break; // stop row5th loop
                                }

                            }

                        }


                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt





/*!
 * \brief 2409 QWB: add parallel 5opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                            Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                            int idRow5th, double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                            double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register


    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nodeParentMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id > maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(w > 9144)
                        //                        printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                        bool existingCandidate = 0;

                        //                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                        //                                ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                        //                            existingCandidate = 1;

                        if(sharedArrayOccupied[w-1] == 1 || sharedArrayOccupied[j-1] == 1 ||sharedArrayOccupied[p-1] == 1
                                ||sharedArrayOccupied[k-1] == 1 ||sharedArrayOccupied[idRow5th-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP) + dist(idRow5th-1, idRow5th, sharedArrayTSP);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;
                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                //                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                //                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;
                                int nd9 = optPossibilities[opt+8] -1;
                                int nd10 = optPossibilities[opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                        + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                    atomicExch(&(sharedArrayOccupied[w-1]), 1);
                                    atomicExch(&(sharedArrayOccupied[j-1]), 1);
                                    atomicExch(&(sharedArrayOccupied[p-1]), 1);
                                    atomicExch(&(sharedArrayOccupied[k-1]), 1);
                                    atomicExch(&(sharedArrayOccupied[idRow5th-1]), 1);

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;
                                unsigned int node9 = (int)sharedArrayTSP[idRow5th-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }
                }


            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt




/*!
 * \brief 2409 QWB: add parallel 5opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory totally 32068bytes shared mem,for 1979 cities and 5-opt possibilities
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_shared_noOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                     Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                     int idRow5th, double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                     double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES]; // 23748bytes = 3*1979*float

    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT]; //8320bytes = 2080floats
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nodeParentMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0)
            printf("GPU StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id > maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(w > 9144)
                        //                        printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                        bool existingCandidate = 0;

                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP) + dist(idRow5th-1, idRow5th, sharedArrayTSP);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;
                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                //                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                //                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;
                                int nd9 = optPossibilities[opt+8] -1;
                                int nd10 = optPossibilities[opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                        + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;
                                unsigned int node9 = (int)sharedArrayTSP[idRow5th-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }
                }


            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt



/*!
 * \brief 2409 QWB: add parallel 5opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_shared_onlyPossibility(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                            Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                            int idRow5th, double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                            double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register


    //    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    //    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];
    //    float iterShared = (float)width / (float)BLOCKSIZE;
    //    for(int opt = 0; opt < iterShared; opt++)
    //    {
    //        int m = threadIdx.x + opt*BLOCKSIZE;
    //        if(m < width)
    //        {
    //            sharedArrayTSP[m].current = arrayTSP[0][m].current;
    //            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
    //            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

    //        }
    //        __syncthreads();
    //    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nodeParentMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id > maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        if(w > 9144)
                            printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                        bool existingCandidate = 0;

                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]) + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;
                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;
                                int nd9 = optPossibilities[opt+8] -1;
                                int nd10 = optPossibilities[opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;
                                unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }
                }


            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt





/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method find the best from one node
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_findBest(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                        Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                        double maxChecks3opt,
                                                        unsigned int iter)
{

    double id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    //    if(id == 0 )
    //        printf(" 0 maxium  %lld , maxim blockId = %lld , maxim blockId = %d, blockDim.x = %d \n ", maxChecks3opt, maxChecks3opt / 1024,  blockIdx.x , blockDim.x);

    //    if(blockIdx.x > maxChecks3opt / 1024 - 4 )
    //        printf(" block maxium  %lld ,  blockId = %d, blockDim.x = %d \n ", maxChecks3opt,  blockIdx.x , blockDim.x);

    if(id < maxChecks3opt)
    {

        //        int packSize = blockDim.x * gridDim.x;
        int row, i, j;
        double subtriplicate = (1.0)/3;
        //        for(int nu = 0; nu <= iter; nu++)
        {

            //             id = local_id;  + nu * packSize;

            //            if(id < maxChecks3opt)
            {

                //WB.Q this way will produce i = j
                double idid = 9*id*id;
                double idMul3 = 3*id;
                double rowN0 = idMul3 + sqrt(idid - (1.0)/9);
                double rowN1 = pow(rowN0, subtriplicate);
                double rowN2 = idMul3 - sqrt(idid - (1.0)/9);
                float rowN3 = pow(rowN2, subtriplicate);
                float rowN4 = rowN1 + rowN3 + 1;

                row = int(rowN4);// check which one works


                //qiao only for test
                if(id >maxChecks3opt - 2)
                    printf("3-opt maxmum id %d,  row %d, i %d, j %d \n", id, row, i,j);


                if(row < width)
                {
                    double id2opt = (row-1)*(row)*(row+1)/6 - id;

                    //WB.Q this way will produce i = j
                    i = int(3 + sqrt(8.0 * (double)id2opt + 1.0)) / 2 ;
                    j = id2opt - (i-2)*(i-1)/2 + 1;


                    if(i<row && i!=row &&i+1!= row &&j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                    {

                        //qiao only for test
                        if(row > width - 2)
                            printf("3-opt maxmum row id %d, row %d, i %d, j %d \n", id, row, i,j);

                        double newLength[4];

                        double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                        newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                        newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                        newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                        newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                        int finalSelect = -1;
                        double optimiz = -INFINITY;
                        for(int i = 0; i < 4; i++)
                        {
                            float opti = oldLength - newLength[i];
                            if(opti > 0 && opti > optimiz)
                            {
                                finalSelect = i;
                                optimiz = opti;

                                //                               if(blockIdx.x == 124698010  ||blockIdx.x == 124698013   || blockIdx.x == 0)
                                //                                printf("3opt GPU blockIdx.x=%d, selec %d, order %d, %d, %d, oldLength %f, newi %f, opti %f, optimiz %f ; node135 %d,%d,%d\n",blockIdx.x, finalSelect,
                                //                                       nn_source.grayValueMap[0][j-1], nn_source.grayValueMap[0][i-1], nn_source.grayValueMap[0][row-1]
                                //                                        , oldLength, newLength[i], opti, optimiz, j-1, i-1, row-1);


                            }
                        }

                        if(finalSelect >= 0)
                        {
                            float optimization = oldLength - newLength[finalSelect];
                            // here automic operation is necessary

                            unsigned int node1 = (int)arrayTSP[0][j-1].current;
                            unsigned int node3 = (int)arrayTSP[0][i-1].current;
                            unsigned int node5 = (int)arrayTSP[0][row-1].current;

                            //                            printf("3opt GPU mode %d, order %d, %d, %d, oldLength %f, new1 %f, new2 %f, new3 %f, new4 %f; node135 %d,%d,%d \n", finalSelect, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5]
                            //                                    , oldLength, newLength[0],newLength[1], newLength[2], newLength[3], node1, node3, node5);

                            float localMinChange = nn_source.minRadiusMap[0][node1];

                            if(optimization > localMinChange)
                            {

                                unsigned long long result = 0;
                                result = result | node3;
                                result = result << 16;
                                result = result | node5;

                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                codekopt = finalSelect * 100 + 3;

                                atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                            }
                        }
                    }//end if i j

                }//end if row < width

            }
        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt





/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterBest(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                        Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                        double maxChecks3opt,
                                                        double iter)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks3opt)
    {
        //qiao only for test
        if(local_id == 0)
            printf("local_id %f \n", local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {


            if(id > 0 && id < maxChecks3opt)
            {

                int row, i, j;
                double subtriplicate = (1.0)/3;


                //                if(id > 350631671)
                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row < width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            if(j> 9145)//350631671)
                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);

                            double newLength[4];

                            double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                            newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                            newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                            newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                            newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                            int finalSelect = -1;
                            double optimiz = -INFINITY;
                            for(int i = 0; i < 4; i++)
                            {
                                float opti = oldLength - newLength[i];
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = i;
                                    optimiz = opti;

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                unsigned int node5 = (int)arrayTSP[0][row-1].current;


                                double localMinChange = nn_source.minRadiusMap[0][node1];
                                double optimization = oldLength - newLength[finalSelect];

                                if(optimization > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 3;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                                }
                            }



                        }//end if i j

                    }//end if row < width

                }

            }


        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt


/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates work correctly final version
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                    Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                    double maxChecks3opt,
                                                    double iter)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks3opt)
    {


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {


            if(id > 0 && id < maxChecks3opt)
            {

                int row, i, j;
                double subtriplicate = (1.0)/3;


                //                if(id > 350631671)
                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row < width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            bool existingCandidate = 0;
                            if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                                existingCandidate = 1;


                            if(j> 9145)//350631671)
                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);



                            if(existingCandidate == 0)
                            {

                                double newLength[4];

                                double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                                newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                                newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                                newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                                newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                                int finalSelect = -1;
                                double optimiz = -INFINITY;
                                for(int i = 0; i < 4; i++)
                                {
                                    float opti = oldLength - newLength[i];
                                    if(opti > 0 && opti > optimiz)
                                    {
                                        finalSelect = i;
                                        optimiz = opti;

                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                    }
                                }


                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][row-1].current;

                                    {

                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 16;
                                        result = result | node5;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 3;

                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    }
                                }

                            }

                        }//end if i j

                    }//end if row < width

                }

            }


        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates work correctly final version
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStride(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                          Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                          double maxChecks3opt,  double maxChecksoptDivide,
                                                          double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;
            id = trunc(id);

            if(id > 0 && id < maxChecks3opt)
            {

                //                if(id > 350631671)
                //                    printf("id %f, local_id %f \n", id, local_id);

                int row, i, j;
                double subtriplicate = (1.0)/3;

                //                if(id > 350631671)
                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    row = int(rowN4);// check which one works

                    if(row < width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        double tempRowRow;
                        //                        double tempRowRowRow;

                        //                        if(iter > 1)
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        }
                        //                        else
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row+1);

                        //                        }

                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        //                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            bool existingCandidate = 0;
                            if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                                existingCandidate = 1;


                            //                            if(j> 9145)//350631671)
                            //                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);


                            if(existingCandidate == 0)
                            {

                                double newLength[4];
                                double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                                newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                                newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                                newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                                newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                                int finalSelect = -1;
                                for(int i = 0; i < 4; i++)
                                {
                                    float opti = oldLength - newLength[i];
                                    //                                    if(opti > 0 && opti > optimiz)
                                    //                                    {
                                    //                                        finalSelect = i;
                                    //                                        optimiz = opti;

                                    //                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                    //                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    //                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                    //                                    }
                                    if(opti > 0)
                                    {
                                        finalSelect = i;
                                        //                                        optimiz = opti;

                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);
                                        break;

                                    }
                                }


                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][row-1].current;


                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 3;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange


                                }

                            }

                        }//end if i j

                    }//end if row < width

                }

            }


        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt


/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method, using sharedArrayOccupied produces small quantity of opts
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStride_shared_noShareOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                               double maxChecks3opt,  double maxChecksoptDivide,
                                                                               double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register


    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }


    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;
            id = trunc(id);

            if(id > 0 && id < maxChecks3opt)
            {

                //                if(id > 350631671)
                //                    printf("id %f, local_id %f \n", id, local_id);

                int row, i, j;
                double subtriplicate = (1.0)/3;


                //WB.Q this way will produce i = j
                double idid = 9*id*id - (1.0)/9 ;
                double idMul3 = 3*id;
                double rowN0 = idMul3 + sqrt(idid );
                double rowN1 = pow(rowN0, subtriplicate);
                double rowN2 = idMul3 - sqrt(idid);
                double rowN3 = pow(rowN2, subtriplicate);
                double rowN4 = rowN1 + rowN3 + 1;

                row = int(rowN4);// check which one works

                if(row < width)
                {

                    double tempRowRow = (double)(row-1) / 6;
                    double tempRowRowRow = tempRowRow*(row)*(row-2);
                    //                        double tempRowRow;
                    //                        double tempRowRowRow;

                    //                        if(iter > 1)
                    //                        {
                    //                            tempRowRow = (double)(row-1) / 6;
                    //                            tempRowRowRow = tempRowRow*(row)*(row-2);
                    //                        }
                    //                        else
                    //                        {
                    //                            tempRowRow = (double)(row-1) / 6;
                    //                            tempRowRowRow = tempRowRow*(row)*(row+1);

                    //                        }

                    double id2opt = fabs( id - tempRowRowRow );
                    //WB.Q this way will produce i = j
                    double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                    i = int(3 + sqrt(sqrtTemp)) / 2 ;
                    j = id2opt - (i-2)*(i-1)/2 + 1;

                    //                        row = int(rowN4);// check which one works

                    if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                    {

                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                            existingCandidate = 1;

                        //                            if(j> 9145)//350631671)
                        //                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);

                        if(existingCandidate == 0)
                        {

                            double newLength[4];
                            double oldLength = dist(j-1, j,sharedArrayTSP) + dist(i-1, i,sharedArrayTSP) + dist(row-1, row,sharedArrayTSP);
                            newLength[0] = dist(j-1, i,sharedArrayTSP) + dist(row-1, i-1,sharedArrayTSP) + dist(row, j,sharedArrayTSP);
                            newLength[1] = dist(j-1, row-1,sharedArrayTSP) + dist(row, i-1,sharedArrayTSP) + dist(i,j,sharedArrayTSP);
                            newLength[2] = dist(j-1, i-1, sharedArrayTSP) + dist(row-1, j, sharedArrayTSP) + dist(row, i,sharedArrayTSP);
                            newLength[3] = dist(j-1, i, sharedArrayTSP) + dist(row-1, j,sharedArrayTSP) + dist(row,i-1,sharedArrayTSP);

                            int finalSelect = -1;
                            //                                double optimiz = -INFINITY;
                            for(int i = 0; i < 4; i++)
                            {
                                float opti = oldLength - newLength[i];
                                //                                    if(opti > 0 && opti > optimiz)
                                //                                    {
                                //                                        finalSelect = i;
                                //                                        optimiz = opti;

                                //                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                //                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                //                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                //                                    }
                                if(opti > 0)
                                {
                                    finalSelect = i;
                                    //                                        optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);
                                    break;

                                }

                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (unsigned int)sharedArrayTSP[j-1].current;
                                unsigned int node3 = (unsigned int)sharedArrayTSP[i-1].current;
                                unsigned int node5 = (unsigned int)sharedArrayTSP[row-1].current;


                                unsigned long long result = 0;
                                result = result | node3;
                                result = result << 16;
                                result = result | node5;

                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                codekopt = finalSelect * 100 + 3;

                                atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange


                            }

                        }

                    }//end if i j

                }//end if row < width



            }


        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method, using sharedArrayOccupied produces small quantity of opts
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStride_sharedwithOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                           Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                           double maxChecks3opt,  double maxChecksoptDivide,
                                                                           double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register


    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];
            sharedArrayOccupied[m] = 0;

        }
        __syncthreads();
    }


    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;
            id = trunc(id);

            if(id > 0 && id < maxChecks3opt)
            {

                //                if(id > 350631671)
                //                    printf("id %f, local_id %f \n", id, local_id);

                int row, i, j;
                double subtriplicate = (1.0)/3;

                //                if(id > 350631671)
                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    row = int(rowN4);// check which one works

                    if(row < width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        double tempRowRow;
                        //                        double tempRowRowRow;

                        //                        if(iter > 1)
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        }
                        //                        else
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row+1);

                        //                        }

                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        //                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            bool existingCandidate = 0;
                            if(sharedArrayOccupied[row-1] == 1 || sharedArrayOccupied[j-1] == 1 ||sharedArrayOccupied[i-1] == 1)
                                existingCandidate = 1;


                            if(j> 9145)//350631671)
                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);



                            if(existingCandidate == 0)
                            {

                                double newLength[4];
                                double oldLength = dist(j-1, j,sharedArrayTSP) + dist(i-1, i,sharedArrayTSP) + dist(row-1, row,sharedArrayTSP);
                                newLength[0] = dist(j-1, i,sharedArrayTSP) + dist(row-1, i-1,sharedArrayTSP) + dist(row, j,sharedArrayTSP);
                                newLength[1] = dist(j-1, row-1,sharedArrayTSP) + dist(row, i-1,sharedArrayTSP) + dist(i,j,sharedArrayTSP);
                                newLength[2] = dist(j-1, i-1, sharedArrayTSP) + dist(row-1, j, sharedArrayTSP) + dist(row, i,sharedArrayTSP);
                                newLength[3] = dist(j-1, i, sharedArrayTSP) + dist(row-1, j,sharedArrayTSP) + dist(row,i-1,sharedArrayTSP);

                                int finalSelect = -1;
                                //                                double optimiz = -INFINITY;
                                for(int i = 0; i < 4; i++)
                                {
                                    float opti = oldLength - newLength[i];
                                    //                                    if(opti > 0 && opti > optimiz)
                                    //                                    {
                                    //                                        finalSelect = i;
                                    //                                        optimiz = opti;

                                    //                                        sharedArrayOccupied[i-1]= 1;
                                    //                                        sharedArrayOccupied[j-1]= 1;
                                    //                                        sharedArrayOccupied[row-1]= 1;

                                    //                                    }
                                    if(opti > 0 )
                                    {
                                        finalSelect = i;
                                        //                                        optimiz = opti;

                                        sharedArrayOccupied[i-1]= 1;
                                        sharedArrayOccupied[j-1]= 1;
                                        sharedArrayOccupied[row-1]= 1;
                                        break;

                                    }
                                }


                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (int)sharedArrayTSP[j-1].current;
                                    unsigned int node3 = (int)sharedArrayTSP[i-1].current;
                                    unsigned int node5 = (int)sharedArrayTSP[row-1].current;

                                    {

                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 16;
                                        result = result | node5;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 3;

                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    }
                                }

                            }

                        }//end if i j

                    }//end if row < width

                }

            }


        }


    }
    __syncthreads();

}// end K_2optOneThreadOne3opt






/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates work correctly final version
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStride_onlySharedOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                           Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                           double maxChecks3opt,  double maxChecksoptDivide,
                                                                           double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register


    //    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];


    //    printf("local_id %f, shared %d, shareOcuu %f \n",
    //           local_id,  sharedArrayTSP[1023].current, sharedArrayTSP[1023].occupied);



    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        //        __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];

        //        float iterShared = (float)width / (float)BLOCKSIZE;

        //        for(int opt = 0; opt < iterShared; opt++)

        //        {
        //            int m = threadIdx.x + opt*BLOCKSIZE;
        //            if(m < width)
        //            {
        //                sharedArrayTSP[m].current = arrayTSP[0][m].current;
        //                sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
        //                sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];
        //            }

        //            __syncthreads();
        //        }



        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks3opt)
            {

                //                if(id > 350631671)
                //                    printf("id %f, local_id %f \n", id, local_id);

                int row, i, j;
                double subtriplicate = (1.0)/3;

                // if(id > 350631671)
                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row < width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        double tempRowRow;
                        //                        double tempRowRowRow;

                        //                        if(iter > 1)
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        }
                        //                        else
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row+1);

                        //                        }

                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            bool existingCandidate = 0;

                            //                            if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                            //                                existingCandidate = 1;

                            if(sharedArrayOccupied[row-1] == 1 || sharedArrayOccupied[j-1] == 1 || sharedArrayOccupied[i-1] == 1)
                                existingCandidate = 1;

                            //                            if(sharedArrayTSP[row-1].occupied == 1 || sharedArrayTSP[j-1].occupied == 1 || sharedArrayTSP[i-1].occupied == 1)
                            //                                existingCandidate = 1;


                            //  if(j> 9145)//350631671)
                            //    printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);



                            if(existingCandidate == 0)
                            {

                                double newLength[4];
                                double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                                newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                                newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                                newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                                newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                                //  printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f , shared %d, shareOcuu %f \n",
                                //      id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp, sharedArrayTSP[j-1].current, sharedArrayTSP[row-1].occupied);



                                int finalSelect = -1;
                                double optimiz = -INFINITY;
                                for(int i = 0; i < 4; i++)
                                {
                                    float opti = oldLength - newLength[i];
                                    if(opti > 0 && opti > optimiz)
                                    {
                                        finalSelect = i;
                                        optimiz = opti;

                                        //   atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        //   atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        //   atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);


                                        atomicExch(&(sharedArrayOccupied[i-1]), 1);
                                        atomicExch(&(sharedArrayOccupied[j-1]), 1);
                                        atomicExch(&(sharedArrayOccupied[row-1]), 1);


                                        //                                           __syncthreads();

                                        //                                        sharedArrayOccupied[i-1] = 1;
                                        //                                        sharedArrayOccupied[j-1] =1;
                                        //                                        sharedArrayOccupied[row-1]= 1;

                                        //  sharedArrayTSP[i-1].occupied = 1;
                                        //  sharedArrayTSP[j-1].occupied =1;
                                        //  sharedArrayTSP[row-1].occupied= 1;

                                    }
                                }


                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][row-1].current;

                                    {

                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 16;
                                        result = result | node5;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 3;

                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    }
                                }

                            }

                        }//end if i j

                    }//end if row < width

                }

            }


        }

    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates work correctly final version
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                              Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                              double maxChecks3opt,  double maxChecksoptDivide,
                                                              double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;
            id = trunc(id);

            if(id > 0 && id < maxChecks3opt)
            {

                int row, i, j;
                double subtriplicate = (1.0)/3;

                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    row = int(rowN4);// check which one works

                    if(row < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        //  row = int(rowN4);// check which one works
                        //  printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            double newLength[4];

                            double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                            newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                            newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                            newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                            newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                            int finalSelect = -1;
                            double optimiz = -INFINITY;
                            for(int i = 0; i < 4; i++)
                            {
                                float opti = oldLength - newLength[i];
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = i;
                                    optimiz = opti;
                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                unsigned int node5 = (int)arrayTSP[0][row-1].current;

                                double localMinChange = nn_source.minRadiusMap[0][node1];
                                double optimization = oldLength - newLength[finalSelect];

                                if(optimization > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 3;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);

                                }
                            }

                        }//end if i j

                    }//end if row < width

                }

            }

        }

    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates work correctly final version
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                     Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                     double maxChecks3opt,  double maxChecksoptDivide,
                                                                     double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];

    float iterShared = (float)width / (float)BLOCKSIZE;

    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];
            //            sharedArrayOccupied[m] = nn_source.minRadiusMap[0][m];
        }
        __syncthreads();
    }




    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;
            id = trunc(id);

            if(id > 0 && id < maxChecks3opt)
            {

                int row, i, j;
                double subtriplicate = (1.0)/3;

                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    row = int(rowN4);// check which one works
                    if(row < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        //                        row = int(rowN4);// check which one works
                        //  printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width)
                        {

                            double newLength[4];

                            double oldLength = dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP) + dist(row-1, row, sharedArrayTSP);
                            newLength[0] = dist(j-1, i, sharedArrayTSP) + dist(row-1, i-1, sharedArrayTSP) + dist(row, j, sharedArrayTSP);
                            newLength[1] = dist(j-1, row-1,sharedArrayTSP) + dist(row, i-1, sharedArrayTSP) + dist(i,j,sharedArrayTSP);
                            newLength[2] = dist(j-1, i-1, sharedArrayTSP) + dist(row-1, j, sharedArrayTSP) + dist(row, i,sharedArrayTSP);
                            newLength[3] = dist(j-1, i, sharedArrayTSP) + dist(row-1, j,sharedArrayTSP) + dist(row,i-1,sharedArrayTSP);

                            int finalSelect = -1;
                            double optimiz = -INFINITY;
                            for(int i = 0; i < 4; i++)
                            {
                                float opti = oldLength - newLength[i];
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = i;
                                    optimiz = opti;
                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[i-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[row-1].current;

                                double localMinChange = nn_source.minRadiusMap[0][node1];
                                double optimization = oldLength - newLength[finalSelect];

                                if(optimization > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 3;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);

                                }
                            }

                        }//end if i j

                    }//end if row < width

                }

            }

        }

    }
    __syncthreads();
}// end K_2optOneThreadOne3opt

/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iter_double(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                           Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                           double maxChecks3opt,
                                                           double iter)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks3opt)
    {

        int packSize = blockDim.x * gridDim.x;
        int row, i, j;
        double subtriplicate = (1.0)/3;

        for(int nu = 0; nu <= iter; nu++)
        {

            double id = local_id + nu * packSize;

            if(id < maxChecks3opt)
            {

                //WB.Q this way will produce i = j
                double idid = 9*id*id;
                double idMul3 = 3*id;
                double rowN0 = idMul3 + sqrt(idid - (1.0)/9);
                double rowN1 = pow(rowN0, subtriplicate);
                double rowN2 = idMul3 - sqrt(idid - (1.0)/9);
                float rowN3 = pow(rowN2, subtriplicate);
                float rowN4 = rowN1 + rowN3 + 1;

                row = int(rowN4);// check which one works

                if(row < width)
                {

                    //                    double rowrowrow = (double)(row-1) / (double )6;
                    //                    double id2opt = 8.0 * (rowrowrow *(row)*(row+1) - id );
                    //                    //WB.Q this way will produce i = j
                    //                    i = int(3 + sqrt(id2opt + 1.0) ) / 2 ;
                    //                    j = int(id2opt - (i-2)*(i-1)/2 + 1);


                    double id2opt = (row-1)*(row)*(row+1)/6 - id;
                    //WB.Q this way will produce i = j
                    i = int(3 + sqrt(8.0 * (double)id2opt + 1.0)) / 2 ;
                    j = id2opt - (i-2)*(i-1)/2 + 1;


                    //                    //qiao only for test
                    //                    if(id == maxChecks3opt - 10)
                    //                        printf("3-opt maxmum row j= %d, i %d, row %d, id % , localid %f, nu %d \n", j, i, row, id, local_id, nu);


                    if(i<row && i!=row &&i+1!= row &&j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                    {

                        //                        //qiao only for test
                        //                        if(nn_source.grayValueMap[0][row-1] > 5000 || nn_source.grayValueMap[0][i-1] > 5000 || nn_source.grayValueMap[0][j-1] > 5000)
                        //                            printf("3-opt maxmum row id %d, row %d, i %d, j %d \n", id, row, i,j);

                        //                        //qiao only for test
                        //                        if(row == width - 10)
                        //                            printf("3-opt maxmum j %d, i %d, row %d,  \n", j, i, row);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                            existingCandidate = 1;


                        if(existingCandidate == 0)
                        {

                            double newLength[4];

                            double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                            newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                            newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                            newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                            newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                            int finalSelect = -1;
                            double optimiz = -INFINITY;
                            for(int i = 0; i < 4; i++)
                            {
                                float opti = oldLength - newLength[i];
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = i;
                                    optimiz = opti;

                                    //                               if(blockIdx.x == 124698010  ||blockIdx.x == 124698013   || blockIdx.x == 0)
                                    //                                printf("3opt GPU blockIdx.x=%d, selec %d, order %d, %d, %d, oldLength %f, newi %f, opti %f, optimiz %f ; node135 %d,%d,%d\n",blockIdx.x, finalSelect,
                                    //                                       nn_source.grayValueMap[0][j-1], nn_source.grayValueMap[0][i-1], nn_source.grayValueMap[0][row-1]
                                    //                                        , oldLength, newLength[i], opti, optimiz, j-1, i-1, row-1);

                                    atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                }
                            }


                            if(finalSelect >= 0)
                            {
                                //                            float optimization = oldLength - newLength[finalSelect];
                                // here automic operation is necessary

                                unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                unsigned int node5 = (int)arrayTSP[0][row-1].current;

                                //                            printf("3opt GPU mode %d, order %d, %d, %d, oldLength %f, new1 %f, new2 %f, new3 %f, new4 %f; node135 %d,%d,%d \n", finalSelect, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5]
                                //                                    , oldLength, newLength[0],newLength[1], newLength[2], newLength[3], node1, node3, node5);

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimization > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 3;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                                }
                            }

                        }



                    }//end if i j

                }//end if row < width

            }
        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall(NeuralNetLinks<BufferDimension, Point> nn_source,
                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                               double maxChecks3opt,
                                               double iter)
{

    double id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    int row, i, j;
    double subtriplicate = (1.0)/3;


    if(id > maxChecks3opt-5)//350631671)
        printf("largeID id %f \n",  id);
    else if (id <5)
        printf("SmallID id %f \n", id);



    if(id < maxChecks3opt)
    {

        //WB.Q this way will produce i = j
        double idid = 9*id*id;
        double idMul3 = 3*id;
        double rowN0 = idMul3 + sqrt(idid - (1.0)/9);
        double rowN1 = pow(rowN0, subtriplicate);
        double rowN2 = idMul3 - sqrt(idid - (1.0)/9);
        double rowN3 = pow(rowN2, subtriplicate);
        float rowN4 = rowN1 + rowN3 + 1;

        row = int(rowN4);// check which one works

        if(row < width)
        {
            //            double rowrowrow = (row-1)*(row)*(row+1)/6  - id;
            //            double id2opt = 8.0 * (rowrowrow) + 1.0;
            //            double sqrtId2opt = 3 + sqrt(id2opt) ;

            //            //WB.Q this way will produce i = j
            //            i = (sqrtId2opt) / 2 ; // i = ((sqrtId2opt) / 2) ;// i = (int)((sqrtId2opt) / 2) ; error
            //            j = id2opt - (i-2)*(i-1)/2 + 1;


            double id2opt = (row-1)*(row)*(row+1)/6 - id;
            double sqrtTemp = 8.0 * (double)id2opt + 1.0;
            double sqrtId2opt = 3 + sqrt(sqrtTemp);
            i = (sqrtId2opt) / 2 ;
            j = id2opt - (i-2)*(i-1)/2 + 1;


            //            double id2opt = (row-1)*(row)*(row+1)/6 - id;
            //            //WB.Q this way will produce i = j
            //            i = int(3 + sqrt(8.0 * (double)id2opt + 1.0)) / 2 ;
            //            j = id2opt - (i-2)*(i-1)/2 + 1;




            //            //qiao only for test
            //            if(row > width -10  ||id > maxChecks3opt - 10)
            //                printf("3-opt maxmum  row %d, id %lld \n", row, id);

            if(i<row && i!=row &&i+1!= row &&j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
            {

                //                        //qiao only for test
                //                        if(nn_source.grayValueMap[0][row-1] > 5000 || nn_source.grayValueMap[0][i-1] > 5000 || nn_source.grayValueMap[0][j-1] > 5000)
                //                            printf("3-opt maxmum row id %d, row %d, i %d, j %d \n", id, row, i,j);

                //                        //qiao only for test
                //                        if(row > width -10 ||  j > width-10 || i > width -10 ||id > maxChecks3opt - 10)
                //                            printf("3-opt maxmum j %d, i %d, row %d, id %lld \n", j, i, row, id);
                printf("3-opt maxmum row j= %d, i %d, row %d, id %f \n", j, i, row, id);



                bool existingCandidate = 0;
                if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                    existingCandidate = 1;

                if(existingCandidate == 0)
                {

                    double newLength[4];

                    double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                    newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                    newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                    newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                    newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                    int finalSelect = -1;
                    double optimiz = -INFINITY;
                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength - newLength[i];
                        if(opti > 0 && opti > optimiz)
                        {
                            finalSelect = i;
                            optimiz = opti;

                            //                               if(blockIdx.x == 124698010  ||blockIdx.x == 124698013   || blockIdx.x == 0)
                            //                                printf("3opt GPU blockIdx.x=%d, selec %d, order %d, %d, %d, oldLength %f, newi %f, opti %f, optimiz %f ; node135 %d,%d,%d\n",blockIdx.x, finalSelect,
                            //                                       nn_source.grayValueMap[0][j-1], nn_source.grayValueMap[0][i-1], nn_source.grayValueMap[0][row-1]
                            //                                        , oldLength, newLength[i], opti, optimiz, j-1, i-1, row-1);

                            atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                        }
                    }


                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (int)arrayTSP[0][j-1].current;
                        unsigned int node3 = (int)arrayTSP[0][i-1].current;
                        unsigned int node5 = (int)arrayTSP[0][row-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                    }

                }

            }//end if i j

        }//end if row < width

    }

    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_oneThreadOne6opt_rockiSmall(NeuralNetLinks<BufferDimension, Point> nn_source,
                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                               double maxChecks6opt, double maxChecks3opt, unsigned int iter)
{

    double id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(id < maxChecks6opt)
    {

        //        int packSize = blockDim.x * gridDim.x;
        double  outi, outj;
        int row,row_1, i, j, i_1, j_1;
        double subtriplicate = (1.0)/3;

        //WB.Q this way will produce i = j
        outi = (3 + sqrt(8.0f * (double )id + 1.0f)) / 2 ;
        outj = id - (outi-2)*(outi-1)/2 + 1;

        //WB.Q this way will produce i = j
        double rowN0 = 3*outi + sqrt(9*outi*outi - (1.0)/9);
        double rowN1 = pow(rowN0, subtriplicate);
        double rowN2 = 3*outi - sqrt(9*outi*outi - (1.0)/9);
        double rowN3 = pow(rowN2, subtriplicate);
        double rowN4 = rowN1 + rowN3 + 1;

        //WB.Q this way will produce i = j
        double rowN0_1 = 3*outj + sqrt(9*outj*outj - (1.0)/9);
        double rowN1_1 = pow(rowN0_1, subtriplicate);
        double rowN2_1 = 3*outj - sqrt(9*outj*outj - (1.0)/9);
        double rowN3_1 = pow(rowN2_1, subtriplicate);
        double rowN4_1 = rowN1_1 + rowN3_1 + 1;

        row = int(rowN4);// check which one works
        row_1 = int(rowN4_1);

        if(row < width && row_1 < width)
        {
            double id2opt = (row-1)*(row)*(row+1)/6 - outi;

            //WB.Q this way will produce i = j
            i = int(3 + sqrt(8.0f * (double)id2opt + 1.0f)) / 2 ;
            j = id2opt - (i-2)*(i-1)/2 + 1;

            double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;

            //WB.Q this way will produce i = j
            i_1 = int(3 + sqrt(8.0f * (double)id2opt_1 + 1.0f)) / 2 ;
            j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


            //                    //qiao only for test
            //                    if(id == maxChecks6opt - 2)
            //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


            if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                    && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                    && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                    )
            {

                //                        if(row > width -2)
                //                            printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                bool existingCandidate = 0;
                if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
                        nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
                        nn_source.minRadiusMap[0][i-1] == 1 ||nn_source.minRadiusMap[0][row-1] == 1)
                    existingCandidate = 1;

                if(existingCandidate == 0)
                {

                    float oldLength = dist(j_1-1, j_1, arrayTSP[0]) + dist(i_1-1, i_1, arrayTSP[0]) + dist(row_1-1, row_1, arrayTSP[0])
                            + dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0])    ;

                    float newLength;
                    int array[12];
                    array[0] = j_1-1;
                    array[1] = j_1;
                    array[2] = i_1-1;
                    array[3] = i_1;
                    array[4] = row_1-1;
                    array[5] = row_1;
                    array[6] = j-1;
                    array[7] = j;
                    array[8] = i-1;
                    array[9] = i;
                    array[10] = row-1;
                    array[11] = row;

                    int finalSelect = -1;
                    float optimiz = -INFINITY;


                    for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                    {
                        int nd1 = nn_source.nodeParentMap[0][opt] -1;
                        int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                        int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                        int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                        int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                        int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                        int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                        int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                        int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                        int nd10 = nn_source.nodeParentMap[0][opt+9] -1;
                        int nd11 = nn_source.nodeParentMap[0][opt+10] -1;
                        int nd12 = nn_source.nodeParentMap[0][opt+11] -1;

                        int optCandi = opt / 12;

                        newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0])
                                + dist(array[nd9],array[nd10], arrayTSP[0]) + dist(array[nd11],array[nd12], arrayTSP[0] );

                        float opti = oldLength - newLength;
                        if(opti > 0 && opti > optimiz)
                        {
                            finalSelect = optCandi;
                            optimiz = opti;


                            atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);
                        }

                    }

                    if(finalSelect >= 0)
                    {

                        int node1 = (int)arrayTSP[0][j_1-1].current;
                        int node3 = (int)arrayTSP[0][i_1-1].current;
                        int node5 = (int)arrayTSP[0][row_1-1].current;
                        int node7 = (int)arrayTSP[0][j-1].current;
                        int node9 = (int)arrayTSP[0][i-1].current;
                        int node11 = (int)arrayTSP[0][row-1].current;

                        //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                        //                            if(optimiz > localMinChange)
                        {

                            //12 is restricted by 64 bit and five numbers
                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 12;
                            result = result | node5;
                            result = result << 12;
                            result = result | node7;
                            result = result << 12;
                            result = result | node9;
                            result = result << 12;
                            result = result | node11;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 6;

                            //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                            //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                        }
                    }
                }

            }//end if i j
        }//end if row < width

    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_oneThreadOne6opt_rockiSmall_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                    Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                    double maxChecks6opt, double maxChecks3opt, unsigned int iter)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks6opt)
    {

        int packSize = blockDim.x * gridDim.x;

        for(int nu = 0; nu <= iter; nu++)
        {

            double id = local_id + nu * packSize;

            if(id < maxChecks6opt)
            {

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                outi = (3 + sqrt(8.0f * (double )id + 1.0f)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                //WB.Q this way will produce i = j
                double rowN0 = 3*outi + sqrt(9*outi*outi - (1.0)/9);
                double rowN1 = pow(rowN0, subtriplicate);
                double rowN2 = 3*outi - sqrt(9*outi*outi - (1.0)/9);
                double rowN3 = pow(rowN2, subtriplicate);
                double rowN4 = rowN1 + rowN3 + 1;

                //WB.Q this way will produce i = j
                double rowN0_1 = 3*outj + sqrt(9*outj*outj - (1.0)/9);
                double rowN1_1 = pow(rowN0_1, subtriplicate);
                double rowN2_1 = 3*outj - sqrt(9*outj*outj - (1.0)/9);
                double rowN3_1 = pow(rowN2_1, subtriplicate);
                double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                row = int(rowN4);// check which one works
                row_1 = int(rowN4_1);

                if(row < width && row_1 < width)
                {
                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;

                    //WB.Q this way will produce i = j
                    i = int(3 + sqrt(8.0f * (double)id2opt + 1.0f)) / 2 ;
                    j = id2opt - (i-2)*(i-1)/2 + 1;

                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;

                    //WB.Q this way will produce i = j
                    i_1 = int(3 + sqrt(8.0f * (double)id2opt_1 + 1.0f)) / 2 ;
                    j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                    //                    //qiao only for test
                    //                    if(id == maxChecks6opt - 2)
                    //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                    if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                            && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                            && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                            )
                    {

                        //                        if(row > width -2)
                        //                            printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
                                nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
                                nn_source.minRadiusMap[0][i-1] == 1 ||nn_source.minRadiusMap[0][row-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(j_1-1, j_1, arrayTSP[0]) + dist(i_1-1, i_1, arrayTSP[0]) + dist(row_1-1, row_1, arrayTSP[0])
                                    + dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0])    ;

                            float newLength;
                            int array[12];
                            array[0] = j_1-1;
                            array[1] = j_1;
                            array[2] = i_1-1;
                            array[3] = i_1;
                            array[4] = row_1-1;
                            array[5] = row_1;
                            array[6] = j-1;
                            array[7] = j;
                            array[8] = i-1;
                            array[9] = i;
                            array[10] = row-1;
                            array[11] = row;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;


                            for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;
                                int nd11 = nn_source.nodeParentMap[0][opt+10] -1;
                                int nd12 = nn_source.nodeParentMap[0][opt+11] -1;

                                int optCandi = opt / 12;

                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0])
                                        + dist(array[nd9],array[nd10], arrayTSP[0]) + dist(array[nd11],array[nd12], arrayTSP[0] );

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                }

                            }

                            if(finalSelect >= 0)
                            {

                                int node1 = (int)arrayTSP[0][j_1-1].current;
                                int node3 = (int)arrayTSP[0][i_1-1].current;
                                int node5 = (int)arrayTSP[0][row_1-1].current;
                                int node7 = (int)arrayTSP[0][j-1].current;
                                int node9 = (int)arrayTSP[0][i-1].current;
                                int node11 = (int)arrayTSP[0][row-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    //12 is restricted by 64 bit and five numbers
                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 12;
                                    result = result | node5;
                                    result = result << 12;
                                    result = result | node7;
                                    result = result << 12;
                                    result = result | node9;
                                    result = result << 12;
                                    result = result | node11;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 6;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }//end if i j
                }//end if row < width

            }
        }
    }
    __syncthreads();
}// end K_6optOneThreadOne6opt



/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_oneThreadOne6opt_qiao_stride_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                     Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                     double maxChecks6opt, double maxChecks3opt, double maxChecksoptDivide,
                                                     double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks6opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id < maxChecks6opt)
            {
                id = trunc(id);

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;
                //                outj = trunc(outj);

                if(outi < maxChecks3opt && outj <maxChecks3opt)
                {
                    //WB.Q this way will produce i = j
                    double idid = 9*outi*outi - (1.0)/9;
                    double idMuli = 3*outi;
                    double rowN0 = idMuli + sqrt(idid);
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMuli - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    //WB.Q this way will produce i = j
                    double ididj= 9*outj*outj - (1.0)/9;
                    double idMulj = 3*outj;
                    double rowN0_1 = idMulj + sqrt(ididj);
                    double rowN1_1 = pow(rowN0_1, subtriplicate);
                    double rowN2_1 = idMulj - sqrt(ididj);
                    double rowN3_1 = pow(rowN2_1, subtriplicate);
                    double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                    row = int(rowN4);// check which one works
                    row_1 = int(rowN4_1);

                    if(row < width && row_1 < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow * row *(row-2);
                        //                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;
                        double id2opt = fabs(outi - tempRowRowRow);

                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;

                        //                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;
                        double tempRowRowj = (double)(row_1-1) / 6;
                        double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
                        double id2opt_1 = fabs(outj - tempRowRowRowj);

                        //WB.Q this way will produce i = j
                        double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
                        i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
                        j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                        //                    //qiao only for test
                        //                    if(id == maxChecks6opt - 2)
                        //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                                && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                                && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                                )
                        {

                            //                            if(row > width -2)
                            //                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);

                            float oldLength = dist(j_1-1, j_1, arrayTSP[0]) + dist(i_1-1, i_1, arrayTSP[0]) + dist(row_1-1, row_1, arrayTSP[0])
                                    + dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0])    ;

                            float newLength;
                            int array[12];
                            array[0] = j_1-1;
                            array[1] = j_1;
                            array[2] = i_1-1;
                            array[3] = i_1;
                            array[4] = row_1-1;
                            array[5] = row_1;
                            array[6] = j-1;
                            array[7] = j;
                            array[8] = i-1;
                            array[9] = i;
                            array[10] = row-1;
                            array[11] = row;

                            int finalSelect = -1;
                            float optimiz = -1;

                            for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                            {
                                int nd1 = nn_source.evtMap[0][opt] -1;
                                int nd2 = nn_source.evtMap[0][opt+1] -1;
                                int nd3 = nn_source.evtMap[0][opt+2] -1;
                                int nd4 = nn_source.evtMap[0][opt+3] -1;
                                int nd5 = nn_source.evtMap[0][opt+4] -1;
                                int nd6 = nn_source.evtMap[0][opt+5] -1;
                                int nd7 = nn_source.evtMap[0][opt+6] -1;
                                int nd8 = nn_source.evtMap[0][opt+7] -1;
                                int nd9 = nn_source.evtMap[0][opt+8] -1;
                                int nd10 = nn_source.evtMap[0][opt+9] -1;
                                int nd11 = nn_source.evtMap[0][opt+10] -1;
                                int nd12 = nn_source.evtMap[0][opt+11] -1;

                                int optCandi = opt / 12;

                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0])
                                        + dist(array[nd9],array[nd10], arrayTSP[0]) + dist(array[nd11],array[nd12], arrayTSP[0] );

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                }

                            }

                            if(finalSelect >= 0)
                            {

                                int node1 = (int)arrayTSP[0][j_1-1].current;
                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {

                                    int node3 = (int)arrayTSP[0][i_1-1].current;
                                    int node5 = (int)arrayTSP[0][row_1-1].current;
                                    int node7 = (int)arrayTSP[0][j-1].current;
                                    int node9 = (int)arrayTSP[0][i-1].current;
                                    int node11 = (int)arrayTSP[0][row-1].current;

                                    //12 is restricted by 64 bit and five numbers
                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 12;
                                    result = result | node5;
                                    result = result << 12;
                                    result = result | node7;
                                    result = result << 12;
                                    result = result | node9;
                                    result = result << 12;
                                    result = result | node11;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 6;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }//end if i j
                    }//end if row < width

                }

            }
        }
    }
    __syncthreads();
}// end K_6optOneThreadOne6opt


/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_bestScheme_shared_qiao_stride_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                      Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                      double maxChecks6opt, double maxChecks3opt, double maxChecksoptDivide,
                                                      double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    //    __shared__ QWChar optPossibilities[OPTPOSSIBILITES6OPT];


    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    //    float iterSharedPossble =  (float)OPTPOSSIBILITES6OPT / (float)BLOCKSIZE;
    //    for(int opt = 0; opt < iterSharedPossble; opt++)
    //    {
    //        int m = threadIdx.x + opt*BLOCKSIZE;

    //        if(m < OPTPOSSIBILITES6OPT)
    //            optPossibilities[m] = nn_source.evtMap[0][m];
    //        __syncthreads();

    //    }

    if(local_id < maxChecks6opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id < maxChecks6opt)
            {
                id = trunc(id);

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;
                //                outj = trunc(outj);

                if(outi < maxChecks3opt && outj <maxChecks3opt)
                {
                    //WB.Q this way will produce i = j
                    double idid = 9*outi*outi - (1.0)/9;
                    double idMuli = 3*outi;
                    double rowN0 = idMuli + sqrt(idid);
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMuli - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    //WB.Q this way will produce i = j
                    double ididj= 9*outj*outj - (1.0)/9;
                    double idMulj = 3*outj;
                    double rowN0_1 = idMulj + sqrt(ididj);
                    double rowN1_1 = pow(rowN0_1, subtriplicate);
                    double rowN2_1 = idMulj - sqrt(ididj);
                    double rowN3_1 = pow(rowN2_1, subtriplicate);
                    double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                    row = int(rowN4);// check which one works
                    row_1 = int(rowN4_1);

                    if(row < width && row_1 < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow * row *(row-2);
                        //                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;
                        double id2opt = fabs(outi - tempRowRowRow);

                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;

                        //                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;
                        double tempRowRowj = (double)(row_1-1) / 6;
                        double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
                        double id2opt_1 = fabs(outj - tempRowRowRowj);

                        //WB.Q this way will produce i = j
                        double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
                        i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
                        j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                        //                    //qiao only for test
                        //                    if(id == maxChecks6opt - 2)
                        //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                                && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                                && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                                )
                        {

                            //                            if(row > width -2)
                            //                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);

                            float oldLength = dist(j_1-1, j_1, sharedArrayTSP) + dist(i_1-1, i_1, sharedArrayTSP) + dist(row_1-1, row_1, sharedArrayTSP)
                                    + dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP) + dist(row-1, row, sharedArrayTSP)    ;

                            float newLength;
                            int array[12];
                            array[0] = j_1-1;
                            array[1] = j_1;
                            array[2] = i_1-1;
                            array[3] = i_1;
                            array[4] = row_1-1;
                            array[5] = row_1;
                            array[6] = j-1;
                            array[7] = j;
                            array[8] = i-1;
                            array[9] = i;
                            array[10] = row-1;
                            array[11] = row;

                            int finalSelect = -1;
                            float optimiz = -1;

                            for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                            {
                                int nd1 = nn_source.evtMap[0][opt] -1;
                                int nd2 = nn_source.evtMap[0][opt+1] -1;
                                int nd3 = nn_source.evtMap[0][opt+2] -1;
                                int nd4 = nn_source.evtMap[0][opt+3] -1;
                                int nd5 = nn_source.evtMap[0][opt+4] -1;
                                int nd6 = nn_source.evtMap[0][opt+5] -1;
                                int nd7 = nn_source.evtMap[0][opt+6] -1;
                                int nd8 = nn_source.evtMap[0][opt+7] -1;
                                int nd9 = nn_source.evtMap[0][opt+8] -1;
                                int nd10 = nn_source.evtMap[0][opt+9] -1;
                                int nd11 = nn_source.evtMap[0][opt+10] -1;
                                int nd12 = nn_source.evtMap[0][opt+11] -1;

                                int optCandi = opt / 12;

                                newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                        + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP)
                                        + dist(array[nd9],array[nd10], sharedArrayTSP) + dist(array[nd11],array[nd12], sharedArrayTSP );

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                }

                            }

                            if(finalSelect >= 0)
                            {

                                int node1 = (int)sharedArrayTSP[j_1-1].current;
                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {

                                    int node3 = (int)sharedArrayTSP[i_1-1].current;
                                    int node5 = (int)sharedArrayTSP[row_1-1].current;
                                    int node7 = (int)sharedArrayTSP[j-1].current;
                                    int node9 = (int)sharedArrayTSP[i-1].current;
                                    int node11 = (int)sharedArrayTSP[row-1].current;

                                    //12 is restricted by 64 bit and five numbers
                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 12;
                                    result = result | node5;
                                    result = result << 12;
                                    result = result | node7;
                                    result = result << 12;
                                    result = result | node9;
                                    result = result << 12;
                                    result = result | node11;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 6;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }//end if i j
                    }//end if row < width

                }

            }
        }
    }
    __syncthreads();
}// end K_6optOneThreadOne6opt





/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_bestScheme_shared_possible_stride_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                          Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                          double maxChecks6opt, double maxChecks3opt, double maxChecksoptDivide,
                                                          double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ QWChar optPossibilities[OPTPOSSIBILITES6OPT];

    float iterSharedPossble =  (float)OPTPOSSIBILITES6OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES6OPT)
            optPossibilities[m] = nn_source.evtMap[0][m];
        __syncthreads();

    }

    if(local_id < maxChecks6opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id < maxChecks6opt)
            {
                id = trunc(id);

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;
                //                outj = trunc(outj);

                if(outi < maxChecks3opt && outj <maxChecks3opt)
                {
                    //WB.Q this way will produce i = j
                    double idid = 9*outi*outi - (1.0)/9;
                    double idMuli = 3*outi;
                    double rowN0 = idMuli + sqrt(idid);
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMuli - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    //WB.Q this way will produce i = j
                    double ididj= 9*outj*outj - (1.0)/9;
                    double idMulj = 3*outj;
                    double rowN0_1 = idMulj + sqrt(ididj);
                    double rowN1_1 = pow(rowN0_1, subtriplicate);
                    double rowN2_1 = idMulj - sqrt(ididj);
                    double rowN3_1 = pow(rowN2_1, subtriplicate);
                    double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                    row = int(rowN4);// check which one works
                    row_1 = int(rowN4_1);

                    if(row < width && row_1 < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow * row *(row-2);
                        //                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;
                        double id2opt = fabs(outi - tempRowRowRow);

                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;

                        //                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;
                        double tempRowRowj = (double)(row_1-1) / 6;
                        double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
                        double id2opt_1 = fabs(outj - tempRowRowRowj);

                        //WB.Q this way will produce i = j
                        double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
                        i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
                        j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                        //                    //qiao only for test
                        //                    if(id == maxChecks6opt - 2)
                        //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                                && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                                && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                                )
                        {

                            //                            if(row > width -2)
                            //                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);

                            float oldLength = dist(j_1-1, j_1, arrayTSP[0]) + dist(i_1-1, i_1, arrayTSP[0]) + dist(row_1-1, row_1, arrayTSP[0])
                                    + dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0])    ;

                            float newLength;
                            int array[12];
                            array[0] = j_1-1;
                            array[1] = j_1;
                            array[2] = i_1-1;
                            array[3] = i_1;
                            array[4] = row_1-1;
                            array[5] = row_1;
                            array[6] = j-1;
                            array[7] = j;
                            array[8] = i-1;
                            array[9] = i;
                            array[10] = row-1;
                            array[11] = row;

                            int finalSelect = -1;
                            float optimiz = -1;

                            for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                            {
                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;
                                int nd9 = optPossibilities[opt+8] -1;
                                int nd10 = optPossibilities[opt+9] -1;
                                int nd11 = optPossibilities[opt+10] -1;
                                int nd12 = optPossibilities[opt+11] -1;

                                int optCandi = opt / 12;

                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0])
                                        + dist(array[nd9],array[nd10], arrayTSP[0]) + dist(array[nd11],array[nd12], arrayTSP[0] );

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                }

                            }

                            if(finalSelect >= 0)
                            {

                                int node1 = (int)arrayTSP[0][j_1-1].current;
                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {

                                    int node3 = (int)arrayTSP[0][i_1-1].current;
                                    int node5 = (int)arrayTSP[0][row_1-1].current;
                                    int node7 = (int)arrayTSP[0][j-1].current;
                                    int node9 = (int)arrayTSP[0][i-1].current;
                                    int node11 = (int)arrayTSP[0][row-1].current;

                                    //12 is restricted by 64 bit and five numbers
                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 12;
                                    result = result | node5;
                                    result = result << 12;
                                    result = result | node7;
                                    result = result << 12;
                                    result = result | node9;
                                    result = result << 12;
                                    result = result | node11;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 6;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }//end if i j
                    }//end if row < width

                }

            }
        }
    }
    __syncthreads();
}// end K_6optOneThreadOne6opt





/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_oneThreadOne6opt_qiao_stride_iter_firstScheme(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                 Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                 double maxChecks6opt, double maxChecks3opt, double maxChecksoptDivide,
                                                                 double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks6opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id < maxChecks6opt)
            {
                id = trunc(id);

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;
                //                outj = trunc(outj);

                if(outi < maxChecks3opt && outj <maxChecks3opt)
                {
                    //WB.Q this way will produce i = j
                    double idid = 9*outi*outi - (1.0)/9;
                    double idMuli = 3*outi;
                    double rowN0 = idMuli + sqrt(idid);
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMuli - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    //WB.Q this way will produce i = j
                    double ididj= 9*outj*outj - (1.0)/9;
                    double idMulj = 3*outj;
                    double rowN0_1 = idMulj + sqrt(ididj);
                    double rowN1_1 = pow(rowN0_1, subtriplicate);
                    double rowN2_1 = idMulj - sqrt(ididj);
                    double rowN3_1 = pow(rowN2_1, subtriplicate);
                    double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                    row = int(rowN4);// check which one works
                    row_1 = int(rowN4_1);

                    if(row < width && row_1 < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow * row *(row-2);
                        //                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;
                        double id2opt = fabs(outi - tempRowRowRow);

                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;

                        //                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;
                        double tempRowRowj = (double)(row_1-1) / 6;
                        double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
                        double id2opt_1 = fabs(outj - tempRowRowRowj);

                        //WB.Q this way will produce i = j
                        double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
                        i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
                        j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                        //                    //qiao only for test
                        //                    if(id == maxChecks6opt - 2)
                        //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                                && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                                && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                                )
                        {

                            //                            if(row > width -2)
                            //                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                            bool existingCandidate = 0;
                            if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
                                    nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
                                    nn_source.minRadiusMap[0][i-1] == 1 ||nn_source.minRadiusMap[0][row-1] == 1)
                                existingCandidate = 1;

                            if(existingCandidate == 0)
                            {

                                float oldLength = dist(j_1-1, j_1, arrayTSP[0]) + dist(i_1-1, i_1, arrayTSP[0]) + dist(row_1-1, row_1, arrayTSP[0])
                                        + dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0])    ;

                                float newLength;
                                int array[12];
                                array[0] = j_1-1;
                                array[1] = j_1;
                                array[2] = i_1-1;
                                array[3] = i_1;
                                array[4] = row_1-1;
                                array[5] = row_1;
                                array[6] = j-1;
                                array[7] = j;
                                array[8] = i-1;
                                array[9] = i;
                                array[10] = row-1;
                                array[11] = row;

                                int finalSelect = -1;
                                //                                float optimiz = -INFINITY;


                                for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                                {
                                    int nd1 = nn_source.evtMap[0][opt] -1;
                                    int nd2 = nn_source.evtMap[0][opt+1] -1;
                                    int nd3 = nn_source.evtMap[0][opt+2] -1;
                                    int nd4 = nn_source.evtMap[0][opt+3] -1;
                                    int nd5 = nn_source.evtMap[0][opt+4] -1;
                                    int nd6 = nn_source.evtMap[0][opt+5] -1;
                                    int nd7 = nn_source.evtMap[0][opt+6] -1;
                                    int nd8 = nn_source.evtMap[0][opt+7] -1;
                                    int nd9 = nn_source.evtMap[0][opt+8] -1;
                                    int nd10 = nn_source.evtMap[0][opt+9] -1;
                                    int nd11 = nn_source.evtMap[0][opt+10] -1;
                                    int nd12 = nn_source.evtMap[0][opt+11] -1;

                                    int optCandi = opt / 12;

                                    newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                            + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0])
                                            + dist(array[nd9],array[nd10], arrayTSP[0]) + dist(array[nd11],array[nd12], arrayTSP[0] );

                                    float opti = oldLength - newLength;
                                    //                                    if(opti > 0 && opti > optimiz)
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;
                                        //                                        optimiz = opti;
                                        atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                        break;
                                    }
                                }

                                if(finalSelect >= 0)
                                {
                                    int node1 = (int)arrayTSP[0][j_1-1].current;
                                    int node3 = (int)arrayTSP[0][i_1-1].current;
                                    int node5 = (int)arrayTSP[0][row_1-1].current;
                                    int node7 = (int)arrayTSP[0][j-1].current;
                                    int node9 = (int)arrayTSP[0][i-1].current;
                                    int node11 = (int)arrayTSP[0][row-1].current;

                                    //                            float localMinChange = nn_source.minRadiusMap[0][node1];
                                    //                            if(optimiz > localMinChange)
                                    {
                                        //12 is restricted by 64 bit and five numbers
                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 12;
                                        result = result | node5;
                                        result = result << 12;
                                        result = result | node7;
                                        result = result << 12;
                                        result = result | node9;
                                        result = result << 12;
                                        result = result | node11;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 6;

                                        //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                        //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                    }
                                }
                            }

                        }//end if i j
                    }//end if row < width
                }
            }
        }
    }
    __syncthreads();
}// end K_6optOneThreadOne6opt





/*!
 * \brief 2408 QWB: add parallel 5-opt from 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_2_qiao_stride_iter_firstScheme_allShare(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                           Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                           double maxChecks6opt, double maxChecks3opt, double maxChecksoptDivide,
                                                           double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];
    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];

    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nVisitedMap[0][m];

        __syncthreads();

    }

    if(local_id < maxChecks6opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;
            id = trunc(id);

            if(id < maxChecks6opt)
            {

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;
                //                outj = trunc(outj);

                if(outi < maxChecks3opt && outj <maxChecks3opt)
                {
                    //WB.Q this way will produce i = j
                    double idid = 9*outi*outi - (1.0)/9;
                    double idMuli = 3*outi;
                    double rowN0 = idMuli + sqrt(idid);
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMuli - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    //WB.Q this way will produce i = j
                    double ididj= 9*outj*outj - (1.0)/9;
                    double idMulj = 3*outj;
                    double rowN0_1 = idMulj + sqrt(ididj);
                    double rowN1_1 = pow(rowN0_1, subtriplicate);
                    double rowN2_1 = idMulj - sqrt(ididj);
                    double rowN3_1 = pow(rowN2_1, subtriplicate);
                    double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                    row = int(rowN4);// check which one works
                    row_1 = int(rowN4_1);

                    if(row < width && row_1 < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow * row *(row-2);
                        double id2opt = fabs(outi - tempRowRowRow);

                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;

                        double tempRowRowj = (double)(row_1-1) / 6;
                        double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
                        double id2opt_1 = fabs(outj - tempRowRowRowj);

                        //WB.Q this way will produce i = j
                        double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
                        i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
                        j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                        //                    //qiao only for test
                        //                    if(id == maxChecks6opt - 2)
                        //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                                && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                                && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                                )
                        {

                            //                            if(row < 20)
                            //                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);

                            bool existingCandidate = 0;
                            if(sharedArrayOccupied[j_1-1] == 1 || sharedArrayOccupied[i_1-1] == 1 ||
                                    sharedArrayOccupied[row_1-1] == 1 ||sharedArrayOccupied[j-1] == 1 ||
                                    sharedArrayOccupied[i-1] == 1)
                                existingCandidate = 1;

                            if(existingCandidate == 0)
                            {

                                float oldLength = dist(j_1-1, j_1, sharedArrayTSP) + dist(i_1-1, i_1, sharedArrayTSP) + dist(row_1-1, row_1, sharedArrayTSP)
                                        + dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP)     ;



                                float newLength;
                                int array[10];
                                array[0] = j_1-1;
                                array[1] = j_1;
                                array[2] = i_1-1;
                                array[3] = i_1;
                                array[4] = row_1-1;
                                array[5] = row_1;
                                array[6] = j-1;
                                array[7] = j;
                                array[8] = i-1;
                                array[9] = i;

                                int finalSelect = -1;

                                for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                                {

                                    int nd1 = optPossibilities[opt] -1;
                                    int nd2 = optPossibilities[opt+1] -1;
                                    int nd3 = optPossibilities[opt+2] -1;
                                    int nd4 = optPossibilities[opt+3] -1;
                                    int nd5 = optPossibilities[opt+4] -1;
                                    int nd6 = optPossibilities[opt+5] -1;
                                    int nd7 = optPossibilities[opt+6] -1;
                                    int nd8 = optPossibilities[opt+7] -1;
                                    int nd9 = optPossibilities[opt+8] -1;
                                    int nd10 = optPossibilities[opt+9] -1;


                                    int optCandi = opt / 10;
                                    //                                     printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                    newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                            + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                    float opti = oldLength - newLength;
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;

                                        //                                        atomicExch(&(sharedArrayOccupied[j_1-1]), 1);
                                        //                                        atomicExch(&(sharedArrayOccupied[i_1-1]), 1);
                                        //                                        atomicExch(&(sharedArrayOccupied[row_1-1]), 1);
                                        //                                        atomicExch(&(sharedArrayOccupied[j-1]), 1);
                                        //                                        atomicExch(&(sharedArrayOccupied[i-1]), 1);

                                        sharedArrayOccupied[j_1-1] = 1;
                                        sharedArrayOccupied[i_1-1]= 1;
                                        sharedArrayOccupied[row_1-1]= 1;
                                        sharedArrayOccupied[j-1]= 1;
                                        sharedArrayOccupied[i-1]= 1;

                                        break; // stop optpossibilities search when meet the first 5-opt of these 5 edges

                                    }
                                }

                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (unsigned int)sharedArrayTSP[j_1-1].current;
                                    unsigned int node3 = (unsigned int)sharedArrayTSP[i_1-1].current;
                                    unsigned int node5 = (unsigned int)sharedArrayTSP[row_1-1].current;
                                    unsigned int node7 = (unsigned int)sharedArrayTSP[j-1].current;
                                    unsigned int node9 = (unsigned int)sharedArrayTSP[i-1].current;

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    break; // stop row5th loop
                                }



                            }

                        }//end if i j
                    }//end if row < width
                }
            }
        }
    }
    __syncthreads();
}// end K_5opt from 6-opt


/*!
 * \brief 2408 QWB: add parallel 5-opt from 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_2_qiao_stride_iter_firstScheme_onlySharePossibility(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                       Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                       double maxChecks6opt, double maxChecks3opt, double maxChecksoptDivide,
                                                                       double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ QWChar optPossibilities[OPTPOSSIBILITES5OPT];
    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];

    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nVisitedMap[0][m];

        __syncthreads();

    }

    if(local_id < maxChecks6opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;
            id = trunc(id);

            if(id < maxChecks6opt)
            {

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;
                //                outj = trunc(outj);

                if(outi < maxChecks3opt && outj <maxChecks3opt)
                {
                    //WB.Q this way will produce i = j
                    double idid = 9*outi*outi - (1.0)/9;
                    double idMuli = 3*outi;
                    double rowN0 = idMuli + sqrt(idid);
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMuli - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    //WB.Q this way will produce i = j
                    double ididj= 9*outj*outj - (1.0)/9;
                    double idMulj = 3*outj;
                    double rowN0_1 = idMulj + sqrt(ididj);
                    double rowN1_1 = pow(rowN0_1, subtriplicate);
                    double rowN2_1 = idMulj - sqrt(ididj);
                    double rowN3_1 = pow(rowN2_1, subtriplicate);
                    double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                    row = int(rowN4);// check which one works
                    row_1 = int(rowN4_1);

                    if(row < width && row_1 < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow * row *(row-2);
                        double id2opt = fabs(outi - tempRowRowRow);

                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;

                        double tempRowRowj = (double)(row_1-1) / 6;
                        double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
                        double id2opt_1 = fabs(outj - tempRowRowRowj);

                        //WB.Q this way will produce i = j
                        double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
                        i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
                        j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                        //                    //qiao only for test
                        //                    if(id == maxChecks6opt - 2)
                        //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                                && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                                && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                                )
                        {

                            //                            if(row < 20)
                            //                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);

                            bool existingCandidate = 0;
                            if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
                                    nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
                                    nn_source.minRadiusMap[0][i-1] == 1)
                                existingCandidate = 1;

                            if(existingCandidate == 0)
                            {

                                float oldLength = dist(j_1-1, j_1, sharedArrayTSP) + dist(i_1-1, i_1, sharedArrayTSP) + dist(row_1-1, row_1, sharedArrayTSP)
                                        + dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP)     ;



                                float newLength;
                                int array[10];
                                array[0] = j_1-1;
                                array[1] = j_1;
                                array[2] = i_1-1;
                                array[3] = i_1;
                                array[4] = row_1-1;
                                array[5] = row_1;
                                array[6] = j-1;
                                array[7] = j;
                                array[8] = i-1;
                                array[9] = i;

                                int finalSelect = -1;

                                for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                                {

                                    int nd1 = optPossibilities[opt] -1;
                                    int nd2 = optPossibilities[opt+1] -1;
                                    int nd3 = optPossibilities[opt+2] -1;
                                    int nd4 = optPossibilities[opt+3] -1;
                                    int nd5 = optPossibilities[opt+4] -1;
                                    int nd6 = optPossibilities[opt+5] -1;
                                    int nd7 = optPossibilities[opt+6] -1;
                                    int nd8 = optPossibilities[opt+7] -1;
                                    int nd9 = optPossibilities[opt+8] -1;
                                    int nd10 = optPossibilities[opt+9] -1;


                                    int optCandi = opt / 10;
                                    //                                     printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                    newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                            + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                    float opti = oldLength - newLength;
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;

                                        atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);

                                        break; // stop optpossibilities search when meet the first 5-opt of these 5 edges

                                    }
                                }

                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (unsigned int)sharedArrayTSP[j_1-1].current;
                                    unsigned int node3 = (unsigned int)sharedArrayTSP[i_1-1].current;
                                    unsigned int node5 = (unsigned int)sharedArrayTSP[row_1-1].current;
                                    unsigned int node7 = (unsigned int)sharedArrayTSP[j-1].current;
                                    unsigned int node9 = (unsigned int)sharedArrayTSP[i-1].current;

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    break; // stop row5th loop
                                }



                            }

                        }//end if i j
                    }//end if row < width
                }
            }
        }
    }
    __syncthreads();
}// end K_5opt from 6-opt


/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_oneThreadOne6opt_qiao_stride_iter_firstScheme_onlySharePossibility(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                                      Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                                      double maxChecks6opt, double maxChecks3opt, double maxChecksoptDivide,
                                                                                      double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ QWChar optPossibilities[OPTPOSSIBILITES6OPT];

    float iterSharedPossble =  (float)OPTPOSSIBILITES6OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES6OPT)
            optPossibilities[m] = (QWChar) nn_source.evtMap[0][m];

        __syncthreads();
    }

    if(local_id < maxChecks6opt)
    {

        double startId = maxChecksoptDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;
            id = trunc(id);

            if(id < maxChecks6opt)
            {

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;
                //                outj = trunc(outj);

                if(outi < maxChecks3opt && outj <maxChecks3opt)
                {
                    //WB.Q this way will produce i = j
                    double idid = 9*outi*outi - (1.0)/9;
                    double idMuli = 3*outi;
                    double rowN0 = idMuli + sqrt(idid);
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMuli - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    //WB.Q this way will produce i = j
                    double ididj= 9*outj*outj - (1.0)/9;
                    double idMulj = 3*outj;
                    double rowN0_1 = idMulj + sqrt(ididj);
                    double rowN1_1 = pow(rowN0_1, subtriplicate);
                    double rowN2_1 = idMulj - sqrt(ididj);
                    double rowN3_1 = pow(rowN2_1, subtriplicate);
                    double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                    row = int(rowN4);// check which one works
                    row_1 = int(rowN4_1);

                    if(row < width && row_1 < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow * row *(row-2);
                        //   double id2opt = (row-1)*(row)*(row+1)/6 - outi;
                        double id2opt = fabs(outi - tempRowRowRow);

                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;

                        //                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;
                        double tempRowRowj = (double)(row_1-1) / 6;
                        double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
                        double id2opt_1 = fabs(outj - tempRowRowRowj);

                        //WB.Q this way will produce i = j
                        double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
                        i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
                        j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                        //                    //qiao only for test
                        //                    if(id == maxChecks6opt - 2)
                        //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                                && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                                && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                                )
                        {

                            //                            if(row > width -2)
                            //                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);



                            bool existingCandidate = 0;
                            if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
                                    nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
                                    nn_source.minRadiusMap[0][i-1] == 1 ||nn_source.minRadiusMap[0][row-1] == 1)
                                existingCandidate = 1;

                            if(existingCandidate == 0)
                            {


                                double oldLength = dist(j_1-1, j_1, arrayTSP[0]) + dist(i_1-1, i_1, arrayTSP[0]) + dist(row_1-1, row_1, arrayTSP[0])
                                        + dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]) ;

                                double newLength;
                                int array[12];
                                array[0] = j_1-1;
                                array[1] = j_1;
                                array[2] = i_1-1;
                                array[3] = i_1;
                                array[4] = row_1-1;
                                array[5] = row_1;
                                array[6] = j-1;
                                array[7] = j;
                                array[8] = i-1;
                                array[9] = i;
                                array[10] = row-1;
                                array[11] = row;

                                int finalSelect = -1;
                                //  float optimiz = -INFINITY;


                                for(int opt = 0; opt < OPTPOSSIBILITES6OPT; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                                {
                                    int nd1 = optPossibilities[opt] -1;
                                    int nd2 = optPossibilities[opt+1] -1;
                                    int nd3 = optPossibilities[opt+2] -1;
                                    int nd4 = optPossibilities[opt+3] -1;
                                    int nd5 = optPossibilities[opt+4] -1;
                                    int nd6 = optPossibilities[opt+5] -1;
                                    int nd7 = optPossibilities[opt+6] -1;
                                    int nd8 = optPossibilities[opt+7] -1;
                                    int nd9 = optPossibilities[opt+8] -1;
                                    int nd10 = optPossibilities[opt+9] -1;
                                    int nd11 = optPossibilities[opt+10] -1;
                                    int nd12 = optPossibilities[opt+11] -1;

                                    // int optCandi = opt / 12;

                                    newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                            + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0])
                                            + dist(array[nd9],array[nd10], arrayTSP[0]) + dist(array[nd11],array[nd12], arrayTSP[0] );

                                    // float opti = oldLength - newLength;
                                    // if(opti > 0 && opti > optimiz) //qiao
                                    if(oldLength > newLength)
                                    {
                                        finalSelect = (int)opt / 12; //optCandi;
                                        float codekopt = finalSelect * 100 + 6;

                                        //                                        printf("6-opt id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1: (%d,%d,%d,%d,%d,%d)， oldLeng %f, newLeng %f \n",
                                        //                                               id, outi,outj, row, i,j, row_1,i_1,j_1, oldLength, newLength);


                                        unsigned int node1 = (unsigned int)arrayTSP[0][j_1-1].current;
                                        unsigned int node3 = (unsigned int)arrayTSP[0][i_1-1].current;
                                        unsigned int node5 = (unsigned int)arrayTSP[0][row_1-1].current;
                                        unsigned int node7 = (unsigned int)arrayTSP[0][j-1].current;
                                        unsigned int node9 = (unsigned int)arrayTSP[0][i-1].current;
                                        unsigned int node11 = (unsigned int)arrayTSP[0][row-1].current;

                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 12;
                                        result = result | node5;
                                        result = result << 12;
                                        result = result | node7;
                                        result = result << 12;
                                        result = result | node9;
                                        result = result << 12;
                                        result = result | node11;

                                        //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                        //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange



                                        //   optimiz = opti;
                                        atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                        break;
                                    }
                                }

                                //                                if(finalSelect >= 0)
                                //                                {
                                //                                    unsigned int node1 = (unsigned int)arrayTSP[0][j_1-1].current;
                                //                                    unsigned int node3 = (unsigned int)arrayTSP[0][i_1-1].current;
                                //                                    unsigned int node5 = (unsigned int)arrayTSP[0][row_1-1].current;
                                //                                    unsigned int node7 = (unsigned int)arrayTSP[0][j-1].current;
                                //                                    unsigned int node9 = (unsigned int)arrayTSP[0][i-1].current;
                                //                                    unsigned int node11 = (unsigned int)arrayTSP[0][row-1].current;

                                //                                    //                            float localMinChange = nn_source.minRadiusMap[0][node1];
                                //                                    //                            if(optimiz > localMinChange)
                                //                                    {
                                //                                        //12 is restricted by 64 bit and five numbers
                                //                                        unsigned long long result = 0;
                                //                                        result = result | node3;
                                //                                        result = result << 12;
                                //                                        result = result | node5;
                                //                                        result = result << 12;
                                //                                        result = result | node7;
                                //                                        result = result << 12;
                                //                                        result = result | node9;
                                //                                        result = result << 12;
                                //                                        result = result | node11;

                                //                                        float codekopt = finalSelect * 100 + 6;

                                //                                        //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                //                                        //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                //                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                //                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                //                                        //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                //                                    }
                                //                                }
                            }

                        }//end if i j
                    }//end if row < width
                }
            }
        }
    }
    __syncthreads();
}// end K_6optOneThreadOne6opt



/*!
 * \brief 2408 QWB: add parallel variable k-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_VariableKopt_qiao_stride_iter_firstScheme_onlySharePossibility(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                             Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                             double maxChecks2opt, double maxChecks3opt,double maxChecks4opt,
                                                                             double maxChecks6opt, double maxChecksoptDivide,
                                                                             double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    __shared__ QWChar fourOptPossibilities[OPTPOSSIBILITES4OPT];
    if(threadIdx.x < OPTPOSSIBILITES4OPT)
        fourOptPossibilities[threadIdx.x] = nn_source.nodeParentMap[0][threadIdx.x ];
    __syncthreads();

    __shared__ QWChar fiveOptPossibilities[OPTPOSSIBILITES5OPT];
    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            fiveOptPossibilities[m] = nn_source.nVisitedMap[0][m];
        __syncthreads();
    }

    //    __shared__ QWChar sixOptPossibilities[OPTPOSSIBILITES6OPT];
    //    float iterShared6Possble =  (float)OPTPOSSIBILITES6OPT / (float)BLOCKSIZE;
    //    for(int opt = 0; opt < iterShared6Possble; opt++)
    //    {
    //        int m = threadIdx.x + opt*BLOCKSIZE;
    //        if(m < OPTPOSSIBILITES6OPT)
    //            sixOptPossibilities[m] = nn_source.evtMap[0][m];

    //        __syncthreads();
    //    }



    double startId = maxChecksoptDivide * (istride);

    //    if(local_id == 0)
    //        printf("StartID %f, local_id %f \n", startId, local_id);


    for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
    {

        id = id + startId;
        id = trunc(id);


        if(id < maxChecks2opt)
        {

            int i, j;

            //WB.Q this way will produce i = j
            //            i = int(3 + sqrt(8.0f * (double)id + 1.0)) / 2 ;
            //            j = id - (i-2)*(i-1)/2 + 1;
            double temp2sqrt = 8.0 * (double)id + 1.0;
            temp2sqrt = 3 + sqrt(temp2sqrt);
            i = (int)temp2sqrt/2;
            double temp2j = (i-2)*(i-1);
            temp2j = temp2j / 2;
            j = id - temp2j +1;

            if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
            {
                //qiao for test to see 2-opt pairs
                // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);
                bool existingCandidate = 0;
                if(nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                    existingCandidate = 1;

                if(existingCandidate == 0)
                {

                    float oldLength = dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP);
                    float newLength = dist(j-1, i-1, sharedArrayTSP) + dist(j, i, sharedArrayTSP);

                    if(newLength < oldLength)
                    {
                        //                        float optimization = oldLength - newLength;
                        // here automic operation is necessary
                        int node1 = (int)sharedArrayTSP[j-1].current;
                        int node3 = (int)sharedArrayTSP[i-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        float codekopt = 2;

                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result);
                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q this way can work for multi-thread operation
                        atomicExch(&(nn_source.minRadiusMap[0][node1]), 1);
                        atomicExch(&(nn_source.minRadiusMap[0][node3]), 1);

                        continue; //end searching
                    }
                }
                //                else
                //                    continue;//these edges are occupied before
            }

        }


        if(id < maxChecks3opt)
        {

            int row, i, j;
            double subtriplicate = (1.0)/3;

            //WB.Q this way will produce i = j
            double idid = 9*id*id - (1.0)/9 ;
            double idMul3 = 3*id;
            double rowN0 = idMul3 + sqrt(idid );
            double rowN1 = pow(rowN0, subtriplicate);
            double rowN2 = idMul3 - sqrt(idid);
            double rowN3 = pow(rowN2, subtriplicate);
            double rowN4 = rowN1 + rowN3 + 1;

            row = int(rowN4);// check which one works

            if(row < width)
            {

                double tempRowRow = (double)(row-1) / 6;
                double tempRowRowRow = tempRowRow*(row)*(row-2);

                double id2opt = fabs( id - tempRowRowRow );
                //WB.Q this way will produce i = j
                double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                i = int(3 + sqrt(sqrtTemp)) / 2 ;
                j = id2opt - (i-2)*(i-1)/2 + 1;

                if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {

                    bool existingCandidate = 0;
                    if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                        existingCandidate = 1;

                    if(existingCandidate == 0)
                    {

                        double newLength[4];
                        double oldLength = dist(j-1, j,sharedArrayTSP) + dist(i-1, i,sharedArrayTSP) + dist(row-1, row,sharedArrayTSP);
                        newLength[0] = dist(j-1, i,sharedArrayTSP) + dist(row-1, i-1,sharedArrayTSP) + dist(row, j,sharedArrayTSP);
                        newLength[1] = dist(j-1, row-1,sharedArrayTSP) + dist(row, i-1,sharedArrayTSP) + dist(i,j,sharedArrayTSP);
                        newLength[2] = dist(j-1, i-1, sharedArrayTSP) + dist(row-1, j, sharedArrayTSP) + dist(row, i,sharedArrayTSP);
                        newLength[3] = dist(j-1, i, sharedArrayTSP) + dist(row-1, j,sharedArrayTSP) + dist(row,i-1,sharedArrayTSP);

                        int finalSelect = -1;
                        for(int i = 0; i < 4; i++)
                        {
                            float opti = oldLength - newLength[i];

                            if(opti > 0)
                            {
                                finalSelect = i;
                                //                                        optimiz = opti;

                                atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);
                                break;
                            }
                        }
                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (int)sharedArrayTSP[j-1].current;
                            unsigned int node3 = (int)sharedArrayTSP[i-1].current;
                            unsigned int node5 = (int)sharedArrayTSP[row-1].current;


                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 3;

                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            continue;
                        }
                    }
                    //                    else
                    //                        break;//these edges are occupied before

                }//end if i j

            }//end if row < width
        }//end 3opt

        if(id < maxChecks4opt)
        {

            double  outi, outj;
            double sqrtOuti = 8.0 * (double)id + 1.0;
            outi = (3 + sqrt(sqrtOuti)) / 2 ;
            outi = trunc(outi);
            outj = id - (outi-2)*(outi-1)/2 + 1;

            if(outi < maxChecks2opt && outj < maxChecks2opt)
            {

                double sqrtOutIK = 8.0 * (double )outi + 1.0;
                int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                int p = outi - (k-2)*(k-1)/2 + 1;

                double sqrtOutJk = 8.0 * (double)outj + 1.0;
                int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                int w = outj - (j-2)*(j-1)/2 + 1;

                if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                {


                    bool existingCandidate = 0;
                    if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1 ||nn_source.minRadiusMap[0][k-1] == 1)
                        existingCandidate = 1;

                    if(existingCandidate == 0)
                    {

                        float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);
                        float newLength;//25 is fixed for 4-opt
                        int array[8];
                        array[0] = w-1;
                        array[1] = w;
                        array[2] = j-1;
                        array[3] = j;
                        array[4] = p-1;
                        array[5] = p;
                        array[6] = k-1;
                        array[7] = k;

                        int finalSelect = -1;

                        for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                        {

                            int nd1 = (int)fourOptPossibilities[opt] -1;
                            int nd2 = (int)fourOptPossibilities[opt+1] -1;
                            int nd3 = (int)fourOptPossibilities[opt+2] -1;
                            int nd4 = (int)fourOptPossibilities[opt+3] -1;
                            int nd5 = (int)fourOptPossibilities[opt+4] -1;
                            int nd6 = (int)fourOptPossibilities[opt+5] -1;
                            int nd7 = (int)fourOptPossibilities[opt+6] -1;
                            int nd8 = (int)fourOptPossibilities[opt+7] -1;

                            int optCandi = opt / 8;
                            //                                printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                            newLength= dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP) + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP);

                            float opti = oldLength - newLength;
                            if(opti > 0)
                            {
                                finalSelect = optCandi;

                                atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                break;
                            }
                        }


                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                            unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                            unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                            unsigned int node7 = (int)sharedArrayTSP[k-1].current;

                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 4;

                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            continue;
                        }
                    }
                    //                    else
                    //                        break;//these edges are occupied before

                }//end if k<p
            }

        }//end 4opt

        if(id < maxChecks4opt)
        {

            id = trunc(id);
            double outi, outj;
            double sqrtOuti = 8.0 * (double)id + 1.0;
            outi = int(3 + sqrt(sqrtOuti)) / 2 ;
            outj = id - (outi-2)*(outi-1)/2 + 1;

            if(outi < maxChecks2opt && outj < maxChecks2opt)
            {
                double sqrtOutIK = 8.0 * (double )outi + 1.0;
                int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                int p = outi - (k-2)*(k-1)/2 + 1;

                double sqrtOutJk = 8.0 * (double)outj + 1.0;
                int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                int w = outj - (j-2)*(j-1)/2 + 1;

                // if(id > maxChecks4opt-2)
                //     printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                if(k > p && p> j&& j>w && k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                {

                    bool existingCandidate = 0;

                    if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                            ||nn_source.minRadiusMap[0][k-1] == 1)  // ||nn_source.minRadiusMap[0][idRow5th-1] == 1
                        existingCandidate = 1;


                    if(existingCandidate == 0)
                    {

                        float oldLength_4 = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);// + dist(idRow5th-1, idRow5th, sharedArrayTSP);

                        float newLength;
                        int array[10];
                        array[0] = w-1;
                        array[1] = w;
                        array[2] = j-1;
                        array[3] = j;
                        array[4] = p-1;
                        array[5] = p;
                        array[6] = k-1;
                        array[7] = k;

                        for (int idRow5th = k+2; idRow5th < width; idRow5th ++)
                        {

                            //                                if(idRow5th ==  width -1)
                            //                                    printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                            if(nn_source.minRadiusMap[0][idRow5th-1] == 1)
                                continue;


                            float oldLength = oldLength_4;

                            oldLength += dist(idRow5th-1, idRow5th, sharedArrayTSP);

                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {

                                int nd1 = fiveOptPossibilities[opt] -1;
                                int nd2 = fiveOptPossibilities[opt+1] -1;
                                int nd3 = fiveOptPossibilities[opt+2] -1;
                                int nd4 = fiveOptPossibilities[opt+3] -1;
                                int nd5 = fiveOptPossibilities[opt+4] -1;
                                int nd6 = fiveOptPossibilities[opt+5] -1;
                                int nd7 = fiveOptPossibilities[opt+6] -1;
                                int nd8 = fiveOptPossibilities[opt+7] -1;
                                int nd9 = fiveOptPossibilities[opt+8] -1;
                                int nd10 = fiveOptPossibilities[opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                        + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0)
                                {
                                    finalSelect = optCandi;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                    break; // stop optpossibilities search when meet the first 5-opt of these 5 edges
                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;
                                unsigned int node9 = (int)sharedArrayTSP[idRow5th-1].current;

                                unsigned long long result = 0;
                                result = result | node3;
                                result = result << 16;
                                result = result | node5;
                                result = result << 16;
                                result = result | node7;
                                result = result << 16;
                                result = result | node9;

                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                codekopt = finalSelect * 100 + 5;

                                //   printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                //        node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                break; // stop row5th loop
                            }
                        }
                    }
                }
            }
        }


        //        if(id < maxChecks6opt)
        //        {
        //            double  outi, outj;
        //            int row,row_1, i, j, i_1, j_1;
        //            double subtriplicate = (1.0)/3;

        //            //WB.Q this way will produce i = j
        //            double sqrtOuti = 8.0 * (double)id + 1.0;
        //            outi = (3 + sqrt(sqrtOuti)) / 2 ;
        //            outi = trunc(outi);
        //            outj = id - (outi-2)*(outi-1)/2 + 1;
        //            //                outj = trunc(outj);

        //            if(outi < maxChecks3opt && outj <maxChecks3opt)
        //            {
        //                //WB.Q this way will produce i = j
        //                double idid = 9*outi*outi - (1.0)/9;
        //                double idMuli = 3*outi;
        //                double rowN0 = idMuli + sqrt(idid);
        //                double rowN1 = pow(rowN0, subtriplicate);
        //                double rowN2 = idMuli - sqrt(idid);
        //                double rowN3 = pow(rowN2, subtriplicate);
        //                double rowN4 = rowN1 + rowN3 + 1;

        //                //WB.Q this way will produce i = j
        //                double ididj= 9*outj*outj - (1.0)/9;
        //                double idMulj = 3*outj;
        //                double rowN0_1 = idMulj + sqrt(ididj);
        //                double rowN1_1 = pow(rowN0_1, subtriplicate);
        //                double rowN2_1 = idMulj - sqrt(ididj);
        //                double rowN3_1 = pow(rowN2_1, subtriplicate);
        //                double rowN4_1 = rowN1_1 + rowN3_1 + 1;

        //                row = int(rowN4);// check which one works
        //                row_1 = int(rowN4_1);

        //                if(row < width && row_1 < width)
        //                {
        //                    double tempRowRow = (double)(row-1) / 6;
        //                    double tempRowRowRow = tempRowRow * row *(row-2);
        //                    //                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;
        //                    double id2opt = fabs(outi - tempRowRowRow);

        //                    //WB.Q this way will produce i = j
        //                    double sqrtTemp = 8.0 * (double)id2opt + 1.0;
        //                    i = int(3 + sqrt(sqrtTemp)) / 2 ;
        //                    j = id2opt - (i-2)*(i-1)/2 + 1;

        //                    //                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;
        //                    double tempRowRowj = (double)(row_1-1) / 6;
        //                    double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
        //                    double id2opt_1 = fabs(outj - tempRowRowRowj);

        //                    //WB.Q this way will produce i = j
        //                    double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
        //                    i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
        //                    j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


        //                    //                    //qiao only for test
        //                    //                    if(id == maxChecks6opt - 2)
        //                    //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


        //                    if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
        //                            && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
        //                            && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
        //                            )
        //                    {

        //                        //                            if(row > width -2)
        //                        //                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


        //                        bool existingCandidate = 0;
        //                        if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
        //                                nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
        //                                nn_source.minRadiusMap[0][i-1] == 1 ||nn_source.minRadiusMap[0][row-1] == 1)
        //                            existingCandidate = 1;

        //                        if(existingCandidate == 0)
        //                        {

        //                            float oldLength = dist(j_1-1, j_1, sharedArrayTSP) + dist(i_1-1, i_1, sharedArrayTSP) + dist(row_1-1, row_1, sharedArrayTSP)
        //                                    + dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP) + dist(row-1, row, sharedArrayTSP)    ;

        //                            float newLength;
        //                            int array[12];
        //                            array[0] = j_1-1;
        //                            array[1] = j_1;
        //                            array[2] = i_1-1;
        //                            array[3] = i_1;
        //                            array[4] = row_1-1;
        //                            array[5] = row_1;
        //                            array[6] = j-1;
        //                            array[7] = j;
        //                            array[8] = i-1;
        //                            array[9] = i;
        //                            array[10] = row-1;
        //                            array[11] = row;

        //                            int finalSelect = -1;
        //                            //                                float optimiz = -INFINITY;


        //                            for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
        //                            {
        //                                int nd1 = nn_source.evtMap[0][opt] -1;
        //                                int nd2 = nn_source.evtMap[0][opt+1] -1;
        //                                int nd3 = nn_source.evtMap[0][opt+2] -1;
        //                                int nd4 = nn_source.evtMap[0][opt+3] -1;
        //                                int nd5 = nn_source.evtMap[0][opt+4] -1;
        //                                int nd6 = nn_source.evtMap[0][opt+5] -1;
        //                                int nd7 = nn_source.evtMap[0][opt+6] -1;
        //                                int nd8 = nn_source.evtMap[0][opt+7] -1;
        //                                int nd9 = nn_source.evtMap[0][opt+8] -1;
        //                                int nd10 = nn_source.evtMap[0][opt+9] -1;
        //                                int nd11 = nn_source.evtMap[0][opt+10] -1;
        //                                int nd12 = nn_source.evtMap[0][opt+11] -1;

        //                                int optCandi = opt / 12;

        //                                newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
        //                                        + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP)
        //                                        + dist(array[nd9],array[nd10], sharedArrayTSP) + dist(array[nd11],array[nd12], sharedArrayTSP );

        //                                float opti = oldLength - newLength;
        //                                //                                    if(opti > 0 && opti > optimiz)
        //                                if(opti > 0)
        //                                {
        //                                    finalSelect = optCandi;
        //                                    //                                        optimiz = opti;
        //                                    atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
        //                                    atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
        //                                    atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
        //                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
        //                                    atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
        //                                    atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

        //                                    break;
        //                                }
        //                            }

        //                            if(finalSelect >= 0)
        //                            {
        //                                int node1 = (int)sharedArrayTSP[j_1-1].current;
        //                                int node3 = (int)sharedArrayTSP[i_1-1].current;
        //                                int node5 = (int)sharedArrayTSP[row_1-1].current;
        //                                int node7 = (int)sharedArrayTSP[j-1].current;
        //                                int node9 = (int)sharedArrayTSP[i-1].current;
        //                                int node11 = (int)sharedArrayTSP[row-1].current;

        //                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];
        //                                //                            if(optimiz > localMinChange)
        //                                {
        //                                    //12 is restricted by 64 bit and five numbers
        //                                    unsigned long long result = 0;
        //                                    result = result | node3;
        //                                    result = result << 12;
        //                                    result = result | node5;
        //                                    result = result << 12;
        //                                    result = result | node7;
        //                                    result = result << 12;
        //                                    result = result | node9;
        //                                    result = result << 12;
        //                                    result = result | node11;

        //                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
        //                                    codekopt = finalSelect * 100 + 6;

        //                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
        //                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
        //                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
        //                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
        //                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
        //                                }
        //                            }
        //                        }

        //                    }//end if i j
        //                }//end if row < width
        //            }
        //        }

    }

    __syncthreads();
}// end K_6optOneThreadOne6opt



/*!
 * \brief 2408 QWB: add parallel variable k-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_VariableKopt_qiao_stride_iter_bestScheme_onlySharePossibility(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                            Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                            double maxChecks2opt, double maxChecks3opt,double maxChecks4opt,
                                                                            double maxChecks6opt, double maxChecksoptDivide,
                                                                            double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    __shared__ QWChar fourOptPossibilities[OPTPOSSIBILITES4OPT];
    if(threadIdx.x < OPTPOSSIBILITES4OPT)
        fourOptPossibilities[threadIdx.x] = nn_source.nodeParentMap[0][threadIdx.x ];
    __syncthreads();

    __shared__ QWChar fiveOptPossibilities[OPTPOSSIBILITES5OPT];
    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            fiveOptPossibilities[m] = nn_source.nVisitedMap[0][m];
        __syncthreads();
    }

    __shared__ QWChar sixOptPossibilities[OPTPOSSIBILITES6OPT];
    float iterShared6Possble =  (float)OPTPOSSIBILITES6OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared6Possble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < OPTPOSSIBILITES6OPT)
            sixOptPossibilities[m] = nn_source.evtMap[0][m];

        __syncthreads();
    }



    double startId = maxChecksoptDivide * (istride);

    for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
    {

        id = id + startId;
        id = trunc(id);


        if(id < maxChecks2opt)
        {

            int i, j;
            id = trunc(id);

            //WB.Q this way will produce i = j
            //            i = int(3 + sqrt(8.0f * (double)id + 1.0)) / 2 ;
            //            j = id - (i-2)*(i-1)/2 + 1;
            double temp2sqrt = 8.0 * (double)id + 1.0;
            temp2sqrt = 3 + sqrt(temp2sqrt);
            i = (int)temp2sqrt/2;
            double temp2j = (i-2)*(i-1);
            temp2j = temp2j / 2;
            j = id - temp2j +1;

            if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
            {

                {

                    float oldLength = dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP);
                    float newLength = dist(j-1, i-1, sharedArrayTSP) + dist(j, i, sharedArrayTSP);

                    if(newLength < oldLength)
                    {
                        float optimization = oldLength - newLength;
                        // here automic operation is necessary
                        int node1 = (int)sharedArrayTSP[j-1].current;
                        int node3 = (int)sharedArrayTSP[i-1].current;

                        float localMinChange = nn_source.minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            unsigned long long result = 0;
                            result = result | node3;
                            float codekopt = 2;

                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result);
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q this way can work for multi-thread operation

                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                        }
                    }
                }
            }

        }


        if(id < maxChecks3opt)
        {

            int row, i, j;
            double subtriplicate = (1.0)/3;

            //WB.Q this way will produce i = j
            double idid = 9*id*id - (1.0)/9 ;
            double idMul3 = 3*id;
            double rowN0 = idMul3 + sqrt(idid );
            double rowN1 = pow(rowN0, subtriplicate);
            double rowN2 = idMul3 - sqrt(idid);
            double rowN3 = pow(rowN2, subtriplicate);
            double rowN4 = rowN1 + rowN3 + 1;

            row = int(rowN4);// check which one works

            if(row < width)
            {

                double tempRowRow = (double)(row-1) / 6;
                double tempRowRowRow = tempRowRow*(row)*(row-2);


                double id2opt = fabs( id - tempRowRowRow );
                //WB.Q this way will produce i = j
                double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                i = int(3 + sqrt(sqrtTemp)) / 2 ;
                j = id2opt - (i-2)*(i-1)/2 + 1;

                if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {

                    double newLength[4];
                    double oldLength = dist(j-1, j,sharedArrayTSP) + dist(i-1, i,sharedArrayTSP) + dist(row-1, row,sharedArrayTSP);
                    newLength[0] = dist(j-1, i,sharedArrayTSP) + dist(row-1, i-1,sharedArrayTSP) + dist(row, j,sharedArrayTSP);
                    newLength[1] = dist(j-1, row-1,sharedArrayTSP) + dist(row, i-1,sharedArrayTSP) + dist(i,j,sharedArrayTSP);
                    newLength[2] = dist(j-1, i-1, sharedArrayTSP) + dist(row-1, j, sharedArrayTSP) + dist(row, i,sharedArrayTSP);
                    newLength[3] = dist(j-1, i, sharedArrayTSP) + dist(row-1, j,sharedArrayTSP) + dist(row,i-1,sharedArrayTSP);

                    int finalSelect = -1;
                    double optimiz = -INFINITY;
                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength - newLength[i];

                        if(opti > 0 && opti > optimiz)
                        {
                            finalSelect = i;
                            optimiz = opti;
                        }
                    }
                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (int)sharedArrayTSP[j-1].current;
                        double localMinChange = nn_source.minRadiusMap[0][node1];
                        double optimization = oldLength - newLength[finalSelect];

                        if(optimization > localMinChange)
                        {
                            unsigned int node3 = (int)sharedArrayTSP[i-1].current;
                            unsigned int node5 = (int)sharedArrayTSP[row-1].current;

                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 3;

                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);

                        }
                    }


                }//end if i j

            }//end if row < width
        }//end 3opt

        if(id < maxChecks4opt)
        {

            double  outi, outj;
            double sqrtOuti = 8.0 * (double)id + 1.0;
            outi = (3 + sqrt(sqrtOuti)) / 2 ;
            outi = trunc(outi);
            outj = id - (outi-2)*(outi-1)/2 + 1;

            if(outi < maxChecks2opt && outj < maxChecks2opt)
            {

                double sqrtOutIK = 8.0 * (double )outi + 1.0;
                int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                int p = outi - (k-2)*(k-1)/2 + 1;

                double sqrtOutJk = 8.0 * (double)outj + 1.0;
                int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                int w = outj - (j-2)*(j-1)/2 + 1;

                if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                {

                    float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);
                    float newLength;//25 is fixed for 4-opt
                    int array[8];
                    array[0] = w-1;
                    array[1] = w;
                    array[2] = j-1;
                    array[3] = j;
                    array[4] = p-1;
                    array[5] = p;
                    array[6] = k-1;
                    array[7] = k;

                    int finalSelect = -1;
                    float optimiz = -INFINITY;

                    for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                    {

                        int nd1 = (int)fourOptPossibilities[opt] -1;
                        int nd2 = (int)fourOptPossibilities[opt+1] -1;
                        int nd3 = (int)fourOptPossibilities[opt+2] -1;
                        int nd4 = (int)fourOptPossibilities[opt+3] -1;
                        int nd5 = (int)fourOptPossibilities[opt+4] -1;
                        int nd6 = (int)fourOptPossibilities[opt+5] -1;
                        int nd7 = (int)fourOptPossibilities[opt+6] -1;
                        int nd8 = (int)fourOptPossibilities[opt+7] -1;

                        int optCandi = opt / 8;
                        //                                printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                        newLength= dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP) + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP);

                        float opti = oldLength - newLength;
                        if(opti > 0 && opti > optimiz)
                        {
                            finalSelect = optCandi;
                            optimiz = opti;
                        }
                    }


                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (int)sharedArrayTSP[w-1].current;

                        float localMinChange = nn_source.minRadiusMap[0][node1];

                        if(optimiz > localMinChange)
                        {
                            unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                            unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                            unsigned int node7 = (int)sharedArrayTSP[k-1].current;

                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 4;

                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);

                        }
                    }


                }//end if k<p
            }

        }//end 4opt


        if(id < maxChecks4opt)
        {

            id = trunc(id);
            double outi, outj;
            double sqrtOuti = 8.0 * (double)id + 1.0;
            outi = (3 + sqrt(sqrtOuti)) / 2 ;
            outi = trunc(outi);
            outj = id - (outi-2)*(outi-1)/2 + 1;

            if(outi < maxChecks2opt && outj < maxChecks2opt)
            {
                double sqrtOutIK = 8.0 * (double )outi + 1.0;
                int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                int p = outi - (k-2)*(k-1)/2 + 1;

                double sqrtOutJk = 8.0 * (double)outj + 1.0;
                int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                int w = outj - (j-2)*(j-1)/2 + 1;


                if(k > p && p> j&& j>w && k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                {

                    float oldLength_4 = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);// + dist(idRow5th-1, idRow5th, sharedArrayTSP);

                    float newLength;
                    int array[10];
                    array[0] = w-1;
                    array[1] = w;
                    array[2] = j-1;
                    array[3] = j;
                    array[4] = p-1;
                    array[5] = p;
                    array[6] = k-1;
                    array[7] = k;

                    for (int idRow5th = k+2; idRow5th < width; idRow5th ++)
                    {

                        float oldLength = oldLength_4;

                        oldLength += dist(idRow5th-1, idRow5th, sharedArrayTSP);

                        array[8] = idRow5th-1;
                        array[9] = idRow5th;

                        int finalSelect = -1;
                        float optimiz = -1;

                        for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                        {

                            int nd1 = fiveOptPossibilities[opt] -1;
                            int nd2 = fiveOptPossibilities[opt+1] -1;
                            int nd3 = fiveOptPossibilities[opt+2] -1;
                            int nd4 = fiveOptPossibilities[opt+3] -1;
                            int nd5 = fiveOptPossibilities[opt+4] -1;
                            int nd6 = fiveOptPossibilities[opt+5] -1;
                            int nd7 = fiveOptPossibilities[opt+6] -1;
                            int nd8 = fiveOptPossibilities[opt+7] -1;
                            int nd9 = fiveOptPossibilities[opt+8] -1;
                            int nd10 = fiveOptPossibilities[opt+9] -1;


                            int optCandi = opt / 10;
                            // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                            newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                    + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                            float opti = oldLength - newLength;
                            if(opti > 0 && opti > optimiz)
                            {
                                finalSelect = optCandi;
                                optimiz = opti;
                            }
                        }

                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                            float localMinChange = nn_source.minRadiusMap[0][node1];

                            if(optimiz > localMinChange)
                            {
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;
                                unsigned int node9 = (int)sharedArrayTSP[idRow5th-1].current;

                                unsigned long long result = 0;
                                result = result | node3;
                                result = result << 16;
                                result = result | node5;
                                result = result << 16;
                                result = result | node7;
                                result = result << 16;
                                result = result | node9;

                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                codekopt = finalSelect * 100 + 5;

                                //   printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                //        node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);

                            }
                        }

                    }

                }
            }
        }

        //        if(id < maxChecks6opt)
        //        {
        //            id = trunc(id);

        //            double  outi, outj;
        //            int row,row_1, i, j, i_1, j_1;
        //            double subtriplicate = (1.0)/3;

        //            //WB.Q this way will produce i = j
        //            double sqrtOuti = 8.0 * (double)id + 1.0;
        //            outi = (3 + sqrt(sqrtOuti)) / 2 ;
        //            outi = trunc(outi);
        //            outj = id - (outi-2)*(outi-1)/2 + 1;
        //            //                outj = trunc(outj);

        //            if(outi < maxChecks3opt && outj <maxChecks3opt)
        //            {
        //                //WB.Q this way will produce i = j
        //                double idid = 9*outi*outi - (1.0)/9;
        //                double idMuli = 3*outi;
        //                double rowN0 = idMuli + sqrt(idid);
        //                double rowN1 = pow(rowN0, subtriplicate);
        //                double rowN2 = idMuli - sqrt(idid);
        //                double rowN3 = pow(rowN2, subtriplicate);
        //                double rowN4 = rowN1 + rowN3 + 1;

        //                //WB.Q this way will produce i = j
        //                double ididj= 9*outj*outj - (1.0)/9;
        //                double idMulj = 3*outj;
        //                double rowN0_1 = idMulj + sqrt(ididj);
        //                double rowN1_1 = pow(rowN0_1, subtriplicate);
        //                double rowN2_1 = idMulj - sqrt(ididj);
        //                double rowN3_1 = pow(rowN2_1, subtriplicate);
        //                double rowN4_1 = rowN1_1 + rowN3_1 + 1;

        //                row = int(rowN4);// check which one works
        //                row_1 = int(rowN4_1);

        //                if(row < width && row_1 < width)
        //                {
        //                    double tempRowRow = (double)(row-1) / 6;
        //                    double tempRowRowRow = tempRowRow * row *(row-2);
        //                    //                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;
        //                    double id2opt = fabs(outi - tempRowRowRow);

        //                    //WB.Q this way will produce i = j
        //                    double sqrtTemp = 8.0 * (double)id2opt + 1.0;
        //                    i = int(3 + sqrt(sqrtTemp)) / 2 ;
        //                    j = id2opt - (i-2)*(i-1)/2 + 1;

        //                    //                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;
        //                    double tempRowRowj = (double)(row_1-1) / 6;
        //                    double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
        //                    double id2opt_1 = fabs(outj - tempRowRowRowj);

        //                    //WB.Q this way will produce i = j
        //                    double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
        //                    i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
        //                    j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


        //                    //                    //qiao only for test
        //                    //                    if(id == maxChecks6opt - 2)
        //                    //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


        //                    if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
        //                            && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
        //                            && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
        //                            )
        //                    {

        //                        //                            if(row > width -2)
        //                        //                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);

        //                        float oldLength = dist(j_1-1, j_1, sharedArrayTSP) + dist(i_1-1, i_1, sharedArrayTSP) + dist(row_1-1, row_1, sharedArrayTSP)
        //                                + dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP) + dist(row-1, row, sharedArrayTSP)    ;

        //                        float newLength;
        //                        int array[12];
        //                        array[0] = j_1-1;
        //                        array[1] = j_1;
        //                        array[2] = i_1-1;
        //                        array[3] = i_1;
        //                        array[4] = row_1-1;
        //                        array[5] = row_1;
        //                        array[6] = j-1;
        //                        array[7] = j;
        //                        array[8] = i-1;
        //                        array[9] = i;
        //                        array[10] = row-1;
        //                        array[11] = row;

        //                        int finalSelect = -1;
        //                        float optimiz = -1;

        //                        for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
        //                        {
        //                            int nd1 = nn_source.evtMap[0][opt] -1;
        //                            int nd2 = nn_source.evtMap[0][opt+1] -1;
        //                            int nd3 = nn_source.evtMap[0][opt+2] -1;
        //                            int nd4 = nn_source.evtMap[0][opt+3] -1;
        //                            int nd5 = nn_source.evtMap[0][opt+4] -1;
        //                            int nd6 = nn_source.evtMap[0][opt+5] -1;
        //                            int nd7 = nn_source.evtMap[0][opt+6] -1;
        //                            int nd8 = nn_source.evtMap[0][opt+7] -1;
        //                            int nd9 = nn_source.evtMap[0][opt+8] -1;
        //                            int nd10 = nn_source.evtMap[0][opt+9] -1;
        //                            int nd11 = nn_source.evtMap[0][opt+10] -1;
        //                            int nd12 = nn_source.evtMap[0][opt+11] -1;

        //                            int optCandi = opt / 12;

        //                            newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
        //                                    + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP)
        //                                    + dist(array[nd9],array[nd10], sharedArrayTSP) + dist(array[nd11],array[nd12], sharedArrayTSP );

        //                            float opti = oldLength - newLength;
        //                            if(opti > 0 && opti > optimiz)
        //                            {
        //                                finalSelect = optCandi;
        //                                optimiz = opti;

        //                            }

        //                        }

        //                        if(finalSelect >= 0)
        //                        {

        //                            int node1 = (int)sharedArrayTSP[j_1-1].current;
        //                            float localMinChange = nn_source.minRadiusMap[0][node1];

        //                            if(optimiz > localMinChange)
        //                            {

        //                                int node3 = (int)sharedArrayTSP[i_1-1].current;
        //                                int node5 = (int)sharedArrayTSP[row_1-1].current;
        //                                int node7 = (int)sharedArrayTSP[j-1].current;
        //                                int node9 = (int)sharedArrayTSP[i-1].current;
        //                                int node11 = (int)sharedArrayTSP[row-1].current;

        //                                //12 is restricted by 64 bit and five numbers
        //                                unsigned long long result = 0;
        //                                result = result | node3;
        //                                result = result << 12;
        //                                result = result | node5;
        //                                result = result << 12;
        //                                result = result | node7;
        //                                result = result << 12;
        //                                result = result | node9;
        //                                result = result << 12;
        //                                result = result | node11;

        //                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
        //                                codekopt = finalSelect * 100 + 6;

        //                                atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
        //                                atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
        //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
        //                            }
        //                        }

        //                    }//end if i j
        //                }//end if row < width

        //            }

        //        }


    }

    __syncthreads();
}// end K_6optOneThreadOne6opt



/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_qiao_stride_iter_firstScheme_ShareNoOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                              Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                              double maxChecks6opt, double maxChecks3opt, double maxChecksoptDivide,
                                                              double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ QWChar optPossibilities[OPTPOSSIBILITES6OPT];
    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];

    float iterSharedPossble =  (float)OPTPOSSIBILITES6OPT / (float)BLOCKSIZE;
    float iterShared = (float)width / (float)BLOCKSIZE;

    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES6OPT)
            optPossibilities[m] = (QWChar) nn_source.evtMap[0][m];

        __syncthreads();
    }

    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];
        }
        __syncthreads();
    }




    if(local_id < maxChecks6opt)
    {

        double startId = maxChecksoptDivide * (istride);

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;



            if(id < maxChecks6opt)
            {
                id = trunc(id);
                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;
                //                outj = trunc(outj);

                if(outi < maxChecks3opt && outj <maxChecks3opt)
                {
                    //WB.Q this way will produce i = j
                    double idid = 9*outi*outi - (1.0)/9;
                    double idMuli = 3*outi;
                    double rowN0 = idMuli + sqrt(idid);
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMuli - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    //WB.Q this way will produce i = j
                    double ididj= 9*outj*outj - (1.0)/9;
                    double idMulj = 3*outj;
                    double rowN0_1 = idMulj + sqrt(ididj);
                    double rowN1_1 = pow(rowN0_1, subtriplicate);
                    double rowN2_1 = idMulj - sqrt(ididj);
                    double rowN3_1 = pow(rowN2_1, subtriplicate);
                    double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                    row = int(rowN4);// check which one works
                    row_1 = int(rowN4_1);

                    if(row < width && row_1 < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow * row *(row-2);
                        //                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;
                        double id2opt = fabs(outi - tempRowRowRow);

                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;

                        //                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;
                        double tempRowRowj = (double)(row_1-1) / 6;
                        double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
                        double id2opt_1 = fabs(outj - tempRowRowRowj);

                        //WB.Q this way will produce i = j
                        double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
                        i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
                        j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                        //                    //qiao only for test
                        //                    if(id == maxChecks6opt - 2)
                        //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                                && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                                && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                                )
                        {

                            //                            if(row > width -2)
                            //                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                            bool existingCandidate = 0;
                            if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
                                    nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
                                    nn_source.minRadiusMap[0][i-1] == 1 ||nn_source.minRadiusMap[0][row-1] == 1)
                                existingCandidate = 1;

                            if(existingCandidate == 0)
                            {

                                float oldLength = dist(j_1-1, j_1, sharedArrayTSP) + dist(i_1-1, i_1, sharedArrayTSP) + dist(row_1-1, row_1, sharedArrayTSP)
                                        + dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP) + dist(row-1, row, sharedArrayTSP)    ;

                                float newLength;
                                int array[12];
                                array[0] = j_1-1;
                                array[1] = j_1;
                                array[2] = i_1-1;
                                array[3] = i_1;
                                array[4] = row_1-1;
                                array[5] = row_1;
                                array[6] = j-1;
                                array[7] = j;
                                array[8] = i-1;
                                array[9] = i;
                                array[10] = row-1;
                                array[11] = row;

                                int finalSelect = -1;
                                //                                float optimiz = -INFINITY;


                                for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                                {
                                    int nd1 = optPossibilities[opt] -1;
                                    int nd2 = optPossibilities[opt+1] -1;
                                    int nd3 = optPossibilities[opt+2] -1;
                                    int nd4 = optPossibilities[opt+3] -1;
                                    int nd5 = optPossibilities[opt+4] -1;
                                    int nd6 = optPossibilities[opt+5] -1;
                                    int nd7 = optPossibilities[opt+6] -1;
                                    int nd8 = optPossibilities[opt+7] -1;
                                    int nd9 = optPossibilities[opt+8] -1;
                                    int nd10 = optPossibilities[opt+9] -1;
                                    int nd11 = optPossibilities[opt+10] -1;
                                    int nd12 = optPossibilities[opt+11] -1;

                                    int optCandi = opt / 12;

                                    newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                            + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP)
                                            + dist(array[nd9],array[nd10], sharedArrayTSP) + dist(array[nd11],array[nd12], sharedArrayTSP );

                                    float opti = oldLength - newLength;
                                    //                                    if(opti > 0 && opti > optimiz)
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;
                                        //                                        optimiz = opti;
                                        atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                        break;
                                    }
                                }

                                if(finalSelect >= 0)
                                {
                                    int node1 = (int)sharedArrayTSP[j_1-1].current;
                                    int node3 = (int)sharedArrayTSP[i_1-1].current;
                                    int node5 = (int)sharedArrayTSP[row_1-1].current;
                                    int node7 = (int)sharedArrayTSP[j-1].current;
                                    int node9 = (int)sharedArrayTSP[i-1].current;
                                    int node11 = (int)sharedArrayTSP[row-1].current;

                                    //                            float localMinChange = nn_source.minRadiusMap[0][node1];
                                    //                            if(optimiz > localMinChange)
                                    {
                                        //12 is restricted by 64 bit and five numbers
                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 12;
                                        result = result | node5;
                                        result = result << 12;
                                        result = result | node7;
                                        result = result << 12;
                                        result = result | node9;
                                        result = result << 12;
                                        result = result | node11;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 6;

                                        //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                        //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                    }
                                }
                            }

                        }//end if i j
                    }//end if row < width
                }
            }
        }
    }
    __syncthreads();
}// end K_6optOneThreadOne6opt



//! 0617 QWB: add parallel 2opt one thread one 2-opt with Rocki
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne2opt_RockiSmall( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                  unsigned long maxChecks,
                                                  unsigned int iter
                                                  ) {

    //    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
    //                                 BLOCKSIZE,
    //                                 16,
    //                                 GRIDSIZE, //for rocki large global
    //                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
    //                                 nn_source.adaptiveMap.width);



    KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks, 1);

    //                K_2opt_oneThreadOne2opt_rockiSmall_shared _KER_CALL_(b, t) (nn_source.densityMap, nn_source.minRadiusMap, linkCoordTourGpu, maxChecks, iter);
    K_2opt_oneThreadOne2opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks, iter);// global

    cudaChk(cudaPeekAtLastError());
}



//! 0617 QWB: add parallel 2opt one thread one 2-opt with Rocki
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne2opt_Rocki_iterStride( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                        Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                        double max2optChecks, double maxChecksoOptDivide,
                                                        double iter, double istride
                                                        ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE,
                                 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);

    //  K_2opt_oneThreadOne2opt_qiaoIterStride_best_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, max2optChecks, maxChecksoOptDivide, iter, istride);
//                  K_2opt_oneThreadOne2opt_qiaoIterStride _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, max2optChecks, maxChecksoOptDivide, iter, istride);// global

    //qiao  selected for iterkoptimal correct lest time than first with occupy 3.5s
    K_2opt_oneThreadOne2opt_qiaoIterStride_best _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, max2optChecks, maxChecksoOptDivide, iter, istride);// global the faster way to converage that iter one node select the first 2-opt

    //qiao correct more time than best with no occupy 4.3s
    //    K_2opt_oneThreadOne2opt_qiaoIterStride_first _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, max2optChecks, maxChecksoOptDivide, iter, istride);// global the faster way to converage that iter one node select the first 2-opt


    cudaChk(cudaPeekAtLastError());
}

//! 0617 QWB: add parallel 2opt one thread one 2-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne4opt_qiao_iterStride(cudaStream_t& stream,
                                                      NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                      Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                      double  maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                      double iter, double istride
                                                      ) {

    //qiao here does not run correctly
    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //                                 maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);
    //    cout << "grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " , INT_MAX="
    //         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    //correct  qiao find each 4 edge's best among 200 re-connection schemes, and select the best for each node1 on global mem
    //        K_4opt_oneThreadOne4opt_qiaoIterStride_Best  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //correct qiao selected 66359 qiao find each 4 edge's best among 200 re-connection schemes, and select the best for each node1 on global mem, use shared mem for arrayTSP
    //        K_4opt_oneThreadOne4opt_qiaoIterStride_Best_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global
    //    K_4opt_oneThreadOne4opt_qiaoIterStride_Best_shared <<< b, t, 49152, stream >>> (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    K_4opt_4opt_qiaoIterStride_Best_sharePossible _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global
    //    K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 24576, stream >>> (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //correct 62940ms 62926ms 71opts select the first opt of 208 schemes
    //    K_4opt_oneThreadOne4opt_qiaoIterStride _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //correct the least time consuming over above three version 55899, produce 33 opts qiao test with all sharedmem
    //    K_4opt_oneThreadOne4opt_qiaoIterStride_sharedwithOccup _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //correct 55890ms 68opts qiao test sharedmem occupy on global  time to iterkoptimal 8.36633e+06
    //    K_4opt_oneThreadOne4opt_qiaoIterStride_shared_noShareOccupy _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //        KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks4opt, 1);
    //        double blocks =  (maxChecks4opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //        cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //             << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;
    //        K_4opt_oneThreadOne4opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, iter);// global

    cudaChk(cudaPeekAtLastError());
}



//! 0617 QWB: add parallel 2opt one thread one 2-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne4opt_qiao_iterStride_d1(cudaStream_t stream1,
                                                         NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                         Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                         double  maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                         double iter, double istride
                                                         ) {

    //qiao here does not run correctly
    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //                                 maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);
    cout << "grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " , INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    //correct  qiao find each 4 edge's best among 200 re-connection schemes, and select the best for each node1 on global mem
    //        K_4opt_oneThreadOne4opt_qiaoIterStride_Best  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //correct qiao selected 66359 qiao find each 4 edge's best among 200 re-connection schemes, and select the best for each node1 on global mem, use shared mem for arrayTSP
    //        K_4opt_oneThreadOne4opt_qiaoIterStride_Best_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global
    K_4opt_oneThreadOne4opt_qiaoIterStride_Best_shared <<< b, t, 24576, stream1 >>> (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //    K_4opt_4opt_qiaoIterStride_Best_sharePossible _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global
    //    K_4opt_4opt_qiaoIterStride_Best_sharePossible <<< b, t, 49152, streams[1] >>> (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //correct 62940ms 62926ms 71opts select the first opt of 208 schemes
    //    K_4opt_oneThreadOne4opt_qiaoIterStride _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //correct the least time consuming over above three version 55899, produce 33 opts qiao test with all sharedmem
    //    K_4opt_oneThreadOne4opt_qiaoIterStride_sharedwithOccup _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //correct 55890ms 68opts qiao test sharedmem occupy on global  time to iterkoptimal 8.36633e+06
    //    K_4opt_oneThreadOne4opt_qiaoIterStride_shared_noShareOccupy _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //        KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks4opt, 1);
    //        double blocks =  (maxChecks4opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //        cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //             << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;
    //        K_4opt_oneThreadOne4opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, iter);// global

    cudaChk(cudaPeekAtLastError());
}


//! 0617 QWB: add parallel 2opt one thread one 2-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne5opt_RockiSmall( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                  int n, double maxChecks4opt,double maxChecks2opt,
                                                  unsigned int iter
                                                  ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //                                 maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);
    //        double blocks =  (maxChecks4opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //        cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //             << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;
    K_5opt_oneThreadOne5opt_rockiSmall_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n ,maxChecks4opt, maxChecks2opt,iter);// global




    //    KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks4opt, 1);
    //    double blocks =  (maxChecks4opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //    cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;
    //    K_5opt_oneThreadOne5opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n ,maxChecks4opt, maxChecks2opt,iter);// global

    cudaChk(cudaPeekAtLastError());
}




//! 0617 QWB: add parallel 5opt
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne5opt_qiao_StrideIter( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                       Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                       int n, double  maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                       double iter, double istride
                                                       ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //                                 maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);

    //    cout << "grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " , INT_MAX="
    //         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;



    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_shared  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n , maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n , maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //    K_5opt_oneThreadOne5opt_qiao_stride_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n , maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n , maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    K_5opt_oneThreadOne5opt_qiao_stride_iter_shared_noOccupy _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n , maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    // only for large size where shared mem is not enough for all TSP and all possibilites
    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_shared_onlyPossibility _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n , maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    cudaChk(cudaPeekAtLastError());
}


//! 0617 QWB: add parallel 5opt
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne5opt_qiao_StrideIterInner5( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                             Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                             double  maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                             double iter, double istride
                                                             ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //  maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);

    cout << "device 0 grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " , INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    //        K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterNSharedOutPossibleLoop _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //qiao try using the best strategy
    //            K_5opt_qiao_stride_iter_SelectBest_iterN  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    K_5opt_qiao_stride_iter_SelectBest_sharePossible _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //selected
    //        K_5opt_qiao_stride_iter_SelectBest_iterN_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //qiao for experimental with no shared mem 71/76 5-opt per run
    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterN  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //qiao for experimental test with shared mem of tsp tour and possibilites, with global occupy, 32/41/59 5-opt per run correct
    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterN_shared_noShareOccupy  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //qiao shared all, including occupies/possibilities/and tsp tour 8opt perrun
    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterNShared  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    cudaChk(cudaPeekAtLastError());
}



//! 0617 QWB: add parallel 5opt
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne5opt_qiao_StrideIterInner5_D1( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                                Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                                double  maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                double iter, double istride
                                                                ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //  maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);

    cout << "device 1 grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " , INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    //qiao try using the best strategy
    //    K_5opt_qiao_stride_iter_SelectBest_iterN  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //    K_5opt_qiao_stride_iter_SelectBest_iterN_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global





    //        K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterNSharedOutPossibleLoop _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //qiao for experimental with no shared mem 71/76 5-opt per run
    K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterN  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //qiao for experimental test with shared mem of tsp tour and possibilites, with global occupy, 32/41/59 5-opt per run correct
    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterN_shared_noShareOccupy  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //qiao shared all, including occupies/possibilities/and tsp tour
    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_firstSelect_iterNShared  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    cudaChk(cudaPeekAtLastError());
}




//! 0624 QWB: add parallel 3opt one thread one 3-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne3opt_RockiSmall( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                  double maxChecks3opt,
                                                  double iter
                                                  ) {


    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE,
                                 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);


    cout << "grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " blocks="  <<  ", INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;


    K_3opt_oneThreadOne3opt_rockiSmall_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt, iter);// global
    K_3opt_oneThreadOne3opt_rockiSmall_iterBest _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt, iter);// global



    //        KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks3opt, 1);

    //        double blocks =  (maxChecks3opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //        cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //             << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    //        if(blocks > 2147483647)
    //        { cout << "Error Grid size bigger than maximum >>>>>>>>>>>>>>>> " << endl;
    //            return;
    //        }

    //        K_3opt_oneThreadOne3opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt, iter);// global


    cudaChk(cudaPeekAtLastError());
}




//! 0624 QWB: add parallel 3opt one thread one 3-opt with Rocki small work correctly donot change any number
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne3opt_qiao_stride( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                   Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                   double maxChecks3opt, double maxChecksoOptDivide,
                                                   double iter, double istride
                                                   ) {


    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE,
                                 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);


    //    cout << "grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " blocks="  <<  ", INT_MAX="
    //         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    //correct 14 20 18  31 opt per run faster than the below one  only 203s 192s
    K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global

    //correct selected for iterkoptimal 14/37/29 3-opt per run and take time 210s
    //        K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest_shared  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global



    //qiao 2024 experimental test of following three kernels
    //correct qiao first no share run faster than using shared 117 3-opt per run qiao test no shared version works faster than the following using shared TSP array 201729
    //    K_3opt_oneThreadOne3opt_rockiSmall_iterStride _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global

    //correct 112opts  6650.7ms qiao test with shared version ,no shared occupy works faster than shared occupy  223779  ms
    //        K_3opt_oneThreadOne3opt_rockiSmall_iterStride_shared_noShareOccupy _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global

    //correct only 1or2 opt of opts than do not use sharedOccupy 6670ms too much time
    //            K_3opt_oneThreadOne3opt_rockiSmall_iterStride_sharedwithOccupy  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global




    //error does not work to build sharedMem while write it on another place
    //K_3opt_oneThreadOne3opt_rockiSmall_iterStride_onlySharedOccupy _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global


    cudaChk(cudaPeekAtLastError());
}


//! 0624 QWB: add parallel 3opt one thread one 3-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne6opt_RockiSmall( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                  double maxChecks6opt, double maxChecks3opt,
                                                  unsigned int iter
                                                  ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);

    double blocks =  (maxChecks6opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    K_6opt_oneThreadOne6opt_rockiSmall_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, iter);// global






    //    KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks6opt, 1);
    //    double blocks =  (maxChecks6opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //    cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;
    //    K_6opt_oneThreadOne6opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, iter);// global

    cudaChk(cudaPeekAtLastError());
}


//! 0624 QWB: add parallel 3opt one thread one 3-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne6opt_qiao_iterStride( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                       Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                       double maxChecks6opt, double maxChecks3opt, double maxChecksoOptDivide,
                                                       double iter, double iStride
                                                       ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);

    //    double blocks =  (maxChecks6opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //    cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    //correct select the best re-connection scheme for one 6opt on global occupy
    //    K_6opt_oneThreadOne6opt_qiao_stride_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, maxChecksoOptDivide, iter, iStride);// global


    //selected
    //        K_6opt_bestScheme_shared_qiao_stride_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, maxChecksoOptDivide, iter, iStride);// global


    K_6opt_bestScheme_shared_possible_stride_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, maxChecksoOptDivide, iter, iStride);// global



    //correct select the first re-connection scheme for one 6opt on global occupy
    //    K_6opt_oneThreadOne6opt_qiao_stride_iter_firstScheme _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, maxChecksoOptDivide, iter, iStride);// global

    //correct select the first re-connection scheme and only shared possibilities
    //    K_6opt_oneThreadOne6opt_qiao_stride_iter_firstScheme_onlySharePossibility _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, maxChecksoOptDivide, iter, iStride);// global


    //    K_6opt_qiao_stride_iter_firstScheme_ShareNoOccupy _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, maxChecksoOptDivide, iter, iStride);// global




    cudaChk(cudaPeekAtLastError());
}



//! 0624 QWB: add parallel 5 opt from 6-opt
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_5opt_qiao_iterStride( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                           Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                           double maxChecks6opt, double maxChecks3opt, double maxChecksoOptDivide,
                                           double iter, double iStride
                                           ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);



    //    20241008
    K_5opt_2_qiao_stride_iter_firstScheme_onlySharePossibility  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, maxChecksoOptDivide, iter, iStride);// global



    cudaChk(cudaPeekAtLastError());
}



//! 0624 QWB: add parallel variable k-opt in one GPU kernel
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_VariableKopt_qiao_iterStride( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                   Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                   double maxChecks2opt, double maxChecks3opt,double maxChecks4opt,
                                                   double maxChecks6opt,  double maxChecksoOptDivide,
                                                   double iter, double iStride
                                                   ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);


    //correct select the first re-connection scheme and only shared possibilities
    K_VariableKopt_qiao_stride_iter_firstScheme_onlySharePossibility _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks3opt, maxChecks4opt, maxChecks6opt, maxChecksoOptDivide, iter, iStride);// global


    //    K_VariableKopt_qiao_stride_iter_bestScheme_onlySharePossibility _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks2opt, maxChecks3opt, maxChecks4opt, maxChecks6opt, maxChecksoOptDivide, iter, iStride);// global



    cudaChk(cudaPeekAtLastError());
}



//! WB.Q add to returenChangeLinks from one node be carefull when using 2 GPUs
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
__device__ void K_returnChangeLinks(NeuralNetLinks<BufferDimension, Point> nn_source,
                                    PointCoord node1, PointCoord& node2, int& changeLink1, int& changeLink2)
{
    int N = nn_source.adaptiveMap.width;

    PointCoord node2_(0, 0);
    nn_source.networkLinks[0][node1[0]].get(0, node2_);
    node2[0] = node2_[0];
    node2[1] = node2_[1];

    // make sure node2 is in right direction of node1
    if(nn_source.grayValueMap[node1[1]][node1[0]] == N-1 && nn_source.grayValueMap[node2[1]][node2[0]] != 0 )
    {
        nn_source.networkLinks[node1[1]][node1[0]].get(1, node2_);
        node2[0] = node2_[0];
        node2[1] = node2_[1];
        changeLink1 = 1;
    }
    else if((nn_source.grayValueMap[node1[1]][node1[0]] != N-1) && nn_source.grayValueMap[node2[1]][node2[0]] - 1 != nn_source.grayValueMap[0][node1[0]] )
    {
        nn_source.networkLinks[node1[1]][node1[0]].get(1, node2_);
        node2[0] = node2_[0];
        node2[1] = node2_[1];
        changeLink1 = 1;
    }
    else
        changeLink1 = 0;

    PointCoord node1_(0, 0);
    nn_source.networkLinks[0][node2[0]].get(0, node1_);
    if((int)node1_[0] != node1[0] || (int)node1_[1] != node1[1])
    {
        changeLink2 = 1;
    }
    else
        changeLink2 = 0;
}


//! WB.Q add to execute non-iteracted 2opt only with node3
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
KERNEL void K_2opt_executeNonItera2optOnlyWithNode3(NeuralNetLinks<BufferDimension, Point> nn_source)
{
    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height)
    {
        if(nn_source.activeMap[0][_x]){
            // execute non interact 2-opt without checking
            PointCoord node1(_x, 0);

            int node3_int = nn_source.densityMap[0][_x];
            PointCoord node3(node3_int, 0);

            // node2
            PointCoord node2(0, 0);
            PointCoord node4(0, 0);

            int changeLink1 = 0;
            int changeLink2 = 0;
            int changeLink3 = 0;
            int changeLink4 = 0;

            K_returnChangeLinks(nn_source, node1, node2, changeLink1, changeLink2);
            K_returnChangeLinks(nn_source, node3, node4, changeLink3, changeLink4);

            nn_source.networkLinks[0][node1[0]].bCell[changeLink1] = node3;
            nn_source.networkLinks[0][node3_int].bCell[changeLink3] = node1;
            nn_source.networkLinks[0][node2[0]].bCell[changeLink2] = node4;
            nn_source.networkLinks[0][node4[0]].bCell[changeLink4] = node2;


        }

    }

    END_KER_SCHED
}



//! QWB: execute non-interacted 2-exchanges only with node3
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_executeNonItera2ExchangeOnlyWithNode3( NeuralNetLinks<BufferDimension, Point>& nn_source) {

    KER_CALL_THREAD_BLOCK_1D(b, t,
                             BLOCKSIZE, 16,
                             nn_source.adaptiveMap.width,
                             nn_source.adaptiveMap.height);
    K_2opt_executeNonItera2optOnlyWithNode3 _KER_CALL_(b, t) (nn_source);

}





}//namespace operators

#endif // EMST_OPERATORS_H
