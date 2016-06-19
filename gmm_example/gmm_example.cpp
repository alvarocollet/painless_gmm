// gmm_example.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "gmm/gmm.h" // Remember, you need to put the $(SolutionDir) in "Additional Include Directories"
#include "gmm/kmeans.h"
#include "gmm/math_utils.h"
#include <Eigen/Core> // Remember, you need to put the path to Eigen in "Additional Include Directories"
#include <Eigen/Geometry>

///////////////////////////////////////////////////////////////////////////////
bool TestKMeans3D() // Basic test to show how to use KMeans 
{
    using namespace AC;

    // Create some centroids
    int numCentroids = 5;
    std::vector<Vec3> centroids(numCentroids);
    float scale = 10;
    centroids[0] = Vec3(-scale, 0, 0);
    centroids[1] = Vec3(0, -scale, 0);
    centroids[2] = Vec3(0, 0, -scale);
    centroids[3] = Vec3(scale, 0, 0);
    centroids[4] = Vec3(0, scale, 0);

    // Create some observations
    int numObservationsPerCentroid = 100;
    int numObservations = numObservationsPerCentroid * numCentroids;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-scale / 4, scale / 4);
    std::vector<Vec3> observations(numObservations);
    std::vector<int> assignmentsGT(numObservations);
    auto noise = std::bind(distribution, generator);
    Vec3 observation;
    for (int k = 0; k < numCentroids; ++k) {
        for (int i = 0; i < numObservationsPerCentroid; ++i) {
            observations[numObservationsPerCentroid*k + i] = Vec3(centroids[k][0] + float(noise()), centroids[k][1] + float(noise()), centroids[k][2] + float(noise()));
            assignmentsGT[numObservationsPerCentroid*k + i] = k;
        }
    }

    // Run kmeans
    KMeans<Vec3> kmeans(numCentroids);
    std::vector<int> assignmentsKMeans(numObservations);
    kmeans.setMaxIterations(100);
    kmeans.Process(observations, 30 /* restart 30 times with different initializations */, assignmentsKMeans);

    // Compute average best distance between centroids and KMeans
    double avgBestDistance = 0;
    int assignment;
    for (auto& centroidGT : centroids)
        avgBestDistance += kmeans.ClosestCentroid(centroidGT, assignment);
    avgBestDistance /= (int)centroids.size();

    // Compute assignment differences between the ground truth and KMeans
    int badAssignments = 0;
    for (int i = 0; i < (int)assignmentsGT.size(); ++i) {
        for (int j = i; j < (int)assignmentsGT.size(); ++j) {
            if ((assignmentsGT[i] == assignmentsGT[j] && assignmentsKMeans[i] != assignmentsKMeans[j]) ||
                (assignmentsGT[i] != assignmentsGT[j] && assignmentsKMeans[i] == assignmentsKMeans[j])) {
                badAssignments++;
            }
        }
    }
    // Assignments are pairwise, so there is a total of N*(N-1)/2 possible bad assignments;
    int maxAssignments = (int)assignmentsGT.size() * ((int)assignmentsGT.size() - 1) / 2;
    // If centroids are within 10% distance of GT, and there are less than 10% assignment errors, declare success.
    if (avgBestDistance < scale && badAssignments < (int)maxAssignments / 10)
        return true;
    else
        return false;
}

///////////////////////////////////////////////////////////////////////////////
/// <summary> Basic test for GMM. Create a noisy set of observations from a (known) multivariate gaussian distribution, 
///           fit GMM to it, and compare the differences in labeling. </summary>
/// <returns> true if the test passes, false if the test fails. </returns>
bool TestGMM3D()
{
    using namespace AC;

    // Create some centroids
    int numModes = 5;
    std::vector<Vec3> centroids(numModes);
    double scale = 10;
    centroids[0] = Vec3(-scale, 0, 0);
    centroids[1] = Vec3(0, -scale, 0);
    centroids[2] = Vec3(0, 0, -scale);
    centroids[3] = Vec3(scale, 0, 0);
    centroids[4] = Vec3(0, scale, 0);

    // Create some covariance matrices, and rotate and scale them
    std::vector<Mat3> covariances(numModes);
    covariances[0] = Vec3(1.0, 1.0, 1.0).asDiagonal() * Eigen::AngleAxisd(0.0, Vec3::UnitZ());
    covariances[1] = Vec3(2.0, 1.0, 0.5).asDiagonal() * Eigen::AngleAxisd(0.2, Vec3::UnitZ());
    covariances[2] = Vec3(1.0, 2.0, 1.0).asDiagonal() * Eigen::AngleAxisd(0.4, Vec3::UnitZ());
    covariances[3] = Vec3(1.0, 1.0, 2.0).asDiagonal() * Eigen::AngleAxisd(0.6, Vec3::UnitZ());
    covariances[4] = Vec3(0.75, 1.0, 0.75).asDiagonal() * Eigen::AngleAxisd(0.8, Vec3::UnitZ());

    // Create some observations
    int numObservations = 100;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-scale / 4, scale / 4);
    std::vector<Vec3> observations(numObservations * numModes);
    std::vector<int> assignmentsGT(numObservations * numModes);
    auto noise = std::bind(distribution, generator);
    Vec3 observation;
    for (int k = 0; k < numModes; ++k) {
        for (int i = 0; i < numObservations; ++i) {
            observations[numObservations*k + i] = covariances[k] * Vec3(centroids[k][0] + noise(), centroids[k][1] + noise(), centroids[k][2] + noise());
            assignmentsGT[numObservations*k + i] = k;
        }
    }

    // Run GMM
    GMM::GMM3D gmm(numModes);
    gmm.Process(observations);

    // Compute hard assignments and probabilities
    std::vector<int> assignmentsGMM(numObservations * numModes);
    double avgProbability = 0;
    AC::OnlineMean<double> avgLogProb;
    for (int i = 0; i < (int)observations.size(); ++i) {
        gmm.ClosestMode(observations[i], assignmentsGMM[i] /* output value */);
        avgLogProb.Push(gmm.LogResponsibility(observations[i], assignmentsGMM[i]));
    }
    avgProbability = exp(avgLogProb.Mean());

    // Compute assignment differences between the ground truth and KMeans
    int badAssignments = 0;
    for (int i = 0; i < (int)assignmentsGT.size(); ++i) {
        for (int j = i; j < (int)assignmentsGT.size(); ++j) {
            if ((assignmentsGT[i] == assignmentsGT[j] && assignmentsGMM[i] != assignmentsGMM[j]) ||
                (assignmentsGT[i] != assignmentsGT[j] && assignmentsGMM[i] == assignmentsGMM[j])) {
                badAssignments++;
            }
        }
    }
    // Assignments are pairwise, so there is a total of N*(N-1)/2 possible bad assignments;
    int maxAssignments = (int)assignmentsGT.size() * ((int)assignmentsGT.size() - 1) / 2;
    // If average point probability is larger than 0.9, and there are less than 10% assignment errors, declare success.
    if (avgProbability > 0.9 && badAssignments < (int)maxAssignments / 10)
        return true;
    else
        return false;
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    bool isKMeansOK = TestKMeans3D();
    bool isGMMOK = TestGMM3D();
    return isKMeansOK && isGMMOK;
}
