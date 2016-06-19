#include "gmm.h"
#include "em.h"

//----------------------------------------------------------------------------
AC::GMM::EM::EM(int numObservations, int numModes, double tolerance) : m_maxIterations(c_MaxIterations), m_numTrainingPoints(numObservations), m_tolerance(tolerance)
{
    // Reserve memory for our temporary vectors (to avoid allocations while processing)
    m_tmpResponsibilities.reserve(numModes);
    for (int k = 0; k < numModes; ++k)
        m_tmpResponsibilities.emplace_back(std::vector<double>(numObservations));
}

//----------------------------------------------------------------------------
void AC::GMM::EM::UpdateResponsibilities(const std::vector<Vec3>& observations, GMM3D& gmm)
{
    _ASSERT(observations.size() == m_numTrainingPoints && m_numTrainingPoints >= gmm.Modes().size() && "Invalid number of observations.");
    bool nonFinite = false;
    int numObservations = (int)observations.size();
    int numModes = (int)gmm.Modes().size();
    for (int o = 0; o < numObservations; ++o) {
        for (int idxMode = 0; idxMode < numModes; ++idxMode) {
            double responsibility = exp(gmm.LogResponsibility(observations[o], idxMode));
            if (!IsFinite(responsibility))
                nonFinite = true;
            m_tmpResponsibilities[idxMode][o] = IsFinite(responsibility) ? responsibility : 0;
        }
    }
}

//----------------------------------------------------------------------------
void AC::GMM::EM::UpdateWeights(GMM3D& gmm)
{
    _ASSERT(m_numTrainingPoints >= gmm.Modes().size() && "Invalid number of observations.");
    int numModes = (int)gmm.Modes().size();
    for (int k = 0; k < numModes; ++k) {
        AC::OnlineMean<double> weight;
        for (int o = 0; o < m_numTrainingPoints; ++o)
            weight.Push(m_tmpResponsibilities[k][o]);
        gmm.Modes(k)->setWeight(std::max(weight.Mean(), c_SafeMinWeight)); // sum_n(p(k|x_n))/N
    }
}

//----------------------------------------------------------------------------
void AC::GMM::EM::UpdateMeans(const std::vector<Vec3>& observations, GMM3D& gmm)
{
    _ASSERT(observations.size() == m_numTrainingPoints && m_numTrainingPoints >= gmm.Modes().size() && "Invalid number of observations.");
    int numModes = (int)gmm.Modes().size();
    AC::OnlineMean<Vec3> sumMean;
    for (int k = 0; k < numModes; ++k) {
        for (int o = 0; o < m_numTrainingPoints; ++o)
            sumMean.Push(observations[o] * m_tmpResponsibilities[k][o]); // sum_n(p(k|x_n)*x_n)/N
                                                                         // In UpdateWeights, we calculate sum_n( p(k|x_n) )/N, so the /N in sumMean cancels this one, and we get:
        gmm.Modes(k)->setMean(sumMean.Mean() / std::max(gmm.Modes(k)->Weight(), c_SafeMinWeight)); // sum_n(p(k|x_n)*x_n)/sum_n(p(k|x_n)) 
        sumMean.Reset(); // Reset for the next use
    }
}

//----------------------------------------------------------------------------
void AC::GMM::EM::UpdateCovariances(const std::vector<Vec3>& observations, GMM3D& gmm)
{
    _ASSERT(observations.size() == m_numTrainingPoints && m_numTrainingPoints >= gmm.Modes().size() && "Invalid number of observations.");
    int numModes = (int)gmm.Modes().size();
    Vec3 centeredObservation;
    Mat3 cov;
    Mat3 centeredObservationOuterProd;
    for (int k = 0; k < numModes; ++k) {
        cov = Mat3::Zero();
        for (int o = 0; o < m_numTrainingPoints; ++o) {
            centeredObservation = observations[o] - gmm.Modes(k)->Mean();
            centeredObservationOuterProd = centeredObservation * centeredObservation.transpose();
            cov += m_tmpResponsibilities[k][o] * centeredObservationOuterProd;
        }
        // In UpdateWeights, we calculate sum_n( p(k|x_n) )/N, and we need sum_n( p(k|x_n) ) in the denominator below
        cov /= m_numTrainingPoints * std::max(gmm.Modes(k)->Weight(), c_SafeMinWeight); // so we divide by sum_n(p(k|x_n))/N * N

        gmm.Modes(k)->setCovariance(cov);
    }
}

//----------------------------------------------------------------------------
bool AC::GMM::EM::Process(const std::vector<Vec3>& observations, GMM3D& gmm)
{
    _ASSERT(m_numTrainingPoints > 0 && m_numTrainingPoints > gmm.Modes().size() && "Invalid number of observations.");
    int numIterations = 0;
    double OldLikelihood;
    double NewLikelihood = -std::numeric_limits<double>::max();
    do {
        // E-Step
        UpdateResponsibilities(observations, gmm);

        // M-Step
        UpdateWeights(gmm);
        UpdateMeans(observations, gmm);
        UpdateCovariances(observations, gmm);

        // Update GMM likelihood (stopping condition)
        OldLikelihood = NewLikelihood;
        NewLikelihood = gmm.LogLikelihood(observations);
    } while (abs((NewLikelihood - OldLikelihood) / OldLikelihood) > m_tolerance && numIterations++ < m_maxIterations);

    // Return true if converged
    return numIterations < m_maxIterations;
}
