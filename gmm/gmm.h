#ifndef __GMM_H__
#define __GMM_H__

#include <vector>
#include <memory>
#include "gaussian.h"

namespace AC
{
    namespace GMM {

        // Constants
        const int c_MaxIterations = 10;
        const int c_KMeansRestarts = 10; // Default number of restarts for KMeans initialization
        const double c_EMTolerance = 1e-4; // EM Tolerance before declaring success in optimization
        const int INVALID_MODE = -1; // If you call closestMode and there are no modes, it returns INVALID_MODE

        class GMM3D {
        public:
            typedef std::shared_ptr<GMM3D> SP;
            typedef GaussianDistribution3D::SP mode_type;

            /// <summary> Constructor. </summary>
            /// <remarks> Alcollet, 7/17/2013. </remarks>
            /// <param name="numModes"> Number of modes (Gaussians) in GMM. </param>
            GMM3D(int numModes);
            GMM3D(const GMM3D& rhs);

            /// <summary> Compute a GMM from a given set of observations. The GMM is initialized using k-means, and then we use EM to train the full-covariance GMMs. </summary>
            /// <param name="observations"> The observations. </param>
            /// <param name="numKMeansRestarts"> [optional] Number of restarts in KMeans initialization. </param>
            /// <param name="scalingFactors"> [optional] Scaling factor for each observation. Useful if the observations have been whitened (so that the output GMM will still be unwhitened). </param>
            /// <returns> true if it succeeds, false if it fails. </returns>
            bool Process(const std::vector<Vec3>& observations, int numKMeansRestarts = c_KMeansRestarts, Vec3* scalingFactors = nullptr);

            /// <summary> Compute the log likelihood of the mixture model for an observation x_n, such that: log P(x_n) = log ( sum_k ( p(x_n|k)*p(k) )) </summary>
            /// <param name="observation"> The observation. </param>
            /// <returns> The log likelihood of the mixture model for this observation </returns>
            double LogLikelihood(const Vec3& observation);

            /// <summary> Compute the likelihood of the mixture model for an observation x_n, such that: P(x_n) = sum_k ( p(x_n|k)*p(k) ) </summary>
            /// <param name="observation"> The observation. </param>
            /// <returns> Likelihood of the mixture model for this observation </returns>
            double Likelihood(const Vec3& observation);

            /// <summary> Compute the log likelihood of the mixture model for a set of observations, as: log P(X) = sum_{x_n in X} log P(x_n) </summary>
            /// <param name="observations"> The observations. </param>
            /// <returns> . </returns>
            double LogLikelihood(const std::vector<Vec3>& observations);

            /// <summary> Compute the log responsibility log(p(k|x_n)) = log(p(x_n|k)) + log(P(k)) - log(p(x_n)). </summary>
            /// <param name="observation"> The observation x_n </param>
            /// <param name="idxMode"> The mode index k. </param>
            /// <returns> The log responsibility log(p(k|x_n)) </returns>
            double LogResponsibility(const Vec3& observation, int idxMode);

            /// <summary> Compute closest mode for a given observation. </summary>
            /// <param name="observation"> The observation. </param>
            /// <param name="assignment">  [out] Index of the closest mode K to this observation. </param>
            /// <returns> The log probability that this observation was created from mode K in the GMM. </returns>
            double ClosestMode(const Vec3& observation, int& mode) const;

            /// <summary> Remove modes with weight less than some (small) tolerance. </summary> 
            /// <param name="tolerance"> Max weight to consider a mode as 'valid'. </param>
            /// <returns> The number of valid modes after removal </returns> 
            size_t RemoveBadModes(double tolerance);

            std::vector<GaussianDistribution3D::SP>& Modes() { return m_modes; }
            const std::vector<GaussianDistribution3D::SP>& Modes() const { return m_modes; }

            GaussianDistribution3D::SP& Modes(int k) { _ASSERT(k < m_modes.size() && k >= 0 && L"Invalid index"); return m_modes[k]; }
            const GaussianDistribution3D::SP& Modes(int k) const { _ASSERT(k < m_modes.size() && k >= 0 && L"Invalid index"); return m_modes[k]; }

            double GlobalWeight() { return m_globalWeight; }
            void SetGlobalWeight(double w) { m_globalWeight = w; }

        private:
            std::vector<GaussianDistribution3D::SP> m_modes; // k-Vector containing the multiple Gaussians
            std::vector<double> m_tmpLogLikelihoods; // k-Vector (temporary) to store the log likelihoods for each Gaussian
            std::vector<double> m_tmpLogWeights; // k-Vector (temporary) to store the LogWeights for each Gaussian
            double m_globalWeight; // Total weight for this GMM distribution (by default, = 1)
        };

        /// <summary> Compute log( sum_n( x1_n * x2_n )) from vectors of logValues1 and logValues2, where logValuesK[n] = log(xK_n).
        ///           We use the log-exp-sum trick to avoid underflow: log( sum_n( x1_n * x2_n )) = log( sum_n( exp( log x1_n + log x2_n ))) = 
        ///           = log(x1_MAX) + log(x2_MAX) + log( sum_n( exp( log(x1_n) - log(x1_MAX) + log(x2_n) - log(x2_MAX))) </summary>
        /// <param name="logValues1"> The first vector of log values. </param>
        /// <param name="logValues2"> The second vector of log values. </param>
        /// <returns> log( sum_n( x1_n * x2_n )) </returns>
        double LogExpSum(const std::vector<double>& logValues1, const std::vector<double>& logValues2);

        /// <summary> Compute the probability of 'observation' to be a sample of gmm1 or gmm2. </summary>
        /// <param name="gmm1">        [in] The first gmm. </param>
        /// <param name="gmm2">        [in] The second gmm. </param>
        /// <param name="observation"> [in] The observation. </param>
        /// <returns> The probability of 'observation' to be a sample of gmm1, i.e., p(obs|gmm1) / (p(obs|gmm1)+p(obs|gmm2))</returns>
        double GMMLikelihoodRatio(GMM3D& gmm1, GMM3D& gmm2, const Vec3& observation);
    }
}

#endif