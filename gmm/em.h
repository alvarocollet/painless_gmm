#ifndef __EM_H__
#define __EM_H__

#include "math_utils.h"
#include "gaussian.h"
#include "gmm.h"

namespace AC
{
    namespace GMM
    {

        // Expectation-Maximization algorithm for GMM
        class EM {

        public:
            EM(int numObservations, int numModes, double tolerance = c_EMDefaultTolerance, int maxIterations = c_EMDefaultMaxIterations);

            /// <summary> Train gaussian mixture model with the EM algorithm. Given a set of observations and an *Initialized* GMM, this function optimizes the location
            ///           of the gaussians via iterative expectations and maximizations. </summary>
            /// <param name="observations"> The set of observations. </param>
            /// <param name="gmm">          [in,out] The computed GMM. This GMM should be already initialized by some other method (e.g., k-means), or at random. </param>
            /// <returns> true if EM converged before reaching the max number of iterations </returns>
            bool Process(const std::vector<Vec3>& observations, GMM3D& gmm);

            /// <summary> Sets maximum number of iterations of EM. </summary>
            /// <param name="maxIters"> The maximum number of iterations. </param>
            void setMaxIterations(int maxIters) { m_maxIterations = maxIters; }

            /// <summary> Sets EM tolerance. The stopping condition in EM is the ratio of improvement of the log likelihood,
            ///           i.e.: tolerance > ((newLogLikelihood - oldLogLikelihood) / oldLogLikelihood) finishes the process. </summary>
            /// <param name="tolerance"> The tolerance. </param>
            void setTolerance(double tolerance) { m_tolerance = tolerance; }

        private:
            /// <summary> Updates the gaussian responsibilities, so that: responsibilities[k][n] = log p_kn = log(p(x_n|k)) + log(P(k)) - log(p(x_n)).
            ///           (the responsibilities vector is stored internally). This corresponds to the E-step in EM. </summary>
            /// <param name="observations"> The input set of observations. </param>
            /// <param name="gmm">          The current GMM. </param>
            void UpdateResponsibilities(const std::vector<Vec3>& observations, GMM3D& gmm);

            /// <summary> Updates the GMM weights P(k) according to the (internally stored) responsibilities vector. This corresponds to the 1st part of the M-Step. </summary>
            /// <param name="gmm"> [out] The gmm with updated weights. </param>
            void UpdateWeights(GMM3D& gmm);

            /// <summary> Updates the GMM means according to the observations and responsibilities. This corresponds to the 2nd part of the M-step. </summary>
            /// <param name="observations"> The input set of observations. </param>
            /// <param name="gmm"> [out] The gmm with updated means. </param>
            void UpdateMeans(const std::vector<Vec3>& observations, GMM3D& gmm);

            /// <summary> Updates the GMM covariance matrices according to the observations, responsibilities and means. This corresponds to the 3rd part of the M-step. </summary>
            /// <param name="observations"> The input set of observations. </param>
            /// <param name="gmm"> [out] The gmm with updated covariance matrices. </param>
            void UpdateCovariances(const std::vector<Vec3>& observations, GMM3D& gmm);

            std::vector<std::vector<double>> m_tmpResponsibilities;
            int m_numTrainingPoints; // Number of observations used in training
            double m_tolerance;      // Stopping condition in EM. If ratio of new vs old log likelihoods is lower than tolerance, finish process. 
            int m_maxIterations;     // Max number of iterations of EM. If we do not get under the tolerance in MaxIterations, we give up.
        };
    }
}

#endif
