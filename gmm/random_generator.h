#ifndef __RANDOM_GENERATOR_H__
#define __RANDOM_GENERATOR_H__

#include <random>

namespace AC {

    class RandomGenerator {
    public:
        RandomGenerator() : m_minValue(0), m_maxValue(1), distribution(m_minValue, m_maxValue) {}
        RandomGenerator(int minValue, int maxValue) : m_minValue(minValue), m_maxValue(maxValue) { distribution.param(std::uniform_int<int>::param_type(minValue, maxValue)); }

        void setLimits(int minValue, int maxValue) { m_minValue = minValue; m_maxValue = maxValue; distribution.param(std::uniform_int<int>::param_type(minValue, maxValue)); }

        /// <summary> Draw an integer random number in range [minValue, maxValue]. </summary>
        /// <returns> The random number in range [minValue, maxValue]. </returns>
        int draw() { return distribution(generator); }

        /// <summary> Return a subset of elements in range [m_minValue, m_maxValue] which are not repeated. </summary>
        /// <typeparam name="typename Vec"> Type of the typename vector (e.g., std::vector or std::array). </typeparam>
        /// <param name="values"> [in,out] The vector of non-repeated values in range [m_minValue, m_maxValue] </param>
        /// <returns> true if it succeeds, false if it fails. </returns>
        template <typename Vec>
        bool NonRepeatingSubset(Vec& values, int maxTries = c_MaxIterations);

        /// <summary> Query if idx is already in sampleIDs or not. </summary>
        /// <typeparam name="typename Vec"> Type of the typename vector (e.g., std::vector or std::array). </typeparam>
        /// <param name="sampleIDs"> [in,out] The vector of sampled IDs. </param>
        /// <param name="idx">       The index. </param>
        /// <returns> true if unique sample identifier &lt;typename vec&gt;, false if not. </returns>
        template <typename Vec>
        bool IsUniqueSampleID(Vec& sampleIDs, int idx);

    private:
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution;
        int m_minValue;
        int m_maxValue;
        static const int c_MaxIterations = 25; // Maximum number of tries to get a non-repeating subset
    };

#include "random_generator.inl"

}

#endif