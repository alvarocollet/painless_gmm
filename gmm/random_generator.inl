//-----------------------------------------------------------------------------
template <typename Vec>
bool AC::RandomGenerator::NonRepeatingSubset(Vec& sampleIDs, int maxTries)
{
    // Invalid case
    int numElements = (int)sampleIDs.size();
    if (sampleIDs.empty() || (m_maxValue - m_minValue + 1) < numElements)
        return false;

    // Trivial case (we only want a non-repeating subset, we don't care about ordering)
    if ((m_maxValue - m_minValue + 1) == numElements) {
        for (int i = 0; i < numElements; ++i)
            sampleIDs[i] = m_minValue + i;
        return true;
    }

    // Regular case (you should set your minimum and maximum value in setLimits)
    sampleIDs[0] = draw();

    for (int i = 1; i < numElements; ++i) {
        int numIterations = 0;
        sampleIDs[i] = draw();
        while (!IsUniqueSampleID<Vec>(sampleIDs, i) && numIterations < c_MaxIterations)
        {
            sampleIDs[i] = (sampleIDs[i] + 1) % numElements; // Keep changing the random samples until the sequence is unique
            numIterations++;
        }
        if (numIterations == maxTries) // Give up if we try too many times
            return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
template <typename Vec>
bool AC::RandomGenerator::IsUniqueSampleID(Vec& sampleIDs, int idx)
{
    _ASSERT(sampleIDs.size() > idx && L"Invalid input");
    bool unique = true;
    // Query if idx is unique inside sampleIDs. For large vector sizes this will be inefficient and we should use a std::map, 
    // but our typical use case is for 3-vectors (in which this is faster).
    for (int j = 0; j < idx; ++j) {
        if (sampleIDs[idx] == sampleIDs[j]) {
            unique = false;
            break;
        }
    }
    return unique;
}