#include <stdio.h>
#include <random>
#include <vector>
#include "pcg/pcg_basic.h"
#include <omp.h>
#include <atomic>

// ============== TEST SETTINGS ==============

#define DETERMINISTIC() false

static const size_t c_lotteryWinFrequency = 10000;

static const size_t c_lotteryTestCountOuter = 1000;
static const size_t c_lotteryTestCountInner = 1000;

static const size_t c_sumTestCount = 10000;

static const size_t c_candidateTestCount = 100000;
static const size_t c_candidateCount = 1000;

// ================== OTHER ==================

static const float c_goldenRatioConjugate = 0.61803398875f;
static uint64_t g_randomSeed = 0;

float Lerp(float A, float B, float t)
{
	return A * (1.0f - t) + B * t;
}

// =================== RNG ===================

float PCGRandomFloat01(pcg32_random_t& rng)
{
	return ldexpf((float)pcg32_random_r(&rng), -32);
}

template <typename T>
T MapFloat(float f, T min, T max)
{
	T range = max - min;
	return min + std::min(size_t(f * float(range + 1)), range);
}

std::vector<float> Generate_WhiteNoise(size_t numSamples, uint64_t sequenceIndex)
{
	pcg32_random_t rng;
	pcg32_srandom_r(&rng, g_randomSeed, sequenceIndex);
	std::vector<float> ret(numSamples);
	for (float& v : ret)
		v = PCGRandomFloat01(rng);
	return ret;
}

std::vector<float> Generate_Stratified(size_t numSamples, uint64_t sequenceIndex)
{
	pcg32_random_t rng;
	pcg32_srandom_r(&rng, g_randomSeed, sequenceIndex);
	std::vector<float> ret(numSamples);
	for (size_t index = 0; index < numSamples; ++index)
		ret[index] = (float(index) + PCGRandomFloat01(rng)) / float(numSamples);
	return ret;
}

std::vector<float> Generate_RegularOffset(size_t numSamples, uint64_t sequenceIndex)
{
	float offset = Generate_WhiteNoise(1, sequenceIndex)[0];
	std::vector<float> ret(numSamples);
	for (size_t index = 0; index < numSamples; ++index)
		ret[index] = (float(index) + offset) / float(numSamples);
	return ret;
}

std::vector<float> Generate_GoldenRatio(size_t numSamples, uint64_t sequenceIndex)
{
	std::vector<float> ret(numSamples);
	ret[0] = Generate_WhiteNoise(1, sequenceIndex)[0];
	for (size_t i = 1; i < numSamples; ++i)
		ret[i] = std::fmod(ret[i - 1] + c_goldenRatioConjugate, 1.0f);
	return ret;
}

std::vector<float> ShuffleSequence(std::vector<float>& sequence, uint64_t shuffleSeed)
{
	std::mt19937 rng((unsigned int)shuffleSeed ^ (unsigned int)g_randomSeed);
	std::shuffle(sequence.begin(), sequence.end(), rng);
	return sequence;
}

std::vector<float> Generate_StratifiedShuffled(size_t numSamples, uint64_t sequenceIndex)
{
	std::vector<float> sequence = Generate_Stratified(numSamples, sequenceIndex);
	return ShuffleSequence(sequence, sequenceIndex);
}

std::vector<float> Generate_RegularOffsetShuffled(size_t numSamples, uint64_t sequenceIndex)
{
	std::vector<float> sequence = Generate_RegularOffset(numSamples, sequenceIndex);
	return ShuffleSequence(sequence, sequenceIndex);
}

// ================== TESTS ==================

template <typename LAMBDA>
void LotteryTest(const LAMBDA& RNG, uint64_t sequenceIndex, const char* label)
{
	// we need a seed per test to generate the winning number, and another seed per test to generate the random numbers
	uint64_t sequenceIndexBase = sequenceIndex * c_lotteryTestCountOuter * c_lotteryTestCountInner * 2;

	// gather up the wins and losses
	std::vector<float> wins(c_lotteryTestCountOuter, 0.0f);
	std::atomic<int> testsFinished(0);
	int lastPercent = -1;
	#pragma omp parallel for
	for (int testIndexOuter = 0; testIndexOuter < c_lotteryTestCountOuter; ++testIndexOuter)
	{
		for (int testIndexInner = 0; testIndexInner < c_lotteryTestCountInner; ++testIndexInner)
		{
			if (omp_get_thread_num() == 0)
			{
				int percent = int(100.0f * float(testsFinished.load()) / float(c_lotteryTestCountOuter * c_lotteryTestCountInner));
				if (percent != lastPercent)
				{
					lastPercent = percent;
					printf("\r  %s: %i%%", label, percent);
				}
			}

			int testIndex = testIndexOuter * c_lotteryTestCountInner + testIndexInner;

			// Generate a winning number
			size_t winningNumber = MapFloat<size_t>(Generate_WhiteNoise(1, sequenceIndexBase + testIndex * 2)[0], 0, c_lotteryWinFrequency - 1);

			// Report whether the player won
			std::vector<float> rng = RNG(c_lotteryWinFrequency, sequenceIndexBase + testIndex * 2 + 1);
			float win = 0.0f;
			for (const float& vf : rng)
			{
				size_t v = MapFloat<size_t>(vf, 0, c_lotteryWinFrequency - 1);
				if (v == winningNumber)
				{
					win = 1.0f;
					break;
				}
			}

			wins[testIndexOuter] = Lerp(wins[testIndexOuter], win, 1.0f / float(testIndexInner + 1));
			testsFinished.fetch_add(1);
		}
	}

	// calculate and return the lose percentage
	float losePercent = 0.0f;
	for (size_t testIndexOuter = 0; testIndexOuter < c_lotteryTestCountOuter; ++testIndexOuter)
		losePercent = Lerp(losePercent, (1.0f - wins[testIndexOuter]), 1.0f / float(testIndexOuter + 1));

	printf("\r  %s: %0.2f%% lose chance\n", label, 100.0f * losePercent);
}

template <typename LAMBDA>
void SumTest(const LAMBDA& RNG, uint64_t sequenceIndex, const char* label)
{
	// we need a seed per test
	uint64_t sequenceIndexBase = sequenceIndex * c_sumTestCount;

	std::vector<float> sumCount(c_sumTestCount, 0.0f);
	#pragma omp parallel for
	for (int testIndex = 0; testIndex < c_sumTestCount; ++testIndex)
	{
		std::vector<float> rng = RNG(25, sequenceIndexBase + testIndex);
		float value = 0.0f;
		for (size_t index = 0; index < rng.size(); ++index)
		{
			value += rng[index];
			if (value >= 1.0f)
			{
				sumCount[testIndex] = float(index + 1);
				break;
			}
		}
		if (value < 1.0f)
			printf("[ERROR] Ran out of random numbers.\n");
	}

	// calculate and return the average count
	float count = 0.0f;
	for (size_t testIndex = 0; testIndex < c_sumTestCount; ++testIndex)
		count = Lerp(count, sumCount[testIndex], 1.0f / float(testIndex + 1));

	printf("  %s: %f numbers to get >= 1.0\n", label, count);
}

template <typename LAMBDA>
void CandidatesTest(const LAMBDA& RNG, uint64_t sequenceIndex, const char* label)
{
	// we need a seed per test
	uint64_t sequenceIndexBase = sequenceIndex * c_candidateTestCount;

	struct TestResults
	{
		float candidatesEvaluated = 0.0f;
		float candidateRankPercent = 0.0f;
	};

	std::vector<TestResults> results(c_candidateTestCount);
	#pragma omp parallel for
	for (int testIndex = 0; testIndex < c_candidateTestCount; ++testIndex)
	{
		std::vector<float> candidates = RNG(c_candidateCount, sequenceIndexBase + testIndex);

		// Find the best candidate in the pre candidate group.
		// THe pre candidate group is candidateCount / e in size
		size_t preCandidates = size_t(float(c_candidateCount) / std::exp(1.0f));
		float bestPreCandidate = 0.0f;
		for (size_t i = 0; i < preCandidates; ++i)
			bestPreCandidate = std::max(bestPreCandidate, candidates[i]);

		// Find the first candidate in the second group that is > that candidate, and take that as the winner
		size_t foundAt = 0;
		float bestCandidate = 0.0f;
		for (size_t i = preCandidates; i < c_candidateCount; ++i)
		{
			if (candidates[i] > bestPreCandidate)
			{
				bestCandidate = candidates[i];
				foundAt = i;
				break;
			}
		}
		if (foundAt == 0)
		{
			foundAt = c_candidateCount - 1;
			bestCandidate = bestPreCandidate;
			//printf("[ERROR] Ran out of random numbers.\n");
		}
		results[testIndex].candidatesEvaluated = float(foundAt);

		// find out how many candidates are better than what we found.
		size_t betterCount = 0;
		for (size_t i = 0; i < c_candidateCount; ++i)
		{
			if (candidates[i] > bestCandidate)
				betterCount++;
		}

		results[testIndex].candidateRankPercent = float(betterCount) / float(c_candidateCount);
	}

	TestResults result;
	for (size_t i = 0; i < c_candidateTestCount; ++i)
	{
		result.candidatesEvaluated = Lerp(result.candidatesEvaluated, results[i].candidatesEvaluated, 1.0f / float(i + 1));
		result.candidateRankPercent = Lerp(result.candidateRankPercent, results[i].candidateRankPercent, 1.0f / float(i + 1));
	}

	printf("  %s: %i / %i candidates looked at, %f%% candidates were better\n", label, int(result.candidatesEvaluated + 0.5f), (int)c_candidateCount, 100.0f * result.candidateRankPercent);
}

int main(int argc, char** argv)
{
#if !DETERMINISTIC()
	std::random_device rd;
	g_randomSeed = rd();
#endif

	printf("e = %f\n", std::exp(1.0f));
	printf("1/e = %f\n\n", 1.0f / std::exp(1.0f));

	// NOTE: more evenly spaced sampling means fewer duplicates, which is why they win more.
	printf("Lottery Lose Chance:\n");
	LotteryTest(Generate_WhiteNoise, 0, "White Noise");
	LotteryTest(Generate_GoldenRatio, 1, "Golden Ratio");
	LotteryTest(Generate_Stratified, 2, "Stratified");
	LotteryTest(Generate_RegularOffset, 3, "Regular Offset");

	// NOTE: shuffling stratified and regular offset cause they are only appropriate when we know the number of samples in advance. we don't for this test.
	printf("\nSumming Random Values:\n");
	SumTest(Generate_WhiteNoise, 0, "White Noise");
	SumTest(Generate_GoldenRatio, 1, "Golden Ratio");
	SumTest(Generate_StratifiedShuffled, 2, "Stratified Shuffled");
	SumTest(Generate_RegularOffsetShuffled, 3, "Regular Offset Shuffled");

	// NOTE: shuffling stratified and regular offset because they are monotonic otherwise, and the best candidate is always the last one.
	printf("\nCandidates:\n");
	CandidatesTest(Generate_WhiteNoise, 0, "White Noise");
	CandidatesTest(Generate_GoldenRatio, 1, "Golden Ratio");
	CandidatesTest(Generate_StratifiedShuffled, 2, "Stratified Shuffled");
	CandidatesTest(Generate_RegularOffsetShuffled, 3, "Regular Offset Shuffled");

	// TODO: inner and outer loops for the tests, to help floating point and memory
	// TODO: make each function report progress (%) before results
	// TODO: make the sumtest and lottery test be in charge for printing out progres and then results. one line. label first then progress, which then becomes results.

	return 0;
}

/*

TODO:
- what other types of noise?
 - blue noise would be nice but hard to generate quickly. can you turn triangle distribution into uniform without rejection sampling?
 - if so, could do red noise as well
- csvs with graphs by python
- could try and make blue noise using an e based MBC algorithm.
 - if it works out, could send it to jcgt or something maybe, as a very short paper.
- move to doubles instead of floats?

Note:
* omit and explain the noises that aren't meaningful to specific tests

Meaningful tests remaining:
- interview

e probability tests:
- Lottery: 1/N chance of winning done N times has 1/e chance of never happening. 36.8%.  (1 - 1/N)^N = 1/e
- Umbrella: N umbrellas, everyone puts there umbrella in and grabs one at random. 1/e chance nobody has their umbrella.
- Shuffle: N cards, shuffle. 1/e chance no card is in it's original spot.
- Interview: 100 people, measure typing speed.  streaming candidates. find the best in 100/e of them (36). Then pick the next of the 100 that is better than that
 - 1/e chance of picking best person.

Notes:
- (1 + 1/N)^N = e for large values of N

*/
