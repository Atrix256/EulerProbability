#include <stdio.h>
#include <random>
#include <vector>
#include "pcg/pcg_basic.h"

// ============== TEST SETTINGS ==============

#define DETERMINISTIC() false

static const size_t c_lotteryWinFrequency = 10000;
static const size_t c_lotteryTestCount = 100;

static const size_t c_sumTestCount = 10000;

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

// ================== TESTS ==================

template <typename LAMBDA>
float LotteryTest(const LAMBDA& RNG, uint64_t sequenceIndex)
{
	// we need a seed per test to generate the winning number, and another seed per test to generate the random numbers
	uint64_t sequenceIndexBase = sequenceIndex * c_lotteryTestCount * 2;

	// gather up the wins and losses
	std::vector<float> wins(c_lotteryTestCount, 0.0f);
	#pragma omp parallel for
	for (int testIndex = 0; testIndex < c_lotteryTestCount; ++testIndex)
	{
		// Generate a winning number
		size_t winningNumber = MapFloat<size_t>(Generate_WhiteNoise(1, sequenceIndexBase + testIndex * 2)[0], 0, c_lotteryWinFrequency - 1);

		// Report whether the player won
		std::vector<float> rng = RNG(c_lotteryWinFrequency, sequenceIndexBase + testIndex * 2 + 1);
		for (const float& vf : rng)
		{
			size_t v = MapFloat<size_t>(vf, 0, c_lotteryWinFrequency - 1);
			if (v == winningNumber)
			{
				wins[testIndex] = 1.0f;
				break;
			}
		}
	}

	// calculate and return the lose percentage
	float losePercent = 0.0f;
	for (size_t testIndex = 0; testIndex < c_lotteryTestCount; ++testIndex)
		losePercent = Lerp(losePercent, (1.0f - wins[testIndex]), 1.0f / float(testIndex + 1));
	return losePercent;
}

template <typename LAMBDA>
float SumTest(const LAMBDA& RNG, uint64_t sequenceIndex)
{
	// we need a seed per test
	uint64_t sequenceIndexBase = sequenceIndex * c_sumTestCount;

	std::vector<float> sumCount(c_sumTestCount, 0.0f);
	//#pragma omp parallel for
	for (int testIndex = 0; testIndex < c_sumTestCount; ++testIndex)
	{
		std::vector<float> rng = RNG(10, sequenceIndexBase + testIndex);
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
	return count;
}

int main(int argc, char** argv)
{
#if !DETERMINISTIC()
	std::random_device rd;
	g_randomSeed = rd();
#endif

	printf("e = %f\n", std::exp(1.0f));
	printf("1/e = %f\n\n", 1.0f / std::exp(1.0f));

	printf("Lottery Lose Chance:\n");
	printf("  White Noise: %0.2f%%\n", 100.0f * LotteryTest(Generate_WhiteNoise, 0));
	printf("  Golden Ratio: %0.2f%%\n", 100.0f * LotteryTest(Generate_GoldenRatio, 1));
	printf("  Stratified: %0.2f%%\n", 100.0f * LotteryTest(Generate_Stratified, 2));
	printf("  Regular Offset: %0.2f%%\n", 100.0f * LotteryTest(Generate_RegularOffset, 3));

	printf("\nSumming Random Values:\n");
	printf("  White Noise: %f\n", SumTest(Generate_WhiteNoise, 0));
	printf("  Golden Ratio: %f\n", SumTest(Generate_GoldenRatio, 1));
	printf("  Stratified: %f\n", SumTest(Generate_Stratified, 2));
	printf("  Regular Offset: %f\n", SumTest(Generate_RegularOffset, 3));

	return 0;
}

/*

TODO:
- what other types of noise?
 - blue noise would be nice but hard to generate quickly. can you turn triangle distribution into uniform without rejection sampling?
 - if so, could do red noise as well
- csvs with graphs by python
- profile
- could try and make blue noise using an e based MBC algorithm.
 - if it works out, could send it to jcgt or something maybe, as a very short paper.

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
