#pragma once

class BlueNoiseStreamPolynomial
{
public:
	BlueNoiseStreamPolynomial(pcg32_random_t rng)
		: m_rng(rng)
	{
		m_lastValues[0] = RandomFloat01();
		m_lastValues[1] = RandomFloat01();
	}
	
	float Next()
	{
		// Filter uniform white noise to remove low frequencies and make it blue.
		// A side effect is the noise becomes non uniform.
		static const float xCoefficients[3] = {0.5f, -1.0f, 0.5f};

		float value = RandomFloat01();

		float y =
			value * xCoefficients[0] +
			m_lastValues[0] * xCoefficients[1] +
			m_lastValues[1] * xCoefficients[2];

		m_lastValues[1] = m_lastValues[0];
		m_lastValues[0] = value;

		// the noise is also [-1,1] now, normalize to [0,1]
		float x = y * 0.5f + 0.5f;

		// Make the noise uniform again by putting it through a piecewise cubic polynomial approximation of the CDF
		// Switched to Horner's method polynomials, and a polynomial array to avoid branching, per Marc Reynolds. Thanks!
		float polynomialCoefficients[16] = {
			5.25964f, 0.039474f, 0.000708779f, 0.0f,
			-5.20987f, 7.82905f, -1.93105f, 0.159677f,
			-5.22644f, 7.8272f, -1.91677f, 0.15507f,
			5.23882f, -15.761f, 15.8054f, -4.28323f
		};
		int first = std::min(int(x * 4.0f), 3) * 4;
		return polynomialCoefficients[first + 3] + x * (polynomialCoefficients[first + 2] + x * (polynomialCoefficients[first + 1] + x * polynomialCoefficients[first + 0]));
	}

private:
	float RandomFloat01()
	{
		// return a uniform white noise random float between 0 and 1.
		// Can use whatever RNG you want, such as std::mt19937.
		return ldexpf((float)pcg32_random_r(&m_rng), -32);
	}

	pcg32_random_t m_rng;
	float m_lastValues[2] = {};
};

class RedNoiseStreamPolynomial
{
public:
	RedNoiseStreamPolynomial(pcg32_random_t rng)
		: m_rng(rng)
	{
		m_lastValues[0] = RandomFloat01();
		m_lastValues[1] = RandomFloat01();
	}

	float Next()
	{
		// Filter uniform white noise to remove high frequencies and make it red.
		// A side effect is the noise becomes non uniform.
		static const float xCoefficients[3] = { 0.25f, 0.5f, 0.25f };

		float value = RandomFloat01();

		float y =
			value * xCoefficients[0] +
			m_lastValues[0] * xCoefficients[1] +
			m_lastValues[1] * xCoefficients[2];

		m_lastValues[1] = m_lastValues[0];
		m_lastValues[0] = value;

		float x = y;

		// Make the noise uniform again by putting it through a piecewise cubic polynomial approximation of the CDF
		// Switched to Horner's method polynomials, and a polynomial array to avoid branching, per Marc Reynolds. Thanks!
		float polynomialCoefficients[16] = {
			5.25964f, 0.039474f, 0.000708779f, 0.0f,
			-5.20987f, 7.82905f, -1.93105f, 0.159677f,
			-5.22644f, 7.8272f, -1.91677f, 0.15507f,
			5.23882f, -15.761f, 15.8054f, -4.28323f
		};
		int first = std::min(int(x * 4.0f), 3) * 4;
		return polynomialCoefficients[first + 3] + x * (polynomialCoefficients[first + 2] + x * (polynomialCoefficients[first + 1] + x * polynomialCoefficients[first + 0]));
	}

private:
	float RandomFloat01()
	{
		// return a uniform white noise random float between 0 and 1.
		// Can use whatever RNG you want, such as std::mt19937.
		return ldexpf((float)pcg32_random_r(&m_rng), -32);
	}

	pcg32_random_t m_rng;
	float m_lastValues[2] = {};
};

// From Nick Appleton:
// https://mastodon.gamedev.place/@nickappleton/110009300197779505
// But I'm using this for the single bit random value needed per number:
// https://blog.demofox.org/2013/07/07/a-super-tiny-random-number-generator/
// Which comes from:
// http://www.woodmann.com/forum/showthread.php?3100-super-tiny-PRNG
class BlueNoiseStreamAppleton
{
public:
	BlueNoiseStreamAppleton(unsigned int seed)
		: m_seed(seed)
		, m_p(0.0f)
	{
	}

	float Next()
	{
		float ret = (GenerateRandomBit() ? 1.0f : -1.0f) / 2.0f - m_p;
		m_p = ret / 2.0f;

		// convert from [-1,1] to [0,1]
		return ret * 0.5f + 0.5f;
	}

private:
	bool GenerateRandomBit()
	{
		m_seed += (m_seed * m_seed) | 5;
		return (m_seed & 0x80000000) != 0;
	}

	unsigned int m_seed;
	float m_p;
};
