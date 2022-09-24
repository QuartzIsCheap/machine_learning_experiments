#ifndef PERCEPTRON_TESTS_HPP
#define PERCEPTRON_TESTS_HPP

#include <random>

#include "perceptron.hpp"

namespace color_brightness {
	auto finite_set_test(
			std::mt19937_64 &prng_generator
	) noexcept -> double;
	
	auto dynamic_training_test(
			std::mt19937_64 &prng_generator
	) noexcept -> double;
}

#include "tests/color_brightness_tests.inl"

#endif //PERCEPTRON_TESTS_HPP
