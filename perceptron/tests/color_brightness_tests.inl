#ifndef PERCEPTRON_COLOR_BRIGHTNESS_TESTS_H
#define PERCEPTRON_COLOR_BRIGHTNESS_TESTS_H

#include <array>
#include <cstddef>
#include <iostream>
#include <random>
#include <utility>

#include "color_brightness_data_set.hpp"

namespace color_brightness {
	auto test_perceptron(
			Perceptron<4> perceptron,
			std::mt19937_64 &prng_generator
	) -> double {
		auto model_accuracy = 0.0;
		
		constexpr auto tests_count = std::size_t(1000);
		auto prng_distribution = std::uniform_real_distribution(0.0, 1.0);
		for (std::size_t i = 0; i < tests_count; ++i) {
			const auto test_color = std::array{
					1.0,
					prng_distribution(prng_generator),
					prng_distribution(prng_generator),
					prng_distribution(prng_generator),
			};
			const auto actual_class =
					test_color[1] + test_color[2] + test_color[3] > 1.5 ? 1 : 0;
			const auto computed_class = perceptron.classify(test_color);
			if (computed_class == actual_class)
				model_accuracy += 100.0 / tests_count;
			else
				std::cout << "Color: {" << test_color[1] << ", "
				          << test_color[2]
				          << ", " << test_color[3] << "}. Class: "
				          << (actual_class == 0 ? "dark" : "bright")
				          << ". Computed class: "
				          << (computed_class == 0 ? "dark" : "bright") << '.'
				          << std::endl;
		}
		
		return model_accuracy;
	}
	
	template<std::size_t N>
	constexpr auto train_perceptron_with_finite_set(
			std::array<std::pair<std::array<double, 4>, int>, N> training_set,
			double bias
	) -> Perceptron<4> {
		auto perceptron_classifier = Perceptron<4>(bias);
		train_perceptron_with_finite_set(training_set, perceptron_classifier);
		return perceptron_classifier;
	}
	
	auto finite_set_test(
			std::mt19937_64 &prng_generator
	) noexcept -> double {
		constexpr auto perceptron = train_perceptron_with_finite_set(
				data_set(),
				-2.5
		);
		return test_perceptron(perceptron, prng_generator);
	}
	
	auto dynamic_training_test(
			std::mt19937_64 &prng_generator
	) noexcept -> double {
		constexpr auto desired_accuracy = 99.5;
		
		auto perceptron = Perceptron<4>(-2.5);
		const auto iterations = train_perceptron_with_performance_goal(
				[](const std::array<double, 4> input) -> bool {
					return input[1] + input[2] + input[3] > 1.5;
				},
				desired_accuracy,
				std::uniform_real_distribution(0.0, 1.0),
				prng_generator,
				perceptron
		);
		std::cout << "The perceptron took " << iterations << " runs to reach "
		          << desired_accuracy << "% accuracy." << std::endl;
		return test_perceptron(perceptron, prng_generator);
	}
}

#endif //PERCEPTRON_COLOR_BRIGHTNESS_TESTS_H
