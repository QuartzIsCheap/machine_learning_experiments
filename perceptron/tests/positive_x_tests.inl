#ifndef PERCEPTRON_POSITIVE_X_TESTS_INL
#define PERCEPTRON_POSITIVE_X_TESTS_INL

#include <random>

#include "../perceptron.hpp"

namespace positive_x {
	template<std::size_t S>
	auto test_perceptron(
			const Perceptron<S> &perceptron,
			std::mt19937_64 &prng_generator
	) -> double {
		auto model_accuracy = 0.0;
		
		constexpr auto tests_count = std::size_t(1000);
		auto prng_distribution = std::uniform_real_distribution(-100.0, 100.0);
		for (std::size_t i = 0; i < tests_count; ++i) {
			auto test_point = typename Perceptron<S>::vector_type{1.0};
			for (std::size_t j = 1; j < S; ++j) {
				test_point[j] = prng_distribution(prng_generator);
			}
			const auto actual_class = test_point[1] > 0 ? 1 : 0;
			const auto computed_class = perceptron.classify(test_point);
			if (computed_class == actual_class)
				model_accuracy += 100.0 / tests_count;
			else
				std::cout << "Point: {" << test_point[1] << ", "
				          << test_point[2] << ", " << test_point[3] << ", "
				          << test_point[4] << ", " << test_point[5] << ", "
				          << test_point[6] << "}. Class: "
				          << (actual_class == 0 ? "negative" : "positive")
				          << ". Computed class: "
				          << (computed_class == 0 ? "negative" : "positive")
				          << '.' << std::endl;
		}
		
		return model_accuracy;
	}
	
	auto dynamic_training_test(
			std::mt19937_64 &prng_generator
	) noexcept -> double {
		constexpr auto desired_accuracy = 99.5;
		
		auto perceptron = Perceptron<7>(0.0);
		const auto iterations = train_perceptron_with_performance_goal(
				[](const std::array<double, 7> point) -> bool {
					return point[1] > 0;
				},
				desired_accuracy,
				std::uniform_real_distribution(-100.0, 100.0),
				prng_generator,
				perceptron
		);
		std::cout << "The perceptron took " << iterations << " runs to reach "
		          << desired_accuracy << "% accuracy." << std::endl;
		return test_perceptron(perceptron, prng_generator);
	}
}

#endif //PERCEPTRON_POSITIVE_X_TESTS_INL
