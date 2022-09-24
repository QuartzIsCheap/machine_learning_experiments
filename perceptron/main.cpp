#include <concepts>
#include <iostream>
#include <random>

#include "tests.hpp"

template<typename T>
concept test_function = requires(const T &f, std::mt19937_64 &g) {
	{ f(g) } -> std::same_as<double>;
};

auto run_test(
		const test_function auto &test,
		std::mt19937_64 &prng_generator,
		const char *test_message
) -> bool {
	const auto test_accuracy = test(prng_generator);
	std::cout << test_message << " : " << test_accuracy << "%." << std::endl;
	return test_accuracy > 98.0;
}

int main() {
	auto exit_status = EXIT_SUCCESS;
	
	auto prng_generator = std::mt19937_64(std::random_device()());
	
	if (!run_test(
			color_brightness::finite_set_test,
			prng_generator,
			"Color brightness test with fixed set training"
	))
		exit_status = EXIT_FAILURE;
	
	std::cout << std::endl;
	
	if (!run_test(
			color_brightness::dynamic_training_test,
			prng_generator,
			"Color brightness test with dynamic training"
	))
		exit_status = EXIT_FAILURE;
	
	return exit_status;
}
