#include <iostream>

#include "tests.hpp"

int main() {
	auto exit_status = EXIT_SUCCESS;
	
	auto prng_generator = std::mt19937_64(std::random_device()());
	
	const auto compile_time_trained_color_brightness_accuracy =
			color_brightness_test_with_finite_set(
					prng_generator
			);
	std::cout << "Color brightness accuracy with fixed set training : "
	          << compile_time_trained_color_brightness_accuracy
	          << std::endl;
	if (compile_time_trained_color_brightness_accuracy < 98.0)
		exit_status = EXIT_FAILURE;
	
	std::cout << std::endl;
	
	const auto dynamically_trained_color_brightness_accuracy =
			color_brightness_test_with_dynamic_training(
					prng_generator
			);
	std::cout << "Color brightness accuracy with dynamic training : "
	          << dynamically_trained_color_brightness_accuracy
	          << std::endl;
	if (dynamically_trained_color_brightness_accuracy < 98.0)
		exit_status = EXIT_FAILURE;
	
	return exit_status;
}
