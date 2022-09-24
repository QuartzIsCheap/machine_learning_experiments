#ifndef PERCEPTRON_LEARNING_INL
#define PERCEPTRON_LEARNING_INL

#include <array>
#include <concepts>
#include <cstddef>
#include <deque>
#include <queue>
#include <random>

template<std::size_t S, std::size_t N>
constexpr auto train_perceptron_with_finite_set(
		const std::array<std::pair<typename Perceptron<S>::vector_type, int>, N> training_set,
		Perceptron <S> &perceptron
) -> void {
	for (const auto &pair: training_set) {
		perceptron.learn(pair.first, pair.second, 0.1);
	}
}

template<std::size_t S, std::size_t validation_sample_size>
auto learn_and_update_performance(
		Perceptron <S> &perceptron,
		const std::predicate<typename Perceptron<S>::vector_type> auto &target_relation,
		std::uniform_real_distribution<double> &prng_distribution,
		std::mt19937_64 &prng_generator,
		std::queue<int, std::deque<int>> &queue,
		double &performance
) -> void {
	const auto test_color = std::array{
			1.0,
			prng_distribution(prng_generator),
			prng_distribution(prng_generator),
			prng_distribution(prng_generator),
	};
	const auto actual_class = target_relation(test_color) ? 1 : 0;
	const auto computed_class = perceptron.learn(
			test_color,
			actual_class,
			0.1
	);
	queue.push(computed_class == actual_class ? 1 : 0);
	performance += queue.back() / (validation_sample_size / 100.0);
}

template<std::size_t S>
auto train_perceptron_with_performance_goal(
		const std::predicate<typename Perceptron<S>::vector_type> auto &target_relation,
		double target_percentage,
		std::mt19937_64 &prng_generator,
		Perceptron <S> &perceptron
) -> std::size_t {
	constexpr auto validation_sample_size = 1000;
	
	auto queue = std::queue(std::deque<int>());
	auto performance = 0.0;
	auto iteration = size_t(0);
	
	auto prng_distribution = std::uniform_real_distribution(0.0, 1.0);
	while (
			performance < target_percentage &&
			queue.size() != validation_sample_size
			) {
		learn_and_update_performance<S, validation_sample_size>(
				perceptron,
				target_relation,
				prng_distribution,
				prng_generator,
				queue,
				performance
		);
		iteration++;
	}
	while (performance < target_percentage) {
		performance -= queue.front() / (validation_sample_size / 100.0);
		queue.pop();
		learn_and_update_performance<S, validation_sample_size>(
				perceptron,
				target_relation,
				prng_distribution,
				prng_generator,
				queue,
				performance
		);
		iteration++;
	}
	
	return iteration;
}

#endif //PERCEPTRON_LEARNING_INL
