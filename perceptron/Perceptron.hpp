#ifndef PERCEPTRON_PERCEPTRON_HPP
#define PERCEPTRON_PERCEPTRON_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <deque>
#include <optional>
#include <queue>

template<std::size_t S>
class Perceptron;

template<std::size_t S, std::size_t N>
constexpr auto train_perceptron_with_finite_set(
		std::array<std::pair<typename Perceptron<S>::vector_type, int>, N> training_set,
		Perceptron<S> &perceptron
) -> void;

template<std::size_t S>
auto train_perceptron_with_performance_goal(
		const std::predicate<typename Perceptron<S>::vector_type> auto &target_relation,
		double target_percentage,
		std::mt19937_64 *prng_generator,
		Perceptron<S> &perceptron
) -> std::size_t;


template<std::size_t S>
class Perceptron {
public:
	static_assert(S >= 2, "A perceptron stores the bias as first element.");
	typedef std::array<double, S> vector_type;

private:
	vector_type m_weights;

public:
	constexpr explicit Perceptron() noexcept;
	
	constexpr explicit Perceptron(double bias) noexcept;
	
	constexpr auto classify(vector_type input) const noexcept -> int;
	
	constexpr auto learn(
			vector_type input,
			int input_class,
			double learning_rate
	) noexcept -> int;
};

template<std::size_t S>
constexpr Perceptron<S>::Perceptron() noexcept :
		m_weights() {
	std::ranges::fill(m_weights, 0.0);
}

template<std::size_t S>
constexpr Perceptron<S>::Perceptron(const double bias) noexcept :
		m_weights() {
	std::ranges::fill(m_weights, 0.0);
	m_weights[0] = bias;
}

template<std::size_t S>
constexpr auto Perceptron<S>::classify(
		const Perceptron<S>::vector_type input
) const noexcept -> int {
	double dot_product = 0.0;
	for (std::size_t i = 0; i < S; ++i) {
		dot_product += m_weights[i] * input[i];
	}
	if (dot_product > 0.0)
		return 1;
	else
		return 0;
}

template<std::size_t S>
constexpr auto Perceptron<S>::learn(
		const Perceptron<S>::vector_type input,
		const int input_class,
		const double learning_rate
) noexcept -> int {
	const int computed_class = this->classify(input);
	const int class_difference = input_class - computed_class;
	const double rated_difference = learning_rate * class_difference;
	for (std::size_t i = 0; i < S; ++i) {
		m_weights[i] += rated_difference * input[i];
	}
	return computed_class;
}


template<std::size_t S, std::size_t N>
constexpr auto train_perceptron_with_finite_set(
		const std::array<std::pair<typename Perceptron<S>::vector_type, int>, N> training_set,
		Perceptron<S> &perceptron
) -> void {
	for (const auto &pair: training_set) {
		perceptron.learn(pair.first, pair.second, 0.1);
	}
}

template<std::size_t S, std::size_t validation_sample_size>
auto learn_and_update_performance(
		Perceptron<S> &perceptron,
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
		Perceptron<S> &perceptron
) -> std::size_t {
	constexpr auto validation_sample_size = 1000;
	
	auto queue = std::queue(std::deque<int>());
	auto performance = 0.0;
	auto iteration = size_t(0);
	
	auto prng_distribution = std::uniform_real_distribution(0.0, 1.0);
	while (performance < target_percentage &&
	       queue.size() != validation_sample_size) {
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

#endif //PERCEPTRON_PERCEPTRON_HPP
