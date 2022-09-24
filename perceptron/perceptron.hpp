#ifndef PERCEPTRON_PERCEPTRON_HPP
#define PERCEPTRON_PERCEPTRON_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <random>
#include <utility>

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

#include "perceptron/perceptron.inl"
#include "perceptron/learning.inl"

#endif //PERCEPTRON_PERCEPTRON_HPP
