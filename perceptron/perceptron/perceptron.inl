#ifndef PERCEPTRON_PERCEPTRON_INL
#define PERCEPTRON_PERCEPTRON_INL

#include <algorithm>
#include <array>
#include <cstddef>

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

#endif //PERCEPTRON_PERCEPTRON_INL
