#include <cstdint>

namespace {

uint32_t xorshift32(uint32_t& state) {
  if (state == 0) {
    state = 0x9e3779b9u;
  }

  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

double next_random(uint32_t& state) {
  return static_cast<double>(xorshift32(state)) / 4294967296.0;
}

}  // namespace

extern "C" {

void run_policy_trials(
  int32_t trial_count,
  int32_t max_steps,
  int32_t start_state_index,
  uint32_t seed,
  const int32_t* terminal_flags,
  const double* weighted_costs,
  const double* module_costs,
  const double* lock_key_costs,
  const int32_t* transition_offsets,
  const int32_t* transition_next_states,
  const double* transition_cumulative_probabilities,
  double* output
) {
  double weighted_cost_sum = 0.0;
  double module_cost_sum = 0.0;
  double lock_key_cost_sum = 0.0;
  double total_steps = 0.0;
  double success_count = 0.0;
  uint32_t random_state = seed;

  for (int32_t trial = 0; trial < trial_count; ++trial) {
    int32_t state_index = start_state_index;

    for (int32_t step = 0; step < max_steps; ++step) {
      if (terminal_flags[state_index] == 1) {
        success_count += 1.0;
        total_steps += static_cast<double>(step);
        break;
      }

      weighted_cost_sum += weighted_costs[state_index];
      module_cost_sum += module_costs[state_index];
      lock_key_cost_sum += lock_key_costs[state_index];

      const int32_t transition_start = transition_offsets[state_index];
      const int32_t transition_end = transition_offsets[state_index + 1];
      const double draw = next_random(random_state);

      int32_t next_state_index = transition_next_states[transition_end - 1];
      for (int32_t transition_index = transition_start; transition_index < transition_end; ++transition_index) {
        if (draw <= transition_cumulative_probabilities[transition_index]) {
          next_state_index = transition_next_states[transition_index];
          break;
        }
      }

      state_index = next_state_index;
    }
  }

  output[0] = weighted_cost_sum;
  output[1] = module_cost_sum;
  output[2] = lock_key_cost_sum;
  output[3] = total_steps;
  output[4] = success_count;
}

}