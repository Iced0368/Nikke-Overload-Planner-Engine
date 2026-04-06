#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <emscripten.h>

namespace {

constexpr int MASK_COUNT = 7;
constexpr int KEY_COUNT_PER_STATE = 3;
constexpr int GRADE_MASK_COUNT = 8;
constexpr int MAX_GRADE_TRANSITIONS_PER_MASK = 8;
constexpr int MAX_OPTION_ACTION_CANDIDATES = 19;
constexpr int SLOT_VALUE_KEY_SIZE = 20;
constexpr int TWO_SLOT_KEY_SIZE = SLOT_VALUE_KEY_SIZE * SLOT_VALUE_KEY_SIZE;

constexpr std::array<int, MASK_COUNT> SINGLE_MASK_INDEX = {-1, 0, 1, -1, 2, -1, -1};
constexpr std::array<int, MASK_COUNT> PAIR_MASK_INDEX = {-1, -1, -1, 0, -1, 1, 2};
constexpr std::array<int, 3> SINGLE_WEIGHT_INDEX_BY_SLOT = {1, 2, 4};
constexpr std::array<int, 3> PAIR_WEIGHT_INDEX_BY_SLOT = {3, 5, 6};

struct AggregateSums {
  std::array<std::vector<double>, 3> single_prob_sums;
  std::array<std::vector<double>, 3> single_dist_sums;
  std::array<std::vector<double>, 3> pair_prob_sums;
  std::array<std::vector<double>, 3> pair_dist_sums;
  double all_prob_sum = 0.0;
  double all_dist_sum = 0.0;

  AggregateSums() {
    for (int slot = 0; slot < 3; ++slot) {
      single_prob_sums[slot].assign(SLOT_VALUE_KEY_SIZE, 0.0);
      single_dist_sums[slot].assign(SLOT_VALUE_KEY_SIZE, 0.0);
      pair_prob_sums[slot].assign(TWO_SLOT_KEY_SIZE, 0.0);
      pair_dist_sums[slot].assign(TWO_SLOT_KEY_SIZE, 0.0);
    }
  }

  void reset() {
    all_prob_sum = 0.0;
    all_dist_sum = 0.0;
    for (int slot = 0; slot < 3; ++slot) {
      std::fill(single_prob_sums[slot].begin(), single_prob_sums[slot].end(), 0.0);
      std::fill(single_dist_sums[slot].begin(), single_dist_sums[slot].end(), 0.0);
      std::fill(pair_prob_sums[slot].begin(), pair_prob_sums[slot].end(), 0.0);
      std::fill(pair_dist_sums[slot].begin(), pair_dist_sums[slot].end(), 0.0);
    }
  }
};

struct SumPair {
  double prob_sum;
  double dist_sum;
};

SumPair get_aggregated_option_transition_sums(
  int protected_mask,
  int state_index,
  const int32_t* single_compatibility_keys,
  const int32_t* pair_compatibility_keys,
  const AggregateSums& aggregates
) {
  const int key_offset = state_index * KEY_COUNT_PER_STATE;
  if (protected_mask == 0) {
    return {aggregates.all_prob_sum, aggregates.all_dist_sum};
  }

  const int single_slot = SINGLE_MASK_INDEX[protected_mask];
  if (single_slot != -1) {
    const int key = single_compatibility_keys[key_offset + single_slot];
    return {
      aggregates.single_prob_sums[single_slot][key],
      aggregates.single_dist_sums[single_slot][key],
    };
  }

  const int pair_slot = PAIR_MASK_INDEX[protected_mask];
  if (pair_slot == -1) {
    return {0.0, 0.0};
  }

  const int key = pair_compatibility_keys[key_offset + pair_slot];
  return {
    aggregates.pair_prob_sums[pair_slot][key],
    aggregates.pair_dist_sums[pair_slot][key],
  };
}

}  // namespace

EM_JS(void, optimizer_progress, (int phase, int completed, int total, double percent), {
  if (typeof globalThis.__optimizerWasmProgress === "function") {
    globalThis.__optimizerWasmProgress(phase, completed, total, percent);
  }
});

extern "C" {

void run_optimizer_core(
  int32_t state_count,
  int32_t iterations,
  int32_t progress_every_iterations,
  double infinite_cost,
  const int8_t* state_module_masks,
  const int32_t* option_triple_keys_by_state,
  const int32_t* single_compatibility_keys,
  const int32_t* pair_compatibility_keys,
  const int32_t* next_state_index_by_module_mask_and_grade_mask,
  const uint8_t* option_target_match,
  const double* option_transition_weights_by_state,
  const uint8_t* grade_transition_counts_by_state_and_mask,
  const uint8_t* grade_transition_grade_masks_by_state_and_mask,
  const double* grade_transition_probabilities_by_state_and_mask,
  const double* weighted_action_cost_by_mask_triplet,
  const double* module_action_cost_by_mask_triplet,
  const double* lock_key_action_cost_by_mask_triplet,
  const uint8_t* option_candidate_counts_by_state,
  const int8_t* option_candidate_next_module_masks_by_state,
  const int8_t* option_candidate_protected_masks_by_state,
  const int8_t* option_candidate_key_masks_by_state,
  const double* option_candidate_probability_masses_by_state,
  double* costs,
  double* expected_module_costs,
  double* expected_lock_key_costs,
  int8_t* action_type_by_state,
  int8_t* action_module_mask_by_state,
  int8_t* action_key_mask_by_state,
  int32_t* iterations_run_out
) {
  std::vector<int32_t> ordered_state_indexes_scratch(state_count);
  for (int32_t state_index = 0; state_index < state_count; ++state_index) {
    ordered_state_indexes_scratch[state_index] = state_index;
  }

  std::array<std::vector<int32_t>, MASK_COUNT> ordered_state_indexes_by_module_mask;
  for (auto& ordered_state_indexes : ordered_state_indexes_by_module_mask) {
    ordered_state_indexes.assign(state_count, 0);
  }
  std::array<int32_t, MASK_COUNT> ordered_state_counts_by_module_mask = {};
  std::array<AggregateSums, MASK_COUNT> option_snapshot_aggregates_by_module_mask;
  std::array<int32_t, MASK_COUNT> option_snapshot_candidate_pointers = {};
  std::vector<double> best_option_cost_scratch(state_count, infinite_cost);
  std::vector<int8_t> best_option_module_mask_scratch(state_count, -1);
  std::vector<int8_t> best_option_key_mask_scratch(state_count, -1);

  auto build_action_cost_index = [](int current_module_mask, int next_module_mask, int key_mask) {
    return (current_module_mask * MASK_COUNT + next_module_mask) * MASK_COUNT + key_mask;
  };

  int32_t iterations_run = 0;
  for (int32_t iteration = 0; iteration < iterations; ++iteration) {
    iterations_run = iteration + 1;
    std::vector<double> snapshot_costs(costs, costs + state_count);
    std::sort(
      ordered_state_indexes_scratch.begin(),
      ordered_state_indexes_scratch.end(),
      [&](int32_t left, int32_t right) { return snapshot_costs[left] < snapshot_costs[right]; }
    );

    ordered_state_counts_by_module_mask.fill(0);
    option_snapshot_candidate_pointers.fill(0);
    std::fill(best_option_cost_scratch.begin(), best_option_cost_scratch.end(), infinite_cost);
    std::fill(best_option_module_mask_scratch.begin(), best_option_module_mask_scratch.end(), -1);
    std::fill(best_option_key_mask_scratch.begin(), best_option_key_mask_scratch.end(), -1);
    for (auto& aggregates : option_snapshot_aggregates_by_module_mask) {
      aggregates.reset();
    }

    for (int32_t ordered_state_index : ordered_state_indexes_scratch) {
      const int module_mask = state_module_masks[ordered_state_index];
      ordered_state_indexes_by_module_mask[module_mask][ordered_state_counts_by_module_mask[module_mask]] = ordered_state_index;
      ordered_state_counts_by_module_mask[module_mask] += 1;
    }

    auto add_candidate = [&](int module_mask, int32_t candidate_index) {
      auto& aggregates = option_snapshot_aggregates_by_module_mask[module_mask];
      const int weight_offset = candidate_index * MASK_COUNT;
      const int key_offset = candidate_index * KEY_COUNT_PER_STATE;
      const double candidate_cost = snapshot_costs[candidate_index];

      aggregates.all_prob_sum += option_transition_weights_by_state[weight_offset];
      aggregates.all_dist_sum += option_transition_weights_by_state[weight_offset] * candidate_cost;

      for (int slot = 0; slot < 3; ++slot) {
        const int single_key = single_compatibility_keys[key_offset + slot];
        const int weight_index = SINGLE_WEIGHT_INDEX_BY_SLOT[slot];
        const double weight = option_transition_weights_by_state[weight_offset + weight_index];
        aggregates.single_prob_sums[slot][single_key] += weight;
        aggregates.single_dist_sums[slot][single_key] += weight * candidate_cost;
      }

      for (int pair_index = 0; pair_index < 3; ++pair_index) {
        const int pair_key = pair_compatibility_keys[key_offset + pair_index];
        const int weight_index = PAIR_WEIGHT_INDEX_BY_SLOT[pair_index];
        const double weight = option_transition_weights_by_state[weight_offset + weight_index];
        aggregates.pair_prob_sums[pair_index][pair_key] += weight;
        aggregates.pair_dist_sums[pair_index][pair_key] += weight * candidate_cost;
      }
    };

    for (int32_t ordered_state_index : ordered_state_indexes_scratch) {
      const double current_cost = snapshot_costs[ordered_state_index];

      for (int module_mask = 0; module_mask < MASK_COUNT; ++module_mask) {
        auto& ordered_candidates = ordered_state_indexes_by_module_mask[module_mask];
        int32_t candidate_pointer = option_snapshot_candidate_pointers[module_mask];
        while (
          candidate_pointer < ordered_state_counts_by_module_mask[module_mask] &&
          snapshot_costs[ordered_candidates[candidate_pointer]] < current_cost - 1e-4
        ) {
          add_candidate(module_mask, ordered_candidates[candidate_pointer]);
          candidate_pointer += 1;
        }
        option_snapshot_candidate_pointers[module_mask] = candidate_pointer;
      }

      const int current_module_mask = state_module_masks[ordered_state_index];
      const int candidate_offset = ordered_state_index * MAX_OPTION_ACTION_CANDIDATES;
      const int candidate_count = option_candidate_counts_by_state[ordered_state_index];

      for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
        const int option_candidate_index = candidate_offset + candidate_index;
        const int next_module_mask = option_candidate_next_module_masks_by_state[option_candidate_index];
        const int protected_mask = option_candidate_protected_masks_by_state[option_candidate_index];
        const int key_mask = option_candidate_key_masks_by_state[option_candidate_index];
        const double probability_mass = option_candidate_probability_masses_by_state[option_candidate_index];
        const auto sums = get_aggregated_option_transition_sums(
          protected_mask,
          ordered_state_index,
          single_compatibility_keys,
          pair_compatibility_keys,
          option_snapshot_aggregates_by_module_mask[next_module_mask]
        );

        const double next_cost =
          sums.dist_sum / probability_mass +
          (1.0 - sums.prob_sum / probability_mass) * current_cost +
          weighted_action_cost_by_mask_triplet[build_action_cost_index(current_module_mask, next_module_mask, key_mask)];

        if (next_cost < best_option_cost_scratch[ordered_state_index] - 1e-4) {
          best_option_cost_scratch[ordered_state_index] = next_cost;
          best_option_module_mask_scratch[ordered_state_index] = static_cast<int8_t>(next_module_mask);
          best_option_key_mask_scratch[ordered_state_index] = static_cast<int8_t>(key_mask);
        }
      }
    }

    double total_improvement = 0.0;
    for (int32_t state_index = 0; state_index < state_count; ++state_index) {
      const int current_module_mask = state_module_masks[state_index];
      const double current_cost = costs[state_index];
      if (action_type_by_state[state_index] == 0) {
        continue;
      }

      if (option_target_match[option_triple_keys_by_state[state_index]] != 0) {
        double current_best_cost = current_cost;
        const int candidate_offset = state_index * MAX_OPTION_ACTION_CANDIDATES;
        const int candidate_count = option_candidate_counts_by_state[state_index];

        for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
          const int option_candidate_index = candidate_offset + candidate_index;
          const int next_module_mask = option_candidate_next_module_masks_by_state[option_candidate_index];
          const int protected_mask = option_candidate_protected_masks_by_state[option_candidate_index];
          const int key_mask = option_candidate_key_masks_by_state[option_candidate_index];
          double next_cost = weighted_action_cost_by_mask_triplet[
            build_action_cost_index(current_module_mask, next_module_mask, key_mask)
          ];
          const int transition_offset = (state_index * MASK_COUNT + protected_mask) * MAX_GRADE_TRANSITIONS_PER_MASK;
          const int transition_count = grade_transition_counts_by_state_and_mask[state_index * MASK_COUNT + protected_mask];
          const int next_state_offset = state_index * MASK_COUNT * GRADE_MASK_COUNT + next_module_mask * GRADE_MASK_COUNT;

          for (int transition_index = 0; transition_index < transition_count; ++transition_index) {
            const int grade_mask = grade_transition_grade_masks_by_state_and_mask[transition_offset + transition_index];
            const double prob = grade_transition_probabilities_by_state_and_mask[transition_offset + transition_index];
            const int next_state_index = next_state_index_by_module_mask_and_grade_mask[next_state_offset + grade_mask];
            next_cost += prob * std::min(costs[next_state_index], current_best_cost);
          }

          const double improvement = current_best_cost - next_cost;
          if (improvement > 1e-4) {
            total_improvement += improvement;
            current_best_cost = next_cost;
            costs[state_index] = next_cost;
            action_type_by_state[state_index] = 2;
            action_module_mask_by_state[state_index] = static_cast<int8_t>(next_module_mask);
            action_key_mask_by_state[state_index] = static_cast<int8_t>(key_mask);
          }
        }
      }

      const double best_next_option_cost = best_option_cost_scratch[state_index];
      const int best_module_mask = best_option_module_mask_scratch[state_index];
      const int best_key_mask = best_option_key_mask_scratch[state_index];
      const double improvement = costs[state_index] - best_next_option_cost;
      if (best_module_mask != -1 && best_key_mask != -1 && improvement > 1e-4) {
        total_improvement += improvement;
        costs[state_index] = best_next_option_cost;
        action_type_by_state[state_index] = 1;
        action_module_mask_by_state[state_index] = static_cast<int8_t>(best_module_mask);
        action_key_mask_by_state[state_index] = static_cast<int8_t>(best_key_mask);
      }
    }

    if (total_improvement == 0.0) {
      if (progress_every_iterations > 0) {
        optimizer_progress(0, iterations_run, iterations, (static_cast<double>(iterations_run) / iterations) * 80.0);
      }
      break;
    }

    if (
      progress_every_iterations > 0 &&
      (iteration == iterations - 1 || iteration % progress_every_iterations == 0)
    ) {
      optimizer_progress(0, iterations_run, iterations, (static_cast<double>(iterations_run) / iterations) * 80.0);
    }
  }

  std::array<AggregateSums, MASK_COUNT> expectation_module_aggregates_by_module_mask;
  std::array<AggregateSums, MASK_COUNT> expectation_lock_key_aggregates_by_module_mask;

  for (int evaluation_iteration = 0; evaluation_iteration < iterations; ++evaluation_iteration) {
    for (int module_mask = 0; module_mask < MASK_COUNT; ++module_mask) {
      expectation_module_aggregates_by_module_mask[module_mask].reset();
      expectation_lock_key_aggregates_by_module_mask[module_mask].reset();
    }

    for (int32_t state_index = 0; state_index < state_count; ++state_index) {
      const int module_mask = state_module_masks[state_index];
      auto& module_aggregates = expectation_module_aggregates_by_module_mask[module_mask];
      auto& lock_key_aggregates = expectation_lock_key_aggregates_by_module_mask[module_mask];
      const int weight_offset = state_index * MASK_COUNT;
      const int key_offset = state_index * KEY_COUNT_PER_STATE;
      const double module_value = expected_module_costs[state_index];
      const double lock_key_value = expected_lock_key_costs[state_index];

      module_aggregates.all_prob_sum += option_transition_weights_by_state[weight_offset];
      module_aggregates.all_dist_sum += option_transition_weights_by_state[weight_offset] * module_value;
      lock_key_aggregates.all_prob_sum += option_transition_weights_by_state[weight_offset];
      lock_key_aggregates.all_dist_sum += option_transition_weights_by_state[weight_offset] * lock_key_value;

      for (int slot = 0; slot < 3; ++slot) {
        const int single_key = single_compatibility_keys[key_offset + slot];
        const int weight_index = SINGLE_WEIGHT_INDEX_BY_SLOT[slot];
        const double weight = option_transition_weights_by_state[weight_offset + weight_index];
        module_aggregates.single_prob_sums[slot][single_key] += weight;
        module_aggregates.single_dist_sums[slot][single_key] += weight * module_value;
        lock_key_aggregates.single_prob_sums[slot][single_key] += weight;
        lock_key_aggregates.single_dist_sums[slot][single_key] += weight * lock_key_value;
      }

      for (int pair_index = 0; pair_index < 3; ++pair_index) {
        const int pair_key = pair_compatibility_keys[key_offset + pair_index];
        const int weight_index = PAIR_WEIGHT_INDEX_BY_SLOT[pair_index];
        const double weight = option_transition_weights_by_state[weight_offset + weight_index];
        module_aggregates.pair_prob_sums[pair_index][pair_key] += weight;
        module_aggregates.pair_dist_sums[pair_index][pair_key] += weight * module_value;
        lock_key_aggregates.pair_prob_sums[pair_index][pair_key] += weight;
        lock_key_aggregates.pair_dist_sums[pair_index][pair_key] += weight * lock_key_value;
      }
    }

    double max_delta = 0.0;
    for (int32_t state_index = 0; state_index < state_count; ++state_index) {
      const int action_type = action_type_by_state[state_index];
      if (action_type == 0) {
        expected_module_costs[state_index] = 0.0;
        expected_lock_key_costs[state_index] = 0.0;
        continue;
      }

      const int current_module_mask = state_module_masks[state_index];
      const int next_module_mask = action_module_mask_by_state[state_index];
      const int key_mask = action_key_mask_by_state[state_index];
      const int protected_mask = next_module_mask | key_mask;
      const int action_cost_index = build_action_cost_index(current_module_mask, next_module_mask, key_mask);
      double next_expected_module_cost = module_action_cost_by_mask_triplet[action_cost_index];
      double next_expected_lock_key_cost = lock_key_action_cost_by_mask_triplet[action_cost_index];

      if (action_type == 2) {
        const int transition_offset = (state_index * MASK_COUNT + protected_mask) * MAX_GRADE_TRANSITIONS_PER_MASK;
        const int transition_count = grade_transition_counts_by_state_and_mask[state_index * MASK_COUNT + protected_mask];
        const int next_state_offset = state_index * MASK_COUNT * GRADE_MASK_COUNT + next_module_mask * GRADE_MASK_COUNT;
        for (int transition_index = 0; transition_index < transition_count; ++transition_index) {
          const int grade_mask = grade_transition_grade_masks_by_state_and_mask[transition_offset + transition_index];
          const double prob = grade_transition_probabilities_by_state_and_mask[transition_offset + transition_index];
          const int next_state_index = next_state_index_by_module_mask_and_grade_mask[next_state_offset + grade_mask];
          next_expected_module_cost += prob * expected_module_costs[next_state_index];
          next_expected_lock_key_cost += prob * expected_lock_key_costs[next_state_index];
        }
      } else {
        const int candidate_offset = state_index * MAX_OPTION_ACTION_CANDIDATES;
        int matching_candidate_index = -1;
        for (int candidate_index = 0; candidate_index < option_candidate_counts_by_state[state_index]; ++candidate_index) {
          const int option_candidate_index = candidate_offset + candidate_index;
          if (
            option_candidate_next_module_masks_by_state[option_candidate_index] == next_module_mask &&
            option_candidate_protected_masks_by_state[option_candidate_index] == protected_mask &&
            option_candidate_key_masks_by_state[option_candidate_index] == key_mask
          ) {
            matching_candidate_index = option_candidate_index;
            break;
          }
        }

        if (matching_candidate_index != -1) {
          const double probability_mass = option_candidate_probability_masses_by_state[matching_candidate_index];
          const auto module_sums = get_aggregated_option_transition_sums(
            protected_mask,
            state_index,
            single_compatibility_keys,
            pair_compatibility_keys,
            expectation_module_aggregates_by_module_mask[next_module_mask]
          );
          const auto lock_key_sums = get_aggregated_option_transition_sums(
            protected_mask,
            state_index,
            single_compatibility_keys,
            pair_compatibility_keys,
            expectation_lock_key_aggregates_by_module_mask[next_module_mask]
          );
          next_expected_module_cost += module_sums.dist_sum / probability_mass;
          next_expected_lock_key_cost += lock_key_sums.dist_sum / probability_mass;
        }
      }

      max_delta = std::max(max_delta, std::abs(next_expected_module_cost - expected_module_costs[state_index]));
      max_delta = std::max(max_delta, std::abs(next_expected_lock_key_cost - expected_lock_key_costs[state_index]));
      expected_module_costs[state_index] = next_expected_module_cost;
      expected_lock_key_costs[state_index] = next_expected_lock_key_cost;
    }

    if (max_delta < 1e-9) {
      if (progress_every_iterations > 0) {
        optimizer_progress(
          1,
          evaluation_iteration + 1,
          iterations,
          80.0 + (static_cast<double>(evaluation_iteration + 1) / iterations) * 20.0
        );
      }
      break;
    }

    if (
      progress_every_iterations > 0 &&
      (evaluation_iteration == iterations - 1 || evaluation_iteration % progress_every_iterations == 0)
    ) {
      optimizer_progress(
        1,
        evaluation_iteration + 1,
        iterations,
        80.0 + (static_cast<double>(evaluation_iteration + 1) / iterations) * 20.0
      );
    }
  }

  *iterations_run_out = iterations_run;
}

}